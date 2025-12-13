import os
import yaml
import torch
import gc
import re
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# --- 匯入 Dataset ---
# 假設你的 dataset/iconqa.py 已經準備好 (如前面對話所示)
from dataset.iconqa import IconQADataset
# 匯入 collate_fn 和 load_model_and_processor
from dataset.utils import load_model_and_processor, collate_fn 

# =================================================================
# 1. CCoT 提示詞模板 (針對 IconQA 優化)
# =================================================================

CCOT_PROMPTS = {
    # -----------------------------------------------------------------
    # Qwen-VL (ChatML 格式)
    # -----------------------------------------------------------------
    "qwen-vl": {
        'stage_1_system': "You are an expert visual reasoning assistant.",
        # Stage 1: 生成 Scene Graph (通用)
        'stage_1_user': """For the provided image and its associated question, generate a scene graph in JSON format.
Analyze the abstract icon, diagram, or chart and extract:
1. Key visual elements (objects/symbols) relevant to the question.
2. Attributes of these elements (color, shape, text, count).
3. Spatial or semantic relationships between them.

Question: {question}
Scene Graph:""", 

        'stage_2_system': "You are an expert in solving visual puzzles and icon questions.",
        
        # Stage 2 (MCQ): 多選題提示
        'stage_2_mcq': """Use the image and the following Scene Graph to reason and answer the multiple-choice question.
Scene Graph: {scene_graph}

Question: {question}
Options:
{options_str}

Reason step-by-step, then provide the final answer as a single option letter (A, B, C, etc.) in the format 'Answer: [Letter]'.
Answer:""",

        # Stage 2 (VQA): 開放題提示
        'stage_2_vqa': """Use the image and the following Scene Graph to reason and answer the question.
Scene Graph: {scene_graph}

Question: {question}

Provide a short, direct answer (single word or number).
Answer:"""
    },
    
    # -----------------------------------------------------------------
    # LLaVA (USER: ... ASSISTANT: 格式)
    # -----------------------------------------------------------------
    "llava": {
        # Stage 1: 生成 Scene Graph
        'stage_1_prompt': """USER: <image>
For the provided image and its associated question, generate a scene graph in JSON format.
Analyze the abstract icon, diagram, or chart and extract:
1. Key visual elements (objects/symbols) relevant to the question.
2. Attributes of these elements (color, shape, text, count).
3. Spatial or semantic relationships between them.

Question: {question}
ASSISTANT: Scene Graph:""",

        # Stage 2 (MCQ): 多選題提示
        'stage_2_mcq': """[INST] <image>
Use the image and the following Scene Graph to reason and answer the multiple-choice question.
Scene Graph: {scene_graph}

Question: {question}
Options:
{options_str}

Reason step-by-step, then provide the final answer as a single option letter (A, B, C, etc.) in the format 'Answer: [Letter]'.
ASSISTANT: Answer:[/INST]""",

        # Stage 2 (VQA): 開放題提示
        'stage_2_vqa': """[INST] <image>
Use the image and the following Scene Graph to reason and answer the question.
Scene Graph: {scene_graph}

Question: {question}

Provide a short, direct answer (single word or number).
ASSISTANT: Answer:[/INST]"""
    }
}

# =================================================================
# 2. 輔助函數 (解析與輸入構建)
# =================================================================

def parse_scene_graph(text):
    """ 從 Stage 1 輸出中提取 Scene Graph """
    try:
        match = re.search(r"Scene Graph:(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            sg = match.group(1).strip()
        else:
            sg = text.strip()
        if sg.startswith("ASSISTANT:"):
            sg = sg.split("ASSISTANT:", 1)[-1].strip()
        return sg if sg else "N/A"
    except:
        return "PARSE_ERROR"

def parse_final_answer_mc(text):
    """ 解析 MCQ 答案 (提取 A/B/C) """
    # 1. 找 "Answer: [A-E]"
    match = re.search(r"Answer:\s*([A-E])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    # 2. 找最後出現的 A-E
    matches = re.findall(r"\b([A-E])\b", text, re.IGNORECASE)
    if matches: return matches[-1].upper()
    return ""

def parse_final_answer_vqa(text):
    """ 解析 VQA 答案 (短語) """
    # 1. 找 "Answer: ..." 後面的內容
    match = re.search(r"Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        ans = match.group(1).strip()
    else:
        ans = text.strip()
    # 只取第一行或第一個詞
    ans = ans.split('\n')[0].strip()
    # 移除句號
    return ans.rstrip('.')

def _format_options(choices):
    """ 將選項列表轉為字串 (A. xxx\nB. xxx) """
    output = ""
    for i, choice in enumerate(choices):
        output += f"{chr(65+i)}. {choice}\n"
    return output.strip()

def _build_ccot_inputs(prompts, mtype, stage, images, questions, choices_batch=None, scene_graphs=None, q_types=None):
    """ 構建 CCoT 輸入參數 """
    batch_size = len(questions)
    
    # --- Qwen-VL ---
    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            if stage == "stage_1":
                # Stage 1: 只看圖和問題生成 SG
                prompt_text = prompts['stage_1_user'].format(question=questions[i])
                messages = [
                    {"role": "system", "content": prompts['stage_1_system']}, 
                    {"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": prompt_text}]}
                ]
            
            elif stage == "stage_2":
                # Stage 2: 根據 SG 回答
                if q_types[i] == 'multiple-choice':
                    options_str = _format_options(choices_batch[i])
                    prompt_text = prompts['stage_2_mcq'].format(
                        scene_graph=scene_graphs[i],
                        question=questions[i],
                        options_str=options_str
                    )
                else: # open-ended
                    prompt_text = prompts['stage_2_vqa'].format(
                        scene_graph=scene_graphs[i],
                        question=questions[i]
                    )
                
                messages = [
                    {"role": "system", "content": prompts['stage_2_system']}, 
                    {"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": prompt_text}]}
                ]
            
            messages_list.append(messages)
        return {"messages_list": messages_list}

    # --- LLaVA ---
    elif mtype == "llava" or mtype == "llava-next":
        prompt_list = []
        for i in range(batch_size):
            if stage == "stage_1":
                prompt = prompts['stage_1_prompt'].format(question=questions[i])
            elif stage == "stage_2":
                if q_types[i] == 'multiple-choice':
                    options_str = _format_options(choices_batch[i])
                    prompt = prompts['stage_2_mcq'].format(
                        scene_graph=scene_graphs[i],
                        question=questions[i],
                        options_str=options_str
                    )
                else:
                    prompt = prompts['stage_2_vqa'].format(
                        scene_graph=scene_graphs[i],
                        question=questions[i]
                    )
            prompt_list.append(prompt)
        return {"prompt_list": prompt_list}
        
    else:
        raise NotImplementedError(f"mtype {mtype} not supported yet.")

# =================================================================
# 3. 通用生成函數 (沿用 VQA 版本)
# =================================================================
def ccot_generate_answer(proc, m, mtype, images, questions, messages_list=None, prompt_list=None, **kwargs):
    # 這裡直接復用您在 vqa_v2 中使用的 ccot_generate_answer 函數
    # 為節省篇幅，請確保您的 vqa_v2 版本代碼中的此函數包含在此處
    # ... (請貼上 ccot_generate_answer 的完整代碼) ...
    
    # 以下為簡化版示意 (LLaVA-NeXT)
    if mtype == "llava-next":
        if prompt_list is None: prompt_list = [""] * len(images)
        # 注意: LLaVA-NeXT 需要 [INST] 格式，請確認 CCOT_PROMPTS 或這裡有處理
        # 這裡假設 prompt_list 已經格式化好
        inputs = proc(images=images, text=prompt_list, return_tensors="pt", padding=True).to(m.device)
        input_len = inputs['input_ids'].shape[1]
        kwargs.pop('prompt_list', None); kwargs.pop('messages_list', None)
        with torch.no_grad():
            outputs = m.generate(**inputs, **kwargs)
        new_tokens = outputs[:, input_len:]
        answers = proc.batch_decode(new_tokens, skip_special_tokens=True)
        return [ans.strip() for ans in answers]

    # ... (請確保 Qwen-VL 等邏輯也包含在內) ...
    return [""] * len(images)

# =================================================================
# 4. 評分函數
# =================================================================
def score_mc_acc(pred, refs):
    if not pred: return 0.0
    pred = pred.strip().upper() 
    refs = [a.strip().upper() for a in refs]
    return 1.0 if pred in refs else 0.0

def score_vqa_acc(pred, refs):
    if not pred: return 0.0
    pred = pred.strip().lower()
    refs = [a.strip().lower() for a in refs]
    # IconQA 的開放題通常是數字或簡單詞，精確匹配即可
    return 1.0 if pred in refs else 0.0

# =================================================================
# 5. 主程式
# =================================================================
def main():
    if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
         os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
         
    config_dir = "configs"
    config_filename = 'iconqa.yaml' # 使用 IconQA 的設定檔
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    print(f"Running CCoT Evaluation on IconQA...")
    
    # 載入 Dataset
    ds = IconQADataset(
        dataset_id=cfg.get("dataset_id", "lmms-lab/ICON-QA"),
        split="validation", 
        num_samples=cfg.get("num_val_samples", None)
    )
    
    batch_size = cfg.get('batch_size', 1)
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_results = {}
    
    # 參數設定
    STAGE1_MAX_TOKENS = 300 # IconQA 的結構可能較複雜，給多一點空間
    STAGE2_MAX_TOKENS = 50  # 答案通常很短

    for mdl in cfg["models"]:
        name = mdl["name"]
        print(f"\nProcessing model: {name}")
        
        # 檢查是否有定義 Prompt
        base_type = mdl['type']
        if base_type == 'llava-next': base_type = 'llava' # 共用 prompt
        
        if base_type not in CCOT_PROMPTS:
            print(f"Warning: No CCoT prompts for '{base_type}'. Skipping.")
            continue
            
        prompts = CCOT_PROMPTS[base_type]
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        mdl_obj.eval()
        
        scores_mc, scores_vqa = [], []
        preds = []

        gen_params_stage1 = {"max_new_tokens": STAGE1_MAX_TOKENS, "do_sample": False}
        gen_params_stage2 = {"max_new_tokens": STAGE2_MAX_TOKENS, "do_sample": False}

        for batch_data in tqdm(ds_loader, desc=name):
            
            images = batch_data["image"]
            questions = batch_data["question"]
            all_answers = batch_data["answers"]
            question_ids = batch_data["question_id"]
            choices_batch = batch_data["choices"]
            q_types = batch_data["question_type"] # IconQADataset 需回傳此欄位
            
            # --- STAGE 1: 生成 Scene Graph ---
            # 注意: Stage 1 不區分題型，都只需要圖和問題
            stage1_inputs = _build_ccot_inputs(
                prompts, base_type, "stage_1", images, questions
            )
            
            stage1_outputs = ccot_generate_answer(
                proc, mdl_obj, mtype,
                images, questions,
                **stage1_inputs,
                **gen_params_stage1
            )
            
            generated_sgs = [parse_scene_graph(out) for out in stage1_outputs]
            
            # --- STAGE 2: 回答問題 ---
            # 這裡需要傳入 choices 和 q_types 來區分 Prompt
            stage2_inputs = _build_ccot_inputs(
                prompts, base_type, "stage_2", 
                images, questions, 
                choices_batch=choices_batch, 
                scene_graphs=generated_sgs,
                q_types=q_types
            )
            
            stage2_outputs = ccot_generate_answer(
                proc, mdl_obj, mtype,
                images, questions,
                **stage2_inputs,
                **gen_params_stage2
            )
            
            # --- 解析與計分 ---
            for i in range(len(stage2_outputs)):
                raw_ans = stage2_outputs[i]
                q_type = q_types[i]
                refs = all_answers[i]
                
                score = 0.0
                parsed_ans = ""
                
                if q_type == 'multiple-choice':
                    parsed_ans = parse_final_answer_mc(raw_ans)
                    score = score_mc_acc(parsed_ans, refs)
                    scores_mc.append(score)
                else:
                    parsed_ans = parse_final_answer_vqa(raw_ans)
                    score = score_vqa_acc(parsed_ans, refs)
                    scores_vqa.append(score)
                    
                preds.append({
                    "qid": question_ids[i],
                    "type": q_type,
                    "pred": parsed_ans,
                    "raw_output": raw_ans,
                    "refs": refs,
                    "score": score,
                    "scene_graph": generated_sgs[i]
                })

        # --- 統計 ---
        mean_mc = sum(scores_mc)/len(scores_mc) if scores_mc else 0
        mean_vqa = sum(scores_vqa)/len(scores_vqa) if scores_vqa else 0
        mean_all = (sum(scores_mc)+sum(scores_vqa)) / (len(scores_mc)+len(scores_vqa)) if (scores_mc or scores_vqa) else 0
        
        all_results[name] = {
            "acc_overall": mean_all,
            "acc_mc": mean_mc, 
            "acc_vqa": mean_vqa,
            "predictions": preds
        }
        
        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    # --- 輸出 ---
    print("\n" + "="*30)
    print("Final CCoT Results (IconQA):")
    for k, v in all_results.items():
        print(f"{k}: Overall={v['acc_overall']:.4f}, MC={v['acc_mc']:.4f}, VQA={v['acc_vqa']:.4f}")
        
    with open("results_ccot_iconqa.json", 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()