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
# 假設 dataset/mcot.py 存在
from dataset.mcot import M3CoTDataset
from dataset.utils import load_model_and_processor, collate_fn 

# =================================================================
# 1. CCoT 提示詞模板 (針對 M3CoT 優化)
# =================================================================

CCOT_PROMPTS = {
    # -----------------------------------------------------------------
    # Qwen-VL (ChatML)
    # -----------------------------------------------------------------
    "qwen-vl": {
        'stage_1_system': "You are an expert visual perception assistant.",
        # Stage 1: 專注於細節描述，因為 M3CoT 通常考細節
        'stage_1_user': """For the provided image and its associated question, generate a Scene Graph in JSON format.
Include:
1. All objects mentioned in the question or relevant to the context.
2. Detailed attributes (color, position, action, state) of these objects.
3. Relationships between objects (spatial, interactive).

Question: {question}
Scene Graph:""", 

        'stage_2_system': "You are an expert in multi-hop visual reasoning.",
        
        # Stage 2: 傳入 SG + Question + Options
        'stage_2_user': """Use the image and the following Scene Graph to reason and answer the multiple-choice question.
Scene Graph: {scene_graph}

Question: {question}
Options:
{options_str}

Reason step-by-step about the relationship between the scene graph and the options.
Finally, provide the answer as a single letter (A, B, C, or D) in the format 'Final Answer: [Letter]'.
Answer:"""
    },
    
    # -----------------------------------------------------------------
    # LLaVA / LLaVA-NeXT
    # -----------------------------------------------------------------
    "llava-next": {
        'stage_1_prompt': """[INST] <image>
For the provided image and its associated question, generate a Scene Graph in JSON format.
Include:
1. All objects mentioned in the question or relevant to the context.
2. Detailed attributes (color, position, action, state) of these objects.
3. Relationships between objects (spatial, interactive).

Question: {question}
ASSISTANT: Scene Graph:[/INST]""",

        'stage_2_prompt': """[INST] <image>
Use the image and the following Scene Graph to reason and answer the multiple-choice question.
Scene Graph: {scene_graph}

Question: {question}
Options:
{options_str}

Reason step-by-step about the relationship between the scene graph and the options.
Finally, provide the answer as a single letter (A, B, C, or D) in the format 'Final Answer: [Letter]'.
ASSISTANT: Answer:[/INST]"""
    }
}

# =================================================================
# 2. 輔助函數
# =================================================================

def parse_scene_graph(text):
    """ 解析 Stage 1 輸出 """
    try:
        match = re.search(r"Scene Graph:(.*)", text, re.DOTALL | re.IGNORECASE)
        sg = match.group(1).strip() if match else text.strip()
        # 清理可能的 Assistant 標籤
        if "ASSISTANT:" in sg:
            sg = sg.split("ASSISTANT:")[-1].strip()
        return sg if sg else "N/A"
    except:
        return "PARSE_ERROR"

def parse_final_answer_mc(text):
    """ 解析 Stage 2 輸出 (提取 A/B/C/D) """
    # 1. 優先匹配標準格式
    match = re.search(r"Final Answer:\s*([A-D])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    match = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # 2. 寬鬆匹配：找最後出現的選項字母
    matches = re.findall(r"\b([A-D])\b", text)
    if matches: return matches[-1].upper()
    
    return ""

def _format_options(choices):
    """ 將選項列表轉為字串 """
    output = ""
    for i, choice in enumerate(choices):
        output += f"{chr(65+i)}. {choice}\n"
    return output.strip()

def _build_ccot_inputs(prompts, mtype, stage, images, questions, choices_batch=None, scene_graphs=None):
    """ 構建模型輸入 """
    batch_size = len(questions)
    
    # --- Qwen-VL ---
    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            if stage == "stage_1":
                prompt_text = prompts['stage_1_user'].format(question=questions[i])
                messages = [
                    {"role": "system", "content": prompts['stage_1_system']}, 
                    {"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": prompt_text}]}
                ]
            elif stage == "stage_2":
                options_str = _format_options(choices_batch[i])
                prompt_text = prompts['stage_2_user'].format(
                    scene_graph=scene_graphs[i],
                    question=questions[i],
                    options_str=options_str
                )
                messages = [
                    {"role": "system", "content": prompts['stage_2_system']}, 
                    {"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": prompt_text}]}
                ]
            messages_list.append(messages)
        return {"messages_list": messages_list}

    # --- LLaVA / LLaVA-NeXT ---
    elif mtype == "llava" or mtype == "llava-next":
        prompt_list = []
        for i in range(batch_size):
            if stage == "stage_1":
                prompt = prompts['stage_1_prompt'].format(question=questions[i])
            elif stage == "stage_2":
                options_str = _format_options(choices_batch[i])
                prompt = prompts['stage_2_prompt'].format(
                    scene_graph=scene_graphs[i],
                    question=questions[i],
                    options_str=options_str
                )
            prompt_list.append(prompt)
        return {"prompt_list": prompt_list}
        
    else:
        raise NotImplementedError(f"mtype {mtype} not supported yet.")

# =================================================================
# 3. 生成與評分
# =================================================================

def ccot_generate_answer(proc, m, mtype, images, questions, messages_list=None, prompt_list=None, **kwargs):
    # 這裡請務必包含您完整的生成邏輯 (支援 batch processing)
    # 以下為 LLaVA-NeXT 的簡化範例
    if mtype == "llava-next":
        if prompt_list is None: return [""] * len(images)
        # 確保 prompt 格式正確 (LLaVA-NeXT 需要 [INST])
        final_prompts = []
        for p in prompt_list:
            if "[INST]" not in p and "USER:" not in p: # 如果還沒格式化
                final_prompts.append(f"[INST] <image>\n{p} [/INST]")
            else:
                final_prompts.append(p)
                
        inputs = proc(images=images, text=final_prompts, return_tensors="pt", padding=True).to(m.device)
        input_len = inputs['input_ids'].shape[1]
        
        # 清理不必要的 kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['messages_list', 'prompt_list']}
        
        with torch.no_grad():
            outputs = m.generate(**inputs, **clean_kwargs)
        new_tokens = outputs[:, input_len:]
        answers = proc.batch_decode(new_tokens, skip_special_tokens=True)
        return [ans.strip() for ans in answers]

    # ... 請補上 Qwen-VL 等其他模型的生成邏輯 ...
    return [""] * len(images)

def score_mc_acc(pred, refs):
    if not pred: return 0.0
    pred = pred.strip().upper() 
    refs = [a.strip().upper() for a in refs]
    return 1.0 if pred in refs else 0.0

# =================================================================
# 4. 主程式
# =================================================================

def main():
    if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
         os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
         
    config_dir = "configs"
    config_filename = 'mcot.yaml'
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    print(f"Running CCoT Evaluation on M3CoT...")
    
    # 載入 M3CoT Dataset
    ds = M3CoTDataset(
        dataset_id=cfg["dataset_id"],
        split="validation", 
        num_samples=cfg.get("num_val_samples", None)
    )
    
    batch_size = cfg.get('batch_size', 1)
    print(f"Using Batch Size: {batch_size}")
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    all_results = {}
    
    # CCoT 參數
    # Stage 1 需要長一點的 context 來生成詳細的 Scene Graph
    STAGE1_MAX_TOKENS = 300 
    # Stage 2 需要包含推理過程 (CoT)，所以也不能太短
    STAGE2_MAX_TOKENS = 200 

    for mdl in cfg["models"]:
        name = mdl["name"]
        print(f"\nProcessing model: {name}")
        
        base_type = mdl['type']
        
        if base_type not in CCOT_PROMPTS:
            print(f"Skipping {name}: No CCoT prompts defined.")
            continue
            
        prompts = CCOT_PROMPTS[base_type]
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        mdl_obj.eval()
        
        scores = []
        preds = []

        gen_params_stage1 = {"max_new_tokens": STAGE1_MAX_TOKENS, "do_sample": False}
        gen_params_stage2 = {"max_new_tokens": STAGE2_MAX_TOKENS, "do_sample": False}

        for batch_data in tqdm(ds_loader, desc=name):
            
            images = batch_data["image"]
            questions = batch_data["question"]
            all_answers = batch_data["answers"]
            question_ids = batch_data["question_id"]
            choices_batch = batch_data["choices"] # M3CoT 必有選項
            
            # --- STAGE 1: 生成 Scene Graph ---
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
            
            # --- STAGE 2: 基於 SG 回答 MCQ ---
            stage2_inputs = _build_ccot_inputs(
                prompts, base_type, "stage_2", 
                images, questions, 
                choices_batch=choices_batch, 
                scene_graphs=generated_sgs
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
                refs = all_answers[i]
                
                parsed_ans = parse_final_answer_mc(raw_ans)
                score = score_mc_acc(parsed_ans, refs)
                scores.append(score)
                    
                preds.append({
                    "qid": question_ids[i],
                    "question": questions[i],
                    "pred": parsed_ans,
                    "raw_output": raw_ans,
                    "refs": refs,
                    "score": score,
                    "scene_graph": generated_sgs[i]
                })

        # --- 統計 ---
        mean_acc = sum(scores)/len(scores) if scores else 0
        all_results[name] = {"acc": mean_acc, "predictions": preds}
        
        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    # --- 輸出 ---
    print("\n" + "="*30)
    print("Final CCoT Results (M3CoT):")
    for k, v in all_results.items():
        print(f"{k}: Accuracy={v['acc']:.4f}")
        
    with open("results_ccot_m3cot.json", 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()