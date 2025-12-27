import os
import yaml
import torch
import gc
import re
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score

# --- 匯入 Dataset ---
# 假設 dataset/pope.py 存在
from dataset.pope import POPEDataset
from dataset.utils import load_model_and_processor, collate_fn 

# =================================================================
# 1. CCoT 提示詞模板 (針對 POPE 優化)
# =================================================================

CCOT_PROMPTS = {
    # -----------------------------------------------------------------
    # Qwen-VL
    # -----------------------------------------------------------------
    "qwen-vl": {
        'stage_1_system': "You are an expert visual perception assistant.",
        # Stage 1: 強調列出所有可見物體，這對 POPE 至關重要
        'stage_1_user': """For the provided image, generate a Scene Graph in JSON format.
Strictly list:
1. All visible objects in the image.
2. The count of each object.
3. Their spatial locations or relationships.

Question: What objects are in the image?
Scene Graph:""", 

        'stage_2_system': "You are an AI assistant verifying object existence.",
        
        # Stage 2: 基於 SG 驗證存在性 (Yes/No)
        'stage_2_user': """Use the image and the following Scene Graph to answer the Yes/No question.
Scene Graph: {scene_graph}

Question: {question}

Step 1: Check if the object mentioned in the question exists in the Scene Graph.
Step 2: Answer 'Yes' if it exists, otherwise answer 'No'.
Final Answer:"""
    },
    
    # -----------------------------------------------------------------
    # LLaVA / LLaVA-NeXT
    # -----------------------------------------------------------------
    "llava-next": {
        'stage_1_prompt': """[INST] <image>
For the provided image, generate a Scene Graph in JSON format.
Strictly list:
1. All visible objects in the image.
2. The count of each object.
3. Their spatial locations or relationships.

Question: What objects are in the image?
ASSISTANT: Scene Graph:[/INST]""",

        'stage_2_prompt': """[INST] <image>
Use the image and the following Scene Graph to answer the Yes/No question.
Scene Graph: {scene_graph}

Question: {question}

Step 1: Check if the object mentioned in the question exists in the Scene Graph.
Step 2: Answer 'Yes' if it exists, otherwise answer 'No'.
ASSISTANT: Final Answer:[/INST]"""
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
        if "ASSISTANT:" in sg: sg = sg.split("ASSISTANT:")[-1].strip()
        return sg if sg else "N/A"
    except:
        return "PARSE_ERROR"

def parse_final_answer_yesno(text):
    """ 解析 Stage 2 輸出 (提取 Yes/No) """
    # 1. 優先匹配 Final Answer
    match = re.search(r"Final Answer:\s*(Yes|No)", text, re.IGNORECASE)
    if match: return match.group(1).capitalize()
    
    # 2. 匹配 Answer
    match = re.search(r"Answer:\s*(Yes|No)", text, re.IGNORECASE)
    if match: return match.group(1).capitalize()
    
    # 3. 找最後出現的 Yes/No
    matches = re.findall(r"\b(Yes|No)\b", text, re.IGNORECASE)
    if matches: return matches[-1].capitalize()
    
    return "No" # 保守策略：如果解析失敗，回答 No (通常幻覺是亂答 Yes)

def _build_ccot_inputs(prompts, mtype, stage, images, questions, scene_graphs=None):
    """ 構建模型輸入 """
    batch_size = len(questions)
    
    # --- Qwen-VL ---
    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            if stage == "stage_1":
                # Stage 1: 只看圖生成 SG
                messages = [
                    {"role": "system", "content": prompts['stage_1_system']}, 
                    {"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": prompts['stage_1_user']}]}
                ]
            elif stage == "stage_2":
                # Stage 2: 看圖 + SG + 問題
                prompt_text = prompts['stage_2_user'].format(
                    scene_graph=scene_graphs[i],
                    question=questions[i]
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
                prompt = prompts['stage_1_prompt'] # POPE Stage 1 prompt 不需要 question 變數，是通用的 "列出物體"
            elif stage == "stage_2":
                prompt = prompts['stage_2_prompt'].format(
                    scene_graph=scene_graphs[i],
                    question=questions[i]
                )
            prompt_list.append(prompt)
        return {"prompt_list": prompt_list}
        
    else:
        raise NotImplementedError(f"mtype {mtype} not supported yet.")

# =================================================================
# 3. 生成函數
# =================================================================

def ccot_generate_answer(proc, m, mtype, images, questions, messages_list=None, prompt_list=None, **kwargs):
    # --- LLaVA-NeXT 範例 ---
    if mtype == "llava-next":
        if prompt_list is None: return [""] * len(images)
        final_prompts = []
        for p in prompt_list:
            if "[INST]" not in p and "USER:" not in p:
                final_prompts.append(f"[INST] <image>\n{p} [/INST]")
            else:
                final_prompts.append(p)
                
        inputs = proc(images=images, text=final_prompts, return_tensors="pt", padding=True).to(m.device)
        input_len = inputs['input_ids'].shape[1]
        
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['messages_list', 'prompt_list']}
        
        with torch.no_grad():
            outputs = m.generate(**inputs, **clean_kwargs)
        new_tokens = outputs[:, input_len:]
        answers = proc.batch_decode(new_tokens, skip_special_tokens=True)
        return [ans.strip() for ans in answers]
        
    # --- Qwen-VL 範例 ---
    if mtype == "qwen-vl":
        outputs_list = [""] * len(questions)
        for i in range(len(questions)):
            txt = proc.apply_chat_template(messages_list[i], tokenize=False, add_generation_prompt=True)
            inp = proc(text=[txt], images=[images[i]], return_tensors="pt").to(m.device)
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['messages_list', 'prompt_list']}
            clean_kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
            with torch.no_grad():
                out = m.generate(**inp, **clean_kwargs)
            out_decoded = proc.decode(out[0], skip_special_tokens=True)
            if "assistant\n" in out_decoded:
                outputs_list[i] = out_decoded.split("assistant\n")[-1].strip()
            else:
                outputs_list[i] = out_decoded.strip()
        return outputs_list

    return [""] * len(images)

def score_yesno(pred, refs):
    if not pred: return 0.0
    pred = pred.strip().capitalize()
    refs = [a.strip().capitalize() for a in refs]
    return 1.0 if pred in refs else 0.0

# =================================================================
# 4. 主程式 (Grouped Eval)
# =================================================================

def main():
    if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
         os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
         
    config_dir = "configs"
    config_filename = 'pope.yaml'
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    print(f"Running CCoT Evaluation on POPE...")
    
    # POPE Categories
    TARGET_CATEGORIES = ["random", "popular", "adversarial"]
    SAMPLES_PER_CATEGORY = 500
    
    batch_size = cfg.get('batch_size', 1)
    print(f"Using Batch Size: {batch_size}")
    
    all_results = {}
    
    # CCoT Params
    STAGE1_MAX_TOKENS = 300 # SG 需要列出所有物體
    STAGE2_MAX_TOKENS = 50  # Yes/No 很短

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
        
        model_category_results = {}
        
        gen_params_1 = {"max_new_tokens": STAGE1_MAX_TOKENS, "do_sample": False}
        gen_params_2 = {"max_new_tokens": STAGE2_MAX_TOKENS, "do_sample": False}

        # --- Loop Categories ---
        for category in TARGET_CATEGORIES:
            print(f"\n--- Running Category: {category.upper()} ---")
            
            ds = POPEDataset(
                dataset_id=cfg["dataset_id"],
                split="test",
                target_category=category,
                num_samples_per_category=SAMPLES_PER_CATEGORY
            )
            ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
            all_true_labels, all_pred_labels = [], []
            preds_list_for_json = []

            for batch_data in tqdm(ds_loader, desc=f"{name} ({category})"):
                images = batch_data["image"]
                questions = batch_data["question"]
                all_answers = batch_data["answers"]
                question_ids = batch_data["question_id"]
                
                # --- STAGE 1: 生成 Scene Graph (List Objects) ---
                stage1_inputs = _build_ccot_inputs(
                    prompts, base_type, "stage_1", images, questions
                )
                
                stage1_outputs = ccot_generate_answer(
                    proc, mdl_obj, mtype,
                    images, questions,
                    **stage1_inputs,
                    **gen_params_1
                )
                
                generated_sgs = [parse_scene_graph(out) for out in stage1_outputs]
                
                # --- STAGE 2: 驗證存在性 (Verify Existence) ---
                stage2_inputs = _build_ccot_inputs(
                    prompts, base_type, "stage_2", 
                    images, questions, 
                    scene_graphs=generated_sgs
                )
                
                stage2_outputs = ccot_generate_answer(
                    proc, mdl_obj, mtype,
                    images, questions,
                    **stage2_inputs,
                    **gen_params_2
                )
                
                # --- 計分 ---
                for i in range(len(stage2_outputs)):
                    raw_ans = stage2_outputs[i]
                    refs = all_answers[i]
                    true_label = refs[0]
                    
                    parsed_ans = parse_final_answer_yesno(raw_ans)
                    
                    all_true_labels.append(true_label)
                    all_pred_labels.append(parsed_ans)
                    
                    score = score_yesno(parsed_ans, refs)
                        
                    preds_list_for_json.append({
                        "qid": str(question_ids[i]),
                        "question": questions[i],
                        "pred": parsed_ans,
                        "raw_output": raw_ans,
                        "refs": refs,
                        "score": score,
                        "scene_graph": generated_sgs[i]
                    })
            
            # --- 計算 Category Metrics ---
            if not all_true_labels: continue
            
            acc = accuracy_score(all_true_labels, all_pred_labels) * 100
            labels = sorted(list(set(all_true_labels + all_pred_labels)))
            report = classification_report(all_true_labels, all_pred_labels, labels=labels, output_dict=True, zero_division=0)
            
            model_category_results[category] = {
                "count": len(all_true_labels),
                "accuracy": acc,
                "metrics": report,
                "predictions": preds_list_for_json
            }
            print(f"  Accuracy: {acc:.2f}%")

        # --- 儲存該模型結果 ---
        all_results[name] = model_category_results
        
        output_file = f"results_ccot_pope_grouped_{name.replace(' ', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results[name], f, indent=2)
        
        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    # --- 最終摘要 ---
    print("\n" + "="*30)
    print("Final CCoT Results (POPE):")
    for model_name, results in all_results.items():
        print(f"\nModel: {model_name}")
        for cat, data in results.items():
            print(f"  {cat.upper()}: Accuracy={data['accuracy']:.2f}%")

if __name__ == "__main__":
    main()