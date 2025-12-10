import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import yaml
import torch
import gc
import json 
import re 
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 匯入 MMMC Dataset ---
from dataset.mmmc import MMMCDataset

# --- 從您現有的 utils.py 匯入 (不修改) ---
from dataset.utils import load_model_and_processor, collate_fn

# --- (新) 匯入 Scikit-learn ---
from sklearn.metrics import classification_report, accuracy_score

# =================================================================
# 【不變】: MMMC (是非題) 專用的評分與解析函數
# =================================================================

def score_yesno(pred, refs):
    """是非題評分標準：pred 標籤必須完全匹配 refs 列表中的唯一標籤 (Yes/No)。"""
    if not pred: return 0.0
    pred = pred.strip().capitalize() # "Yes" or "No"
    refs = [a.strip().capitalize() for a in refs]
    return 1.0 if pred in refs else 0.0

def parse_final_answer_yesno(full_text):
    """從完整的 CoT 輸出中提取最終的是非答案 (Yes/No)。"""
    # 1. 尋找 "Final Answer: [Yes/No]"
    match = re.search(r"Final Answer:\s*(Yes|No)", full_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    # 2. 備用：尋找 "Answer: [Yes/No]"
    match = re.search(r"Answer:\s*(Yes|No)", full_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    # 3. 備用：尋找最後一個被提及的 "Yes" 或 "No"
    matches = re.findall(r"\b(Yes|No)\b", full_text, re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()
    return "" # 提取失敗

def format_conflict_prompt(original_sentence):
    """格式化一個提示，要求模型判斷圖像和文本之間是否存在衝突。"""
    prompt = "You are given an image and a question related to it. Your task is to determine whether there is a conflict between the image and the question.\n\n"
    prompt += "--- Example Start ---\n"
    prompt += "Image: A picture showing a cat sitting on a sofa.\n"
    prompt += "Question: What is the dog doing?\n"
    prompt += "Answer: There is no dog in the image, so the conflict is present.\n"
    prompt += "Final Answer: Yes.\n"
    prompt += "--- Example End ---\n\n"
    prompt += "--- Task Start ---\n"
    prompt += (f"Is there a conflict between the image and the following question \"{original_sentence}\"?\n\n")
    prompt += ("You can answer with brief or full reasoning. State your final answer in the format 'Final Answer: [Yes/No]'.\n\n")
    prompt += ("Answer:\n")
    return prompt

# =================================================================
# 【不變】: MMMC 專用的答案生成函數
# =================================================================

def generate_answer_yesno(proc, m, mtype, images, questions, **kwargs):
    """此函數不接受 'choices_batch'，並使用衝突檢測提示。"""
    batch_size = len(questions)
    
    # Qwen-VL (批次處理)
    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_conflict_prompt(questions[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
        text_prompts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=text_prompts, images=images, return_tensors="pt", padding=True).to(m.device)
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        with torch.no_grad():
            res = m.generate(**inputs, **kwargs)
        responses = proc.batch_decode(res, skip_special_tokens=True)
        final_answers_full = [resp.split("assistant\n")[-1].strip() for resp in responses]
        return [parse_final_answer_yesno(ans) for ans in final_answers_full]

    # LLaVA / dvit_llava (批次處理)
    if mtype == "llava" or mtype == "dvit_llava":
        prompt_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_conflict_prompt(questions[i])
            prompt_list.append(f"USER: <image>\n{VQA_PROMPT} ASSISTANT:")
        inputs = proc(images=images, text=prompt_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(m.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        outputs = m.generate(**inputs, **kwargs)
        new_tokens = outputs[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)
        return [parse_final_answer_yesno(ans) for ans in answers_full]
        
    return [""] * batch_size

# =================================================================
# 【修改】: 主評估迴圈 (加入 sklearn  metrics)
# =================================================================

def main():
    config_dir = "configs"
    config_filename = 'mmmc.yaml' # <--- 使用 mmmc.yaml
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    ds = MMMCDataset(
        dataset_id=cfg["dataset_id"],
        split="test", 
        num_samples=cfg["num_val_samples"]
    )
    
    batch_size = cfg.get('batch_size', 1)
    print(f"Using Batch Size: {batch_size}")
    
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    all_results = {}
    
    for mdl in cfg["models"]:
        name = mdl["name"]
        print(f"\n--- Evaluating Model: {name} ---")
        
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        mdl_obj.eval()
        
        # 【修改】: 新增列表以儲存所有標籤
        preds_list_for_json = [] # 用於儲存 JSON
        all_true_labels = []     # 用於 sklearn (真實答案)
        all_pred_labels = []     # 用V_SKlearn (模型預測)
        
        generate_params = {
            "max_new_tokens": cfg.get("max_new_tokens", 300),
            "do_sample": False,
            "min_new_tokens": 1,
            "num_beams": 1
        }

        for batch_data in tqdm(ds_loader, desc=name):
            
            images = batch_data["image"]
            questions = batch_data["question"]
            all_answers = batch_data["answers"] 
            question_ids = batch_data["question_id"]

            ans_list = generate_answer_yesno(
                proc, mdl_obj, mtype, 
                images, questions, 
                **generate_params
            )
            
            for i in range(len(ans_list)):
                pred_ans = ans_list[i]
                true_refs = all_answers[i]
                true_label = true_refs[0] # 從 ['Yes'] 提取 'Yes'
                
                if not true_label or true_label == "INVALID":
                    continue
                
                # 【修改】: 儲存真實標籤和預測標籤
                all_true_labels.append(true_label)
                all_pred_labels.append(pred_ans)
                
                # (可選) 儲存 JSON 用的詳細資料
                score = score_yesno(pred_ans, true_refs)
                preds_list_for_json.append({
                    "qid": question_ids[i], 
                    "question": questions[i], 
                    "pred": pred_ans, 
                    "refs": true_refs, 
                    "score": score
                })

        # --- 【修改】: 迴圈結束後，計算並顯示 sklearn metrics ---
        
        # 確保有結果
        if not all_true_labels:
            print(f"No valid samples found for {name}.")
            continue

        # 計算整體準確率 (Accuracy)
        mean_acc = accuracy_score(all_true_labels, all_pred_labels) * 100

        # 獲取分類報告 (Precision, Recall, F1-Score)
        # 確保 "Yes" 和 "No" 都有在標籤中
        labels = sorted(list(set(all_true_labels + all_pred_labels)))
        if not labels:
            labels = ["Yes", "No"] # 預設

        print(f"\n--- Metrics for {name} ---")
        print(f"Overall Accuracy: {mean_acc:.2f}%")
        
        # 產生報告 (文本格式)
        report_str = classification_report(
            all_true_labels, 
            all_pred_labels, 
            labels=labels,
            zero_division=0
        )
        print("Classification Report:")
        print(report_str)

        # 產生報告 (字典格式，用於儲存)
        report_dict = classification_report(
            all_true_labels, 
            all_pred_labels, 
            labels=labels,
            output_dict=True,
            zero_division=0
        )
        
        all_results[name] = {
            "overall_accuracy": mean_acc,
            "classification_report": report_dict,
            "predictions": preds_list_for_json
        }
        
        output_file = f"results_mmmc_metrics_{name.replace(' ', '_').replace('/', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results[name], f, indent=4, ensure_ascii=False)
        print(f"Results and metrics for {name} saved to {output_file}")

        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Final MMMC (Conflict Detection) Metrics Summary ===")
    for k, v in all_results.items():
        print(f"Model: {k}")
        print(f"  Overall Accuracy: {v['overall_accuracy']:.2f}%")
        # 打印 'Yes' (衝突) 類別的 F1-Score 作為參考
        if "Yes" in v["classification_report"]:
             print(f"  F1-Score (Yes): {v['classification_report']['Yes']['f1-score']:.4f}")
        if "No" in v["classification_report"]:
             print(f"  F1-Score (No): {v['classification_report']['No']['f1-score']:.4f}")

if __name__ == "__main__":
    main()