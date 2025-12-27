import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import yaml
import torch
import gc
torch.cuda.empty_cache()
gc.collect()
import json 
import re 
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 匯入 POPE Dataset ---
from dataset.pope import POPEDataset

# --- 從您現有的 utils.py 匯入 (不修改) ---
from dataset.utils import load_model_and_processor, collate_fn

# --- 匯入 Scikit-learn ---
from sklearn.metrics import classification_report, accuracy_score

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

# =================================================================
# 【新函數】: POPE 專用的 Prompt
# =================================================================

def format_pope_prompt(question):
    """
    格式化一個提示，要求模型回答關於圖像內容的是非題。
    'question' 來自 POPE 資料集 (e.g., "Is there an apple?")
    """
    prompt = ("Based on the image, answer the following Yes/No question. You can either answer directly or with some reasoning. State your final answer in the format 'Final Answer: [Yes/No]'.\n\n")
    
    prompt += f"Question: {question}\n"
    
    prompt += ("Answer:\n")
    
    return prompt

# =================================================================
# 【重用】: 答案生成函數 (修改為使用 format_pope_prompt)
# =================================================================

def generate_answer_yesno(proc, m, mtype, images, questions, **kwargs):
    """
    此函數使用 POPE 專用的提示 (format_pope_prompt)。
    支援: Gemma3, Qwen-VL, LLaVA, InstructBLIP, Qwen3-VL
    """
    batch_size = len(questions)
    
    # --- Gemma 3 ---
    if mtype == "gemma3":
        messages_list = []
        batch_images = []
        for i in range(batch_size):
            VQA_PROMPT = format_pope_prompt(questions[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
            batch_images.append(images[i])

        texts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=texts, images=batch_images, return_tensors="pt", padding=True).to(m.device)

        kwargs.setdefault('max_new_tokens', 200)
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)
            
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = proc.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [parse_final_answer_yesno(ans) for ans in output_text]

    # --- Qwen-VL ---
    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_pope_prompt(questions[i]) 
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

    # --- LLaVA Series ---
    if mtype == "llava" or mtype == "dvit_llava" or mtype == "llava-next":
        prompt_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_pope_prompt(questions[i]) 
            if mtype == "llava-next":
                prompt_text = f"[INST] <image>\n{VQA_PROMPT} [/INST]"
            else:
                prompt_text = f"USER: <image>\n{VQA_PROMPT} ASSISTANT:"
            prompt_list.append(prompt_text)
            
        inputs = proc(images=images, text=prompt_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(m.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        
        with torch.no_grad():
            outputs = m.generate(**inputs, **kwargs)
        new_tokens = outputs[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)
        return [parse_final_answer_yesno(ans) for ans in answers_full]

    # --- InstructBLIP ---
    if mtype == "instructblip":
        prompt_list = []
        for i in range(batch_size):
            prompt_list.append(format_pope_prompt(questions[i]))
            
        inputs = proc(images=images, text=prompt_list, return_tensors="pt", padding=True).to(m.device)
        input_len = inputs['input_ids'].shape[1]
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        
        with torch.no_grad():
            out = m.generate(**inputs, **kwargs)
        new_tokens = out[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)
        return [parse_final_answer_yesno(ans) for ans in answers_full]

    # --- Qwen3-VL ---
    if mtype == "qwen3-vl":
        messages_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_pope_prompt(questions[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
            
        texts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=texts, images=images, return_tensors="pt", padding=True).to(m.device)
        
        kwargs.setdefault('max_new_tokens', 200)
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)
            
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = proc.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [parse_final_answer_yesno(ans) for ans in output_text]
    if mtype == "structural_llava":
        answers = []
        for i in range(batch_size):
            # 1. 格式化問題
            # 使用 POPE 專用的 Prompt (要求回答 Yes/No)
            q_text = format_pope_prompt(questions[i])
            
            try:
                # 2. 呼叫模型推論
                # model.generate_answer 內部會處理 Florence 幾何特徵提取
                raw_ans = m.generate_answer(images[i], q_text)
                
                # 3. 解析答案
                # 優先使用 CoT 解析邏輯
                parsed = parse_final_answer_yesno(raw_ans)
                
                # 如果解析失敗，進行關鍵字掃描 (因為 POPE 很單純)
                if not parsed:
                    raw_lower = raw_ans.lower()
                    # 簡單啟發式規則：找最後出現的 yes 或 no
                    # 注意：要避免匹配到 "not" 裡面的 "no"
                    matches = re.findall(r"\b(yes|no)\b", raw_lower)
                    if matches:
                        parsed = matches[-1].capitalize()
                    else:
                        parsed = "No" # 預設保守回答
                
                answers.append(parsed)
                
            except Exception as e:
                print(f"[Error] Sample {i}: {e}")
                answers.append("No") # 出錯時保守回答 No
        return answers
        
    return [""] * batch_size

# =================================================================
# 【核心修改】: 主評估迴圈 (分組評估)
# =================================================================

def main():
    config_dir = "configs"
    config_filename = 'pope.yaml'
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    # 【新的評估目標】: 三種類別
    TARGET_CATEGORIES = ["random", "popular", "adversarial"]
    SAMPLES_PER_CATEGORY = 500
    
    batch_size = cfg.get('batch_size', 4)
    print(f"Using Batch Size: {batch_size}")
    
    all_results = {}
    
    for mdl in cfg["models"]:
        name = mdl["name"]
        print(f"\n--- Evaluating Model: {name} ---")
        
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        mdl_obj.eval()
        
        model_category_results = {} # 儲存此模型在三個類別上的結果

        generate_params = {
            "max_new_tokens": cfg.get("max_new_tokens", 300),
            "do_sample": False,
            "min_new_tokens": 1,
            "num_beams": 1
        }

        # 【關鍵修改】: 迴圈處理三種類別
        for category in TARGET_CATEGORIES:
            print(f"\n--- Running Category: {category.upper()} ({SAMPLES_PER_CATEGORY} samples) ---")
            
            # 1. 載入並採樣當前類別的資料
            ds = POPEDataset(
                dataset_id=cfg["dataset_id"],
                split="test", # POPE 使用 'test' split
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
                
                ans_list = generate_answer_yesno(
                    proc, mdl_obj, mtype, 
                    images, questions, 
                    **generate_params
                )
                
                for i in range(len(ans_list)):
                    pred_ans = ans_list[i]
                    true_refs = all_answers[i]
                    true_label = true_refs[0]
                    
                    if not true_label or true_label == "INVALID":
                        continue
                    
                    all_true_labels.append(true_label)
                    all_pred_labels.append(pred_ans)
                    
                    score = score_yesno(pred_ans, true_refs)
                    preds_list_for_json.append({
                        "qid": question_ids[i], 
                        "category": category,
                        "question": questions[i], 
                        "pred": pred_ans, 
                        "refs": true_refs, 
                        "score": score
                    })
                del images, questions, all_answers, ans_list
                if 'batch_data' in locals():
                    del batch_data
                gc.collect() 
                torch.cuda.empty_cache(
            
            # 這是關鍵：如果 Python 物件沒死，PyTorch 就不能釋放 GPU 記憶體
            )

            # 2. 計算並儲存當前類別的 metrics
            if not all_true_labels:
                print(f"Skipping category {category}: No data processed.")
                continue

            mean_acc = accuracy_score(all_true_labels, all_pred_labels) * 100
            
            labels = sorted(list(set(all_true_labels + all_pred_labels)))
            report_dict = classification_report(
                all_true_labels, 
                all_pred_labels, 
                labels=labels,
                output_dict=True,
                zero_division=0
            )

            model_category_results[category] = {
                "count": len(all_true_labels),
                "accuracy": mean_acc,
                "metrics": report_dict,
                "predictions": preds_list_for_json # 每個類別單獨儲存預測
            }
            
            print(f"  Accuracy: {mean_acc:.2f}%")
            if 'Yes' in report_dict:
                print(f"  Yes F1: {report_dict['Yes']['f1-score']:.4f}, No F1: {report_dict['No']['f1-score']:.4f}")

        # 3. 儲存模型的所有類別結果
        all_results[name] = model_category_results
        
        output_file = f"results_pope_grouped_{name.replace(' ', '_').replace('/', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results[name], f, indent=4, ensure_ascii=False)
        print(f"\n--- Grouped results for {name} saved to {output_file} ---")

        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Final POPE Grouped Metrics Summary ===")
    for model_name, results in all_results.items():
        print(f"\nModel: {model_name}")
        for category, data in results.items():
            print(f"  {category.upper()} (N={data['count']}): Accuracy={data['accuracy']:.2f}%")
            if 'Yes' in data['metrics']:
                print(f"    Precision (Yes): {data['metrics']['Yes']['precision']:.4f}, Recall (Yes): {data['metrics']['Yes']['recall']:.4f}, F1 (Yes): {data['metrics']['Yes']['f1-score']:.4f}")

if __name__ == "__main__":
    main()