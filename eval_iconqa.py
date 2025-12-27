import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import yaml
import torch
import gc
import json 
import re 
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 匯入 IconQA Dataset ---
from dataset.iconqa import IconQADataset

# --- 從您現有的 utils.py 匯入 (不修改) ---
from dataset.utils import load_model_and_processor, collate_fn

# =================================================================
# (A) 多選題 (MCQ) 相關函數 (來自 ScienceQA)
# =================================================================

def score_mc_acc(pred, refs):
    """MCQ 評分: 標籤 A/B/C 是否匹配"""
    if not pred: return 0.0
    pred = pred.strip().upper() 
    refs = [a.strip().upper() for a in refs]
    return 1.0 if pred in refs else 0.0

def parse_final_answer_mc(full_text):
    """MCQ 解析: 從 CoT 中提取 A/B/C"""
    match = re.search(r"Final Answer:\s*([A-Z])", full_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"Answer:\s*([A-Z])", full_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"Option:\s*([A-Z])", full_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", full_text)
    if matches: return matches[-1].upper() 
    return ""

def format_mc_prompt_cot(question, choices_list):
    """MCQ 提示: CoT 鏈式思考"""
    prompt = "Answer the following multiple-choice question either directly or with some reasoning. Conclude the final answer with the correct option letter in the format 'Final Answer: [Letter]'.\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Options:\n"
    for i, choice in enumerate(choices_list):
        label = chr(65 + i)
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer: "
    return prompt

def generate_answer_mc(proc, m, mtype, images, questions, choices_batch, **kwargs):
    """
    MCQ 生成: 呼叫模型 (長回答, CoT)
    支援: Gemma3, Qwen-VL, LLaVA, InstructBLIP, Qwen3-VL
    """
    batch_size = len(questions)
    
    # --- Gemma 3 ---
    if mtype == "gemma3":
        messages_list = []
        batch_images = []
        for i in range(batch_size):
            VQA_PROMPT = format_mc_prompt_cot(questions[i], choices_batch[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
            batch_images.append(images[i])
            
        texts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=texts, images=batch_images, return_tensors="pt", padding=True).to(m.device)
        
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = proc.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [parse_final_answer_mc(ans) for ans in output_text]

    # --- Qwen-VL ---
    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_mc_prompt_cot(questions[i], choices_batch[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
        text_prompts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=text_prompts, images=images, return_tensors="pt", padding=True).to(m.device)
        
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        with torch.no_grad():
            res = m.generate(**inputs, **kwargs)
        responses = proc.batch_decode(res, skip_special_tokens=True)
        final_answers_full = [resp.split("assistant\n")[-1].strip() for resp in responses]
        return [parse_final_answer_mc(ans) for ans in final_answers_full]

    # --- LLaVA Series ---
    if mtype == "llava" or mtype == "dvit_llava" or mtype == "llava-next":
        prompt_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_mc_prompt_cot(questions[i], choices_batch[i])
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
        return [parse_final_answer_mc(ans) for ans in answers_full]

    # --- InstructBLIP ---
    if mtype == "instructblip":
        prompt_list = []
        for i in range(batch_size):
            prompt_list.append(format_mc_prompt_cot(questions[i], choices_batch[i]))
        inputs = proc(images=images, text=prompt_list, return_tensors="pt", padding=True).to(m.device)
        input_len = inputs['input_ids'].shape[1]
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        
        with torch.no_grad():
            out = m.generate(**inputs, **kwargs)
        new_tokens = out[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)
        return [parse_final_answer_mc(ans) for ans in answers_full]

    # --- Qwen3-VL ---
    if mtype == "qwen3-vl":
        messages_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_mc_prompt_cot(questions[i], choices_batch[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
            
        texts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=texts, images=images, return_tensors="pt", padding=True).to(m.device)
        
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = proc.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [parse_final_answer_mc(ans) for ans in output_text]
    if mtype == "structural_llava":
        answers = []
        for i in range(batch_size):
            # 1. 格式化問題 (加入選項列表)
            q_text = format_mc_prompt_cot(questions[i], choices_batch[i])
            
            try:
                # 2. 呼叫模型推論
                raw_ans = m.generate_answer(images[i], q_text)
                
                # 3. 解析答案 (A/B/C/D)
                parsed = parse_final_answer_mc(raw_ans)
                
                # Fallback: 如果 CoT 解析失敗，找最後出現的選項字母
                if not parsed:
                    matches = re.findall(r"\b([A-E])\b", raw_ans.upper())
                    parsed = matches[-1] if matches else ""
                
                answers.append(parsed)
            except Exception as e:
                print(f"[MCQ Error] Sample {i}: {e}")
                answers.append("") 
        return answers

    return [""] * batch_size

# =================================================================
# (B) 開放式問答 (VQA) 相關函數 (新函數)
# =================================================================

def score_vqa(pred, refs):
    """VQA 評分: 答案文本是否精確匹配"""
    if not pred: return 0.0
    pred = pred.strip().lower()
    refs = [a.strip().lower() for a in refs]
    return 1.0 if pred in refs else 0.0

def parse_final_answer_vqa(full_text):
    """VQA 解析: 清理答案文本"""
    try:
        # 尋找 'Final Answer:' (不分大小寫)
        split_tag = "Final Answer:"
        parts = full_text.split(split_tag)
        if len(parts) < 2:
             # 如果找不到 'Final Answer:'，嘗試直接取最後一個詞
             return full_text.strip().split()[-1]
             
        # 取得 'Final Answer:' 後的所有內容
        final_part = parts[-1]
        
        # 清理並獲取第一個詞
        final_answer = final_part.strip().split('\n')[0].strip().split(' ')[0].strip()
        
        # 移除可能的標點符號
        final_answer = final_answer.rstrip('.,')
        
        return final_answer
    except Exception:
        return "" # 發生錯誤時返回空字串

def format_vqa_prompt(question):
    """VQA 提示: 簡單直接提問"""
    return f"Answer the following question either directly or with reasoning. Give the final answer in a single word like 'Final Answer: [single word]'.\n\nQuestion: {question}\nAnswer:"

def generate_answer_vqa(proc, m, mtype, images, questions, **kwargs):
    """
    VQA 生成: 呼叫模型 (短回答)
    支援: Gemma3, Qwen-VL, LLaVA, InstructBLIP, Qwen3-VL
    """
    batch_size = len(questions)
    
    # --- Gemma 3 ---
    if mtype == "gemma3":
        messages_list = []
        batch_images = []
        for i in range(batch_size):
            VQA_PROMPT = format_vqa_prompt(questions[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
            batch_images.append(images[i])
            
        texts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=texts, images=batch_images, return_tensors="pt", padding=True).to(m.device)
        
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = proc.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [parse_final_answer_vqa(ans) for ans in output_text]

    # --- Qwen-VL ---
    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_vqa_prompt(questions[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
        text_prompts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=text_prompts, images=images, return_tensors="pt", padding=True).to(m.device)
        
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        with torch.no_grad():
            res = m.generate(**inputs, **kwargs)
        responses = proc.batch_decode(res, skip_special_tokens=True)
        final_answers_full = [resp.split("assistant\n")[-1].strip() for resp in responses]
        return [parse_final_answer_vqa(ans) for ans in final_answers_full]

    # --- LLaVA Series ---
    if mtype == "llava" or mtype == "dvit_llava" or mtype == "llava-next":
        prompt_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_vqa_prompt(questions[i])
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
        return [parse_final_answer_vqa(ans) for ans in answers_full]

    # --- InstructBLIP ---
    if mtype == "instructblip":
        prompt_list = []
        for i in range(batch_size):
            prompt_list.append(format_vqa_prompt(questions[i]))
        inputs = proc(images=images, text=prompt_list, return_tensors="pt", padding=True).to(m.device)
        input_len = inputs['input_ids'].shape[1]
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        
        with torch.no_grad():
            out = m.generate(**inputs, **kwargs)
        new_tokens = out[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)
        return [parse_final_answer_vqa(ans) for ans in answers_full]

    # --- Qwen3-VL ---
    if mtype == "qwen3-vl":
        messages_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_vqa_prompt(questions[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
        
        texts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=texts, images=images, return_tensors="pt", padding=True).to(m.device)
        
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = proc.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [parse_final_answer_vqa(ans) for ans in output_text]
    if mtype == "structural_llava":
        answers = []
        for i in range(batch_size):
            # 1. 格式化問題 (簡單問答 Prompt)
            q_text = format_vqa_prompt(questions[i])
            
            try:
                # 2. 呼叫模型推論
                raw_ans = m.generate_answer(images[i], q_text)
                
                # 3. 解析答案 (清理標點符號等)
                parsed = parse_final_answer_vqa(raw_ans)
                
                # Fallback: 如果解析回傳空，就直接用原始輸出
                if not parsed:
                    parsed = raw_ans.strip()
                
                answers.append(parsed)
            except Exception as e:
                print(f"[VQA Error] Sample {i}: {e}")
                answers.append("")
        return answers

    return [""] * batch_size


# =================================================================
# (C) 主評估迴圈 (重構)
# =================================================================

def main():
    config_dir = "configs"
    config_filename = 'iconqa.yaml'
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    # 1. 載入 IconQA 數據集 (Validation Split)
    ds = IconQADataset(
        dataset_id=cfg["dataset_id"],
        split="validation", 
        num_samples=cfg["num_val_samples"]
    )
    
    batch_size = cfg.get('batch_size', 1)
    print(f"Using Batch Size: {batch_size}")
    
    ds_loader = DataLoader(
        ds, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_fn 
    )
    
    all_results = {}
    
    for mdl in cfg["models"]:
        name = mdl["name"]
        print(f"\n--- Evaluating Model: {name} ---")
        
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        mdl_obj.eval()
        
        # <--- 修改: 分別儲存不同類型的分數
        scores_mc, scores_vqa, preds = [], [], []
        
        # <--- 修改: 為 VQA 和 MCQ 建立不同的生成參數
        generate_params_mc = {
            "max_new_tokens": cfg.get("max_new_tokens_mc", 200), # MCQ 需要長 CoT
            "do_sample": False,
            "min_new_tokens": 1,
            "num_beams": 1
        }
        generate_params_vqa = {
            "max_new_tokens": cfg.get("max_new_tokens_vqa", 100), # VQA 只需要短答案
            "do_sample": False,
            "min_new_tokens": 1,
            "num_beams": 1
        }

        for batch_data in tqdm(ds_loader, desc=name):
            
            images = batch_data["image"]
            questions = batch_data["question"]
            all_answers = batch_data["answers"] 
            question_ids = batch_data["question_id"]
            choices_batch = batch_data["choices"]
            question_types = batch_data["question_type"] # <--- 獲取類型

            # <--- 關鍵修改: 根據類型分離批次
            
            mc_indices = [i for i, t in enumerate(question_types) if t == 'multiple-choice']
            vqa_indices = [i for i, t in enumerate(question_types) if t == 'open-ended']
            
            # 建立一個列表來按順序儲存結果
            results = [None] * len(questions)

            # --- 處理 MCQ ---
            if mc_indices:
                mc_images = [images[i] for i in mc_indices]
                mc_questions = [questions[i] for i in mc_indices]
                mc_choices = [choices_batch[i] for i in mc_indices]
                
                mc_ans_list = generate_answer_mc(
                    proc, mdl_obj, mtype, 
                    mc_images, mc_questions, mc_choices, 
                    **generate_params_mc
                )
                # 將結果放回 results 列表
                for i, ans in enumerate(mc_ans_list):
                    results[mc_indices[i]] = (ans, 'mc')

            # --- 處理 VQA ---
            if vqa_indices:
                vqa_images = [images[i] for i in vqa_indices]
                vqa_questions = [questions[i] for i in vqa_indices]
                
                vqa_ans_list = generate_answer_vqa(
                    proc, mdl_obj, mtype, 
                    vqa_images, vqa_questions, 
                    **generate_params_vqa
                )
                # 將結果放回 results 列表
                for i, ans in enumerate(vqa_ans_list):
                    results[vqa_indices[i]] = (ans, 'vqa')

            # --- 迴圈評分 (已合併的結果) ---
            for i in range(len(results)):
                if results[i] is None: continue
                
                ans, q_type = results[i]
                refs = all_answers[i]
                
                # 跳過無效答案
                if "INVALID" in refs or not refs or not all(r.strip() for r in refs):
                    continue
                
                score = 0.0
                if q_type == 'mc':
                    score = score_mc_acc(ans, refs) 
                    scores_mc.append(score)
                elif q_type == 'vqa':
                    score = score_vqa(ans, refs)
                    scores_vqa.append(score)
                
                preds.append({
                    "qid": question_ids[i], 
                    "q_type": q_type,
                    "question": questions[i], 
                    "choices": choices_batch[i], 
                    "pred": ans, 
                    "refs": refs, 
                    "score": score
                })
            del images, questions, all_answers
            if 'batch_data' in locals():
                del batch_data
            
            # 這是關鍵：如果 Python 物件沒死，PyTorch 就不能釋放 GPU 記憶體
            gc.collect() 
            torch.cuda.empty_cache()

        # <--- 修改: 計算並報告三種準確率
        mean_acc_mc = (sum(scores_mc) / len(scores_mc)) * 100 if scores_mc else 0.0
        mean_acc_vqa = (sum(scores_vqa) / len(scores_vqa)) * 100 if scores_vqa else 0.0
        total_scores = scores_mc + scores_vqa
        mean_acc_overall = (sum(total_scores) / len(total_scores)) * 100 if total_scores else 0.0
        
        all_results[name] = {
            "acc_overall": mean_acc_overall,
            "acc_mc": mean_acc_mc,
            "acc_vqa": mean_acc_vqa,
            "counts": {"mc": len(scores_mc), "vqa": len(scores_vqa)},
            "predictions": preds
        }
        
        output_file = f"results_iconqa_{name.replace(' ', '_').replace('/', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results[name], f, indent=4, ensure_ascii=False)
        print(f"Results for {name} saved to {output_file}")

        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Final ICON-QA Evaluation Results ===")
    for k, v in all_results.items():
        print(f"Model: {k}")
        print(f"  Overall ACC: {v['acc_overall']:.2f}%")
        print(f"  MCQ ACC:     {v['acc_mc']:.2f}% (Count: {v['counts']['mc']})")
        print(f"  VQA ACC:     {v['acc_vqa']:.2f}% (Count: {v['counts']['vqa']})")

if __name__ == "__main__":
    main()