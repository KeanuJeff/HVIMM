# evaluate_mcot.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import yaml
import torch
import gc
import json 
import re # <-- 新增
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 匯入 M3CoT Dataset ---
from dataset.mcot import M3CoTDataset 

# --- 從您現有的 utils.py 匯入 (不修改) ---
from dataset.utils import load_model_and_processor, collate_fn

# =================================================================
# 【新函數】: M3CoT 專用的評分與解析函數 (定義於此檔案)
# =================================================================

def mcot_score(pred, refs):
    """
    M3CoT 評分標準：pred 標籤必須完全匹配 refs 列表中的唯一標籤。
    """
    if not pred: return 0.0
    pred = pred.strip().upper() # 確保是大寫標籤
    refs = [a.strip().upper() for a in refs]
    
    # refs 是 ['C'], pred 必須是 'C'
    return 1.0 if pred in refs else 0.0

def parse_final_answer_mc(full_text):
    """
    從完整的 CoT 輸出中提取最終的選項標籤 (A, B, C, D)。
    """
    # 1. 尋找 "Final Answer: [Label]" 格式
    match = re.search(r"Final Answer:\s*([A-D])", full_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2. 如果找不到，尋找結尾是 "Answer: [Label]"
    match = re.search(r"Answer:\s*([A-D])", full_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # 3. 如果找不到，尋找結尾是 "Option: [Label]"
    match = re.search(r"Option:\s*([A-D])", full_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 4. 如果都找不到，嘗試提取 CoT 文本中的最後一個大寫字母 A, B, C, D
    matches = re.findall(r"\b([A-D])\b", full_text)
    if matches:
        return matches[-1].upper() # 返回最後一個匹配項

    return "" # 提取失敗

def format_mc_prompt_cot(question, choices_list):
    """(新函數) 格式化 CoT 提示，要求模型推理並給出標籤"""
    prompt = "Answer the following multiple-choice question. First, provide your answer and then conclude with the correct option letter in the format 'Final Answer: [Letter]'.\n\n"
    prompt += "--- Example Start ---\n"
    prompt += "Question: some example question?\n"
    prompt += "Options:\n"
    prompt += "A. Option A\n"
    prompt += "B. Option B\n"
    prompt += "C. Option C\n"
    prompt += "Answer: After analyzing the question, the correct option is B.\n"
    prompt += "Final Answer: B.\n"
    prompt += "--- Example End ---\n\n"
    prompt += "--- Task Start ---\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Options:\n"
    for i, choice in enumerate(choices_list):
        label = chr(65 + i) # A, B, C, D
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt

# =================================================================
# 【新函數】: M3CoT 專用的答案生成函數 (定義於此檔案)
# =================================================================

def generate_answer_mc(proc, m, mtype, images, questions, choices_batch, **kwargs):
    """
    此函數接受 'choices_batch' 並使用多選題 CoT 提示。
    整合了 Gemma3, Qwen-VL, LLaVA, InstructBLIP, Qwen3-VL 等模型的批次生成邏輯。
    """
    batch_size = len(questions)
    
    # ----------------------------------------------------
    # Gemma 3 (批次處理)
    # ----------------------------------------------------
    if mtype == "gemma3":
        messages_list = []
        batch_images = [] 

        for i in range(batch_size):
            # 套用 MC CoT Prompt
            prompt_text = format_mc_prompt_cot(questions[i], choices_batch[i])
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[i]}, 
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            messages_list.append(messages)
            batch_images.append(images[i]) 

        texts = [
            proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]
        
        inputs = proc(
            text=texts,
            images=batch_images,
            padding=True,
            return_tensors="pt"
        ).to(m.device)

        kwargs.setdefault('max_new_tokens', 200)
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = proc.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        return [parse_final_answer_mc(ans) for ans in output_text]

    # ----------------------------------------------------
    # Qwen-VL (批次處理)
    # ----------------------------------------------------
    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_mc_prompt_cot(questions[i], choices_batch[i])
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": images[i]},
                        {"type": "text", "text": VQA_PROMPT} 
                    ]
                }
            ]
            messages_list.append(messages)

        text_prompts = [
            proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]
        
        inputs = proc(
            text=text_prompts, images=images, return_tensors="pt", padding=True
        ).to(m.device)
        
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        with torch.no_grad():
            res = m.generate(**inputs, **kwargs)
        
        responses = proc.batch_decode(res, skip_special_tokens=True)
        
        final_answers_full = []
        split_tag = "assistant\n"
        for response in responses:
            if split_tag in response:
                answer = response.split(split_tag)[-1].strip()
            else:
                answer = response.strip()
            final_answers_full.append(answer)

        return [parse_final_answer_mc(ans) for ans in final_answers_full]

    # ----------------------------------------------------
    # LLaVA / dvit_llava (批次處理)
    # ----------------------------------------------------
    if mtype == "llava" or mtype == "dvit_llava" or mtype == "llava-next":
        prompt_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_mc_prompt_cot(questions[i], choices_batch[i])
            
            if mtype == "llava-next":
                # Mistral 版本使用 [INST]
                prompt_text = f"[INST] <image>\n{VQA_PROMPT} [/INST]"
            else:
                # 舊版 LLaVA 使用 USER/ASSISTANT
                prompt_text = f"USER: <image>\n{VQA_PROMPT} ASSISTANT:"
                
            prompt_list.append(prompt_text)
        
        inputs = proc(
            images=images, text=prompt_list, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(m.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]
        
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        
        with torch.no_grad():
            outputs = m.generate(**inputs, **kwargs)
        
        new_tokens = outputs[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)

        return [parse_final_answer_mc(ans) for ans in answers_full]

    # ----------------------------------------------------
    # InstructBLIP (批次處理)
    # ----------------------------------------------------
    if mtype == "instructblip":
        prompt_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_mc_prompt_cot(questions[i], choices_batch[i])
            prompt_list.append(VQA_PROMPT)

        inp = proc(
            images=images,
            text=prompt_list,
            return_tensors="pt", 
            padding=True
        ).to(m.device)
        input_len = inp['input_ids'].shape[1]
        
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        with torch.no_grad():
            out = m.generate(**inp, **kwargs)
            
        new_tokens = out[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)
        
        return [parse_final_answer_mc(ans) for ans in answers_full]

    # ----------------------------------------------------
    # Qwen3-VL (批次處理)
    # ----------------------------------------------------
    if mtype == "qwen3-vl":
        messages_list = []
        
        for i in range(batch_size):
            prompt_text = format_mc_prompt_cot(questions[i], choices_batch[i])
            
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[i]},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            messages_list.append(msg)

        texts = [
            proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]

        inputs = proc(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(m.device)

        kwargs.setdefault('max_new_tokens', 200)
        
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = proc.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        return [parse_final_answer_mc(ans) for ans in output_text]
    if mtype == "structural_llava":
        answers = []
        for i in range(batch_size):
            # 1. 格式化輸入 (加入選項 A, B, C, D)
            q_text = format_mc_prompt_cot(questions[i], choices_batch[i])
            
            try:
                # 2. 呼叫模型推論
                # model.generate_answer 內部會處理 Florence 幾何特徵提取
                # 並且自動加上 LLaVA 的 USER/ASSISTANT 標籤
                raw_response = m.generate_answer(images[i], q_text)
                
                # 3. 解析答案
                # 嘗試提取 "Final Answer: C"
                parsed = parse_final_answer_mc(raw_response)
                
                # 如果解析失敗，簡單 fallback
                if not parsed:
                    # 找最後出現的選項字母
                    candidates = re.findall(r"\b([A-D])\b", raw_response.upper())
                    parsed = candidates[-1] if candidates else ""
                
                answers.append(parsed)
                
            except Exception as e:
                print(f"[Error] Sample {i}: {e}")
                answers.append("") # 出錯填空
                
        return answers
        
    return [""] * batch_size


# =================================================================
# 主評估迴圈 (修改自 evaluate_vqa.py)
# =================================================================

def main():
    config_dir = "configs"
    config_filename = 'mcot.yaml' 
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    # 1. 載入 M3CoT 數據集
    ds = M3CoTDataset(
        dataset_id=cfg["dataset_id"],
        split="validation", 
        num_samples=cfg["num_val_samples"]
    )
    
    batch_size = cfg.get('batch_size', 1)
    print(f"Using Batch Size: {batch_size}")
    
    # 2. 使用 utils.py 中的 collate_fn
    ds_loader = DataLoader(
        ds, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_fn 
    )
    
    all_results = {}
    
    # 3. 迴圈評估 mcot.yaml 中定義的所有模型
    for mdl in cfg["models"]:
        name = mdl["name"]
        print(f"\n--- Evaluating Model: {name} ---")
        
        # 4. 使用 utils.py 中的 load_model_and_processor
        # (它會正確載入基礎模型和您已有的 LoRA 模型)
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        torch.compile(mdl_obj, mode="reduce-overhead")
        mdl_obj.eval()
        scores, preds = [], []
        
        generate_params = {
            "max_new_tokens": cfg.get("max_new_tokens", 400),
            "do_sample": False,
            "min_new_tokens": 1,
            "num_beams": 1
        }

        for batch_data in tqdm(ds_loader, desc=name):
            
            images = batch_data["image"]
            questions = batch_data["question"]
            all_answers = batch_data["answers"] # 這是 ['C'] 列表
            question_ids = batch_data["question_id"]
            choices_batch = batch_data["choices"] # <-- 獲取選項

            # 5. 呼叫此檔案中定義的 generate_answer_mc
            ans_list = generate_answer_mc(
                proc, mdl_obj, mtype, 
                images, 
                questions, 
                choices_batch, # 傳入選項
                **generate_params
            )
            
            for i in range(len(ans_list)):
                ans = ans_list[i] # 這是模型輸出的 'C'
                q = questions[i]
                refs = all_answers[i] # 這是 ['C']
                qid = question_ids[i]
                
                if not refs or not all(r.strip() for r in refs):
                    continue
                
                # 6. 呼叫此檔案中定義的 mcot_score
                score = mcot_score(ans, refs) 
                scores.append(score)
                preds.append({"qid": qid, "question": q, "choices": choices_batch[i], "pred": ans, "refs": refs, "score": score})
            del images, questions, all_answers, ans_list
            if 'batch_data' in locals():
                del batch_data
            
            # 這是關鍵：如果 Python 物件沒死，PyTorch 就不能釋放 GPU 記憶體
            gc.collect() 
            torch.cuda.empty_cache()

        if scores:
            mean_acc = (sum(scores) / len(scores)) * 100
        else:
            mean_acc = 0.0
            
        all_results[name] = {"acc": mean_acc, "predictions": preds}
        
        output_file = f"results_mcot_{name.replace(' ', '_').replace('/', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results[name], f, indent=4, ensure_ascii=False)
        print(f"Results for {name} saved to {output_file}")

        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Final M3CoT Evaluation Results ===")
    for k, v in all_results.items():
        print(f"{k}: {v['acc']:.2f}%")

if __name__ == "__main__":
    main()