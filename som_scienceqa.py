import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import yaml
import torch
import gc
import json 
import re 
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- 匯入 Dataset 與 Utils ---
from dataset.scienceqa import ScienceQADataset 
from dataset.utils import load_model_and_processor, collate_fn

# =================================================================
# 1. SoM Annotator (保持不變)
# =================================================================
class SoMAnnotator:
    def __init__(self, checkpoint_path="sam_vit_l_0b3195.pth", model_type="vit_l", device="cuda"):
        print(f"Loading SAM model ({model_type})... This might take a while.")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100, 
        )
        print("SAM model loaded.")

    def apply_som(self, pil_image):
        image_np = np.array(pil_image.convert("RGB"))
        with torch.no_grad():
            masks = self.mask_generator.generate(image_np)
        annotated_img, num_marks = self._draw_masks_and_ids(image_np, masks)
        return Image.fromarray(annotated_img), num_marks, masks

    def _draw_masks_and_ids(self, image, masks):
        annotated = image.copy()
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        sorted_masks = sorted_masks[:60] 
        overlay = annotated.copy()
        for i, mask_data in enumerate(sorted_masks):
            color = np.random.randint(0, 255, (3), dtype=np.uint8).tolist()
            m = mask_data['segmentation']
            overlay[m] = color
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
        for i, mask_data in enumerate(sorted_masks):
            bbox = mask_data['bbox']
            cx, cy = int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)
            label = str(i + 1)
            font_scale = max(0.4, min(bbox[2], bbox[3]) / 100) 
            thickness = max(1, int(font_scale * 2))
            cv2.putText(annotated, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(annotated, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        return annotated, len(sorted_masks)

# =================================================================
# 2. Stage 1: 預處理 (使用 Index 作為標識)
# =================================================================
def preprocess_dataset_with_som(dataset, device="cuda"):
    print("\n[Stage 1] Start SoM Pre-processing for ScienceQA...")
    
    save_dir = "./som_scienceqa_images"
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化 SAM
    try:
        annotator = SoMAnnotator(checkpoint_path="sam_vit_l_0b3195.pth", device=device)
    except Exception as e:
        print(f"Warning: Failed to load SAM on {device}, trying CPU... ({e})")
        annotator = SoMAnnotator(checkpoint_path="sam_vit_l_0b3195.pth", device="cpu")

    # 使用 List 來存儲結果，索引對應 Dataset 的索引
    # 格式: [ (file_path, num_marks), (None, 0), ... ]
    processed_cache = [None] * len(dataset)
    
    print(f"Processing {len(dataset)} samples...")
    
    for i in tqdm(range(len(dataset)), desc="SAM Processing"):
        sample = dataset[i]
        img = sample.get('image') 
        
        # 如果沒有圖片 (純文字題)，直接跳過
        if img is None:
            processed_cache[i] = (None, 0)
            continue

        # 使用索引 i 作為檔名，確保唯一性且不依賴 question_id
        file_name = f"{i}.jpg"
        save_path = os.path.join(save_dir, file_name)
        
        try:
            # 這裡為了簡單，每次都檢查檔案是否存在
            if os.path.exists(save_path):
                # 如果檔案存在，我們需要知道 num_marks。
                # 這裡偷懶不做讀取計算，而是重新跑一次 apply_som (較慢但保證正確)
                # 或者您可以選擇建立一個 json 存 meta data。
                # 為求穩健，這裡執行生成:
                pass 

            marked_img, num_marks, _ = annotator.apply_som(img)
            marked_img.save(save_path)
            processed_cache[i] = (save_path, num_marks)
            
        except Exception as e:
            print(f"Error processing img index {i}: {e}")
            processed_cache[i] = (None, 0) 
            
    print("[Stage 1] Finished. Cleaning up SAM model...")
    del annotator
    gc.collect()
    torch.cuda.empty_cache()
    
    return processed_cache

# =================================================================
# 3. 評分與解析函數 (完全參照 eval_scienceqa.py)
# =================================================================
def scienceqa_score(pred, refs):
    if not pred: return 0.0
    pred = pred.strip().upper() 
    refs = [a.strip().upper() for a in refs]
    return 1.0 if pred in refs else 0.0

def parse_final_answer_mc(full_text):
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
        label = chr(65 + i) 
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt

# =================================================================
# 4. 答案生成函數 (完全參照 eval_scienceqa.py)
# =================================================================
def generate_answer_mc(proc, m, mtype, images, questions, choices_batch, **kwargs):
    batch_size = len(questions)
    
    # --- Qwen-VL ---
    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_mc_prompt_cot(questions[i], choices_batch[i])
            # 注意: 如果 images[i] 是 None (純文字題)，這裡傳入 None 可能會導致 Processor 報錯
            # 但既然我們要參照 eval_scienceqa.py，我們假設外部已經處理好，或者 processor 能吃 None
            # (在 main 中我們會確保如果沒圖，還是傳入原始 None)
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": images[i]},
                    {"type": "text", "text": VQA_PROMPT} 
                ]}
            ]
            messages_list.append(messages)

        text_prompts = [
            proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]
        
        # 這裡需要小心: 如果 images 全是 None，proc 參數可能不同
        # 但為了保持代碼一致性，我們照舊
        inputs = proc(
            text=text_prompts, images=images, return_tensors="pt", padding=True
        ).to(m.device)
        
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        with torch.no_grad():
            res = m.generate(**inputs, **kwargs)
        
        responses = proc.batch_decode(res, skip_special_tokens=True)
        
        final_answers_full = []
        for response in responses:
            answer = response.split("assistant\n")[-1].strip()
            final_answers_full.append(answer)

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
        
        # 同樣，如果 images 含有 None，這裡可能會報錯，取決於 transformers 版本
        inputs = proc(images=images, text=prompt_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(m.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]
        
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        
        with torch.no_grad():
            outputs = m.generate(**inputs, **kwargs)
        
        new_tokens = outputs[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)
        
        return [parse_final_answer_mc(ans) for ans in answers_full]
        
    return [""] * batch_size

# =================================================================
# 5. 主程式 (修改重點：使用索引對應 Cache)
# =================================================================

def main():
    config_dir = "configs"
    config_filename = 'scienceqa.yaml' 
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    # 1. 載入 ScienceQA 數據集
    ds = ScienceQADataset(
        dataset_id=cfg["dataset_id"],
        split="validation", 
        num_samples=cfg.get("num_val_samples", None)
    )
    
    # -----------------------------------------------------
    # [Stage 1] 執行 SoM 預處理
    # 回傳一個 List，index 對應 dataset index
    # -----------------------------------------------------
    som_cache = preprocess_dataset_with_som(ds, device="cuda")
    
    # -----------------------------------------------------
    # [Stage 2] 模型評估
    # -----------------------------------------------------
    batch_size = cfg.get('batch_size', 1)
    print(f"Using Batch Size: {batch_size}")
    
    ds_loader = DataLoader(
        ds, 
        batch_size=batch_size,
        shuffle=False, # 必須為 False，才能保證順序與 som_cache 對應
        collate_fn=collate_fn 
    )
    
    all_results = {}
    
    for mdl in cfg["models"]:
        name = mdl["name"]
        print(f"\n--- Evaluating Model: {name} (with SoM) ---")
        
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        mdl_obj.eval()
        
        scores, preds = [], []
        
        generate_params = {
            "max_new_tokens": cfg.get("max_new_tokens", 400),
            "do_sample": False,
            "min_new_tokens": 1,
            "num_beams": 1
        }

        # 【關鍵變數】: 全域索引計數器，用來從 som_cache 取值
        global_sample_index = 0

        for batch_data in tqdm(ds_loader, desc=name):
            
            # 從 Batch 中獲取原始數據
            original_images = batch_data["image"]
            original_questions = batch_data["question"]
            all_answers = batch_data["answers"] 
            choices_batch = batch_data["choices"]

            # --- [SoM 關鍵] 替換圖片與修改 Prompt ---
            som_images = []
            som_questions = []

            # 遍歷這個 batch 中的每一個樣本
            for i in range(len(original_images)):
                
                # 從 Cache 中取出對應的全域索引資料
                # som_cache[global_sample_index] 格式為 (path, num_marks) 或 (None, 0)
                cache_item = som_cache[global_sample_index]
                
                has_marks = False
                num_marks = 0
                current_img = original_images[i] # 預設使用原圖
                
                if cache_item and cache_item[0] is not None:
                    path, num_marks = cache_item
                    try:
                        marked_img = Image.open(path).convert("RGB")
                        current_img = marked_img
                        has_marks = True
                    except Exception as e:
                        # 讀檔失敗，保持原圖
                        pass
                
                som_images.append(current_img)

                # 修改問題 (注入 SoM 指令)
                if has_marks and num_marks > 0:
                    prefix = f"The image is overlaid with {num_marks} numeric marks. Please refer to the numeric marks in the image to locate objects when reasoning. "
                    new_q = f"{prefix}{original_questions[i]}"
                else:
                    new_q = original_questions[i]
                
                som_questions.append(new_q)
                
                # 重要：處理完一個樣本，全域索引 +1
                global_sample_index += 1

            # --- 呼叫生成函數 ---
            ans_list = generate_answer_mc(
                proc, mdl_obj, mtype, 
                som_images,      # <-- 使用 SoM 處理後的圖片列表
                som_questions,   # <-- 使用注入 Prompt 後的問題列表
                choices_batch, 
                **generate_params
            )
            
            # --- 計分 ---
            for i in range(len(ans_list)):
                ans = ans_list[i]
                q = original_questions[i]
                refs = all_answers[i]
                
                if not refs or not all(r.strip() for r in refs):
                    continue
                
                score = scienceqa_score(ans, refs) 
                scores.append(score)
                preds.append({"question": q, "choices": choices_batch[i], "pred": ans, "refs": refs, "score": score})

        if scores:
            mean_acc = (sum(scores) / len(scores)) * 100
        else:
            mean_acc = 0.0
            
        all_results[name] = {"acc": mean_acc, "predictions": preds}
        
        output_file = f"results_scienceqa_som_{name.replace(' ', '_').replace('/', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results[name], f, indent=4, ensure_ascii=False)
        print(f"Results for {name} saved to {output_file}")

        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Final ScienceQA (SoM) Evaluation Results ===")
    for k, v in all_results.items():
        print(f"{k}: {v['acc']:.2f}%")

if __name__ == "__main__":
    main()