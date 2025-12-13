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
# 假設您的資料集路徑是 dataset/scienceqa.py
from dataset.scienceqa import ScienceQADataset 
from eval_scienceqa import generate_answer_mc
from dataset.utils import load_model_and_processor, collate_fn

# =================================================================
# 1. SoM Annotator (負責生成標記 - 與之前通用)
# =================================================================
class SoMAnnotator:
    def __init__(self, checkpoint_path="sam_vit_l_0b3195.pth", model_type="vit_l", device="cuda"):
        print(f"Loading SAM model ({model_type})... This might take a while.")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        
        # 設定 SAM 參數
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
        # 轉為 Numpy 且確保是 RGB
        image_np = np.array(pil_image.convert("RGB"))
        
        with torch.no_grad():
            masks = self.mask_generator.generate(image_np)
            
        annotated_img, num_marks = self._draw_masks_and_ids(image_np, masks)
        return Image.fromarray(annotated_img), num_marks, masks

    def _draw_masks_and_ids(self, image, masks):
        annotated = image.copy()
        # 根據面積排序 (大 -> 小)
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        # 限制標記數量
        sorted_masks = sorted_masks[:60] 
        
        overlay = annotated.copy()
        
        # 1. 畫 Mask
        for i, mask_data in enumerate(sorted_masks):
            color = np.random.randint(0, 255, (3), dtype=np.uint8).tolist()
            m = mask_data['segmentation']
            overlay[m] = color

        alpha = 0.4
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

        # 2. 畫 ID
        for i, mask_data in enumerate(sorted_masks):
            bbox = mask_data['bbox']
            cx, cy = int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)
            label = str(i + 1)
            
            # 動態調整字體大小
            font_scale = max(0.4, min(bbox[2], bbox[3]) / 100) 
            thickness = max(1, int(font_scale * 2))
            
            # 黑色描邊 + 白色字體
            cv2.putText(annotated, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(annotated, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (255, 255, 255), thickness)

        return annotated, len(sorted_masks)

# =================================================================
# 2. Stage 1: 預處理並存檔 (針對 ScienceQA 特別處理無圖情況)
# =================================================================
def preprocess_dataset_with_som(dataset, device="cuda"):
    print("\n[Stage 1] Start SoM Pre-processing for ScienceQA...")
    
    save_dir = "./som_scienceqa_images"
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化 SAM
    try:
        annotator = SoMAnnotator(checkpoint_path="sam_vit_h_4b8939.pth", device=device)
    except Exception as e:
        print(f"Warning: Failed to load SAM on {device}, trying CPU... ({e})")
        annotator = SoMAnnotator(checkpoint_path="sam_vit_h_4b8939.pth", device="cpu")

    processed_cache = {} 
    print(f"Processing {len(dataset)} samples...")
    
    for i in tqdm(range(len(dataset)), desc="SAM Processing"):
        sample = dataset[i]
        img = sample.get('image') # ScienceQA 可能沒有圖片
        qid = str(sample['question_id']) # 確保 ID 是字串
        
        # 如果沒有圖片 (純文字題)，直接跳過
        if img is None:
            processed_cache[qid] = (None, 0)
            continue

        file_name = f"{qid}.jpg"
        save_path = os.path.join(save_dir, file_name)
        
        # 如果檔案已存在，可以選擇跳過 (這裡演示覆寫)
        try:
            marked_img, num_marks, _ = annotator.apply_som(img)
            marked_img.save(save_path)
            processed_cache[qid] = (save_path, num_marks)
        except Exception as e:
            print(f"Error processing img for qid {qid}: {e}")
            processed_cache[qid] = (None, 0) # 標記失敗
            
    print("[Stage 1] Finished. Cleaning up SAM model...")
    del annotator
    gc.collect()
    torch.cuda.empty_cache()
    
    return processed_cache

# =================================================================
# 3. ScienceQA 專用評分邏輯 (複製自 evaluate_scienceqa.py)
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
    """
    格式化 CoT 提示。注意：我們會在 main 函數中將 'SoM 指令' 注入到 question 變數中。
    """
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
    prompt += f"Question: {question}\n\n" # <--- SoM 指令會跟著 question 進來
    prompt += "Options:\n"
    for i, choice in enumerate(choices_list):
        label = chr(65 + i) 
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt

# =================================================================
# 4. 主程式 (整合 SoM 與 ScienceQA)
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
    # -----------------------------------------------------
    som_cache = preprocess_dataset_with_som(ds, device="cuda")
    
    # -----------------------------------------------------
    # [Stage 2] 模型評估
    # -----------------------------------------------------
    batch_size = cfg.get('batch_size', 1)
    print(f"Using Batch Size: {batch_size}")
    
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
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

        for batch_data in tqdm(ds_loader, desc=name):
            
            # 從 Batch 中獲取原始數據
            original_images = batch_data["image"]
            original_questions = batch_data["question"]
            all_answers = batch_data["answers"] 
            question_ids = batch_data["question_id"]
            choices_batch = batch_data["choices"]

            # --- [SoM 關鍵] 替換圖片與修改 Prompt ---
            som_images = []
            som_questions = []

            for i, raw_qid in enumerate(question_ids):
                qid = str(raw_qid)
                
                # 1. 讀取圖片 (處理無圖情況)
                # ScienceQA 有些樣本沒圖片，original_images[i] 可能是 None 或特殊物件
                # Check cache first
                if qid in som_cache and som_cache[qid][0] is not None:
                    # 有 SoM 標記的圖
                    path, num_marks = som_cache[qid]
                    try:
                        marked_img = Image.open(path).convert("RGB")
                        som_images.append(marked_img)
                        has_marks = True
                    except:
                        # 讀檔失敗，退回原圖
                        som_images.append(original_images[i])
                        num_marks = 0
                        has_marks = False
                else:
                    # 沒有 SoM Cache (可能是純文字題，或標記失敗)
                    som_images.append(original_images[i])
                    num_marks = 0
                    has_marks = False

                # 2. 修改問題 (注入 SoM 指令)
                # 只有當「有圖片」且「有標記」時，才加入提示詞
                if has_marks and num_marks > 0:
                    prefix = f"The image is overlaid with {num_marks} numeric marks. Please refer to the numeric marks in the image to locate objects when reasoning. "
                    new_q = f"{prefix}{original_questions[i]}"
                else:
                    new_q = original_questions[i]
                
                som_questions.append(new_q)

            # --- 呼叫生成函數 ---
            ans_list = generate_answer_mc(
                proc, mdl_obj, mtype, 
                som_images,      # <-- 使用 SoM 圖片 (或原圖/None)
                som_questions,   # <-- 使用 SoM 問題
                choices_batch, 
                **generate_params
            )
            
            # --- 計分 ---
            for i in range(len(ans_list)):
                ans = ans_list[i]
                refs = all_answers[i]
                qid = str(question_ids[i])
                
                if not refs or not all(r.strip() for r in refs):
                    continue
                
                score = scienceqa_score(ans, refs) 
                scores.append(score)
                preds.append({
                    "qid": qid, 
                    "question": original_questions[i], 
                    "choices": choices_batch[i],
                    "pred": ans, 
                    "refs": refs, 
                    "score": score,
                    "som_applied": (qid in som_cache and som_cache[qid][0] is not None)
                })

            del som_images, som_questions, original_images
            gc.collect()
            torch.cuda.empty_cache()

        # 統計
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