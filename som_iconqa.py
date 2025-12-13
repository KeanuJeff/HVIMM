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
from eval_iconqa import generate_answer_vqa, generate_answer_mc
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- 匯入 Dataset 與 Utils ---
from dataset.iconqa import IconQADataset
# 假設 generate_answer_mc 等函數都在這個腳本內或是從某處匯入，這裡沿用您提供的 eval_iconqa.py 結構
# 為了方便，我們假設 utils 邏輯與 eval_iconqa.py 相同，這裡直接從您的模組匯入
from dataset.utils import load_model_and_processor, collate_fn

# =================================================================
# 1. SoM Annotator (與 som_vqa.py 相同)
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
        # 轉為 Numpy 且確保是 RGB (IconQA 可能有 RGBA 或 Grayscale)
        image_np = np.array(pil_image.convert("RGB"))
        
        with torch.no_grad():
            masks = self.mask_generator.generate(image_np)
            
        annotated_img, num_marks = self._draw_masks_and_ids(image_np, masks)
        return Image.fromarray(annotated_img), num_marks, masks

    def _draw_masks_and_ids(self, image, masks):
        annotated = image.copy()
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        sorted_masks = sorted_masks[:60] # 限制標記數量
        
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
            
            # IconQA 圖片可能較小，字體大小要做防呆
            font_scale = max(0.4, min(bbox[2], bbox[3]) / 100) 
            thickness = max(1, int(font_scale * 2))
            
            cv2.putText(annotated, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(annotated, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (255, 255, 255), thickness)

        return annotated, len(sorted_masks)

# =================================================================
# 2. Stage 1: 預處理 (針對 IconQA)
# =================================================================
def preprocess_dataset_with_som(dataset, device="cuda"):
    print("\n[Stage 1] Start SoM Pre-processing for IconQA...")
    
    # 初始化 SAM (使用 try-except 處理可能的顯存問題)
    try:
        annotator = SoMAnnotator(checkpoint_path="sam_vit_h_4b8939.pth", device=device)
    except Exception as e:
        print(f"Warning: Failed to load SAM on {device}, trying CPU... ({e})")
        annotator = SoMAnnotator(checkpoint_path="sam_vit_h_4b8939.pth", device="cpu")

    save_dir = "./som_iconqa_images"
    os.makedirs(save_dir, exist_ok=True)
    
    processed_cache = {} 
    
    print(f"Processing {len(dataset)} images...")
    
    for i in tqdm(range(len(dataset)), desc="SAM Processing"):
        sample = dataset[i]
        img = sample['image']
        qid = sample['question_id']
        
        # 檢查是否已經跑過 (斷點續傳)
        file_name = f"{qid}.jpg"
        save_path = os.path.join(save_dir, file_name)
        
        if os.path.exists(save_path):
             # 偷懶做法：如果不讀檔就算不出 num_marks，這裡設為 -1 或重新讀取
             # 為了精確，我們假設若檔案存在，則不再生成，但在這裡我們需要 num_marks 
             # 建議：如果是正式跑，可以把 metadata 存成 json。
             # 這裡簡單起見，我們如果檔案存在就不重跑 SAM，但需要重讀圖片來畫 Prompt 嗎？
             # 其實 Prompt 需要 num_marks。如果沒有存 meta，就得重跑或略過。
             # 簡單策略：覆寫，或假設已存在。這裡演示「覆寫」以確保正確。
             pass 

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
# 3. 評分與生成函數 (複製自 eval_iconqa.py)
# =================================================================
# ... (這裡請貼上您 eval_iconqa.py 中的 score_mc_acc, parse_final_answer_mc, 
#      format_mc_prompt_cot, generate_answer_mc, 
#      score_vqa, parse_final_answer_vqa, format_vqa_prompt, generate_answer_vqa) ...
# 為了篇幅，我這裡直接引用您提供的邏輯，假設它們已經定義好了。

# (以下重新定義 format 函數以示範如何不需要修改 generate 函數就能注入 SoM Prompt)

def score_mc_acc(pred, refs):
    if not pred: return 0.0
    pred = pred.strip().upper() 
    refs = [a.strip().upper() for a in refs]
    return 1.0 if pred in refs else 0.0

def parse_final_answer_mc(full_text):
    match = re.search(r"Final Answer:\s*([A-Z])", full_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"Answer:\s*([A-Z])", full_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", full_text)
    if matches: return matches[-1].upper() 
    return ""

def format_mc_prompt_cot(question, choices_list):
    # 這裡 question 會被傳入 "The image is overlaid... Question: ..."
    prompt = "Answer the following multiple-choice question either directly or with some reasoning. Conclude the final answer with the correct option letter in the format 'Final Answer: [Letter]'.\n\n"
    prompt += f"{question}\n\n" # 注意這裡直接放 question
    prompt += "Options:\n"
    for i, choice in enumerate(choices_list):
        label = chr(65 + i)
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer: "
    return prompt


def score_vqa(pred, refs):
    if not pred: return 0.0
    pred = pred.strip().lower()
    refs = [a.strip().lower() for a in refs]
    return 1.0 if pred in refs else 0.0

def parse_final_answer_vqa(full_text):
    try:
        split_tag = "Final Answer:"
        parts = full_text.split(split_tag)
        if len(parts) < 2: return full_text.strip().split()[-1]
        final_part = parts[-1]
        final_answer = final_part.strip().split('\n')[0].strip().split(' ')[0].strip()
        final_answer = final_answer.rstrip('.,')
        return final_answer
    except: return ""

def format_vqa_prompt(question):
    return f"Answer the following question either directly or with reasoning. Give the final answer in a single word like 'Final Answer: [single word]'.\n\n{question}\nAnswer:"


# =================================================================
# 4. 主程式 (整合 SoM)
# =================================================================

def main():
    config_dir = "configs"
    config_filename = 'iconqa.yaml'
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    # 1. 載入 Dataset
    ds = IconQADataset(
        dataset_id=cfg.get("dataset_id", "lmms-lab/ICON-QA"),
        split="validation", 
        num_samples=cfg.get("num_val_samples", None)
    )
    
    # =====================================================
    # [Stage 1] 執行 SoM 預處理 (生成 Mask 並存檔)
    # =====================================================
    # 這裡如果不想每次都跑，可以加入一個判斷，讀取已存在的 cache json
    som_cache = preprocess_dataset_with_som(ds, device="cuda")

    # =====================================================
    # [Stage 2] 模型評估
    # =====================================================
    batch_size = cfg.get('batch_size', 1)
    print(f"Using Batch Size: {batch_size}")
    
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_results = {}
    
    for mdl in cfg["models"]:
        name = mdl["name"]
        print(f"\n--- Evaluating Model: {name} (with SoM) ---")
        
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        mdl_obj.eval()
        
        scores_mc, scores_vqa, preds = [], [], []
        
        generate_params_mc = {
            "max_new_tokens": cfg.get("max_new_tokens_mc", 200),
            "do_sample": False
        }
        generate_params_vqa = {
            "max_new_tokens": cfg.get("max_new_tokens_vqa", 100),
            "do_sample": False
        }

        for batch_data in tqdm(ds_loader, desc=name):
            
            # 從 Batch 中獲取原始數據
            original_images = batch_data["image"]
            original_questions = batch_data["question"]
            all_answers = batch_data["answers"] 
            question_ids = batch_data["question_id"]
            choices_batch = batch_data["choices"]
            question_types = batch_data["question_type"]

            # --- [SoM 關鍵步驟] 替換圖片與修改 Prompt ---
            som_images = []
            som_questions = []

            for i, qid in enumerate(question_ids):
                # 1. 替換圖片: 從 Cache 讀取帶有標記的圖
                if qid in som_cache and som_cache[qid][0] is not None:
                    path, num_marks = som_cache[qid]
                    try:
                        marked_img = Image.open(path).convert("RGB")
                        som_images.append(marked_img)
                    except:
                        # 讀檔失敗就用原圖
                        som_images.append(original_images[i])
                        num_marks = 0
                else:
                    som_images.append(original_images[i])
                    num_marks = 0

                # 2. 修改 Prompt: 插入 SoM 提示詞
                # 這段提示詞 會被視為 Question 的一部分傳入 format 函數
                if num_marks > 0:
                    prefix = f"The image is overlaid with {num_marks} numeric marks. Please refer to the numeric marks in the image to locate objects when reasoning. "
                    new_q = f"{prefix}Question: {original_questions[i]}"
                else:
                    new_q = f"Question: {original_questions[i]}"
                
                som_questions.append(new_q)

            # --- 分離 MCQ 與 VQA (使用修改後的圖片與問題) ---
            mc_indices = [i for i, t in enumerate(question_types) if t == 'multiple-choice']
            vqa_indices = [i for i, t in enumerate(question_types) if t == 'open-ended']
            
            results = [None] * len(som_questions)

            # 處理 MCQ
            if mc_indices:
                mc_imgs = [som_images[i] for i in mc_indices]
                mc_qs = [som_questions[i] for i in mc_indices] # 這裡已經包含 SoM 提示
                mc_choices = [choices_batch[i] for i in mc_indices]
                
                mc_ans_list = generate_answer_mc(
                    proc, mdl_obj, mtype, 
                    mc_imgs, mc_qs, mc_choices, 
                    **generate_params_mc
                )
                for i, ans in enumerate(mc_ans_list):
                    results[mc_indices[i]] = (ans, 'mc')

            # 處理 VQA
            if vqa_indices:
                vqa_imgs = [som_images[i] for i in vqa_indices]
                vqa_qs = [som_questions[i] for i in vqa_indices]
                
                vqa_ans_list = generate_answer_vqa(
                    proc, mdl_obj, mtype, 
                    vqa_imgs, vqa_qs, 
                    **generate_params_vqa
                )
                for i, ans in enumerate(vqa_ans_list):
                    results[vqa_indices[i]] = (ans, 'vqa')

            # --- 統計分數 (與原本相同) ---
            for i in range(len(results)):
                if results[i] is None: continue
                
                ans, q_type = results[i]
                refs = all_answers[i]
                
                if "INVALID" in refs or not refs: continue
                
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
                    "question": original_questions[i], # 存檔時保留原始問題比較乾淨
                    "pred": ans, 
                    "refs": refs, 
                    "score": score,
                    "som_applied": (qid in som_cache and som_cache[qid][0] is not None)
                })

            # 清理
            del som_images, som_questions, original_images
            gc.collect()
            torch.cuda.empty_cache()

        # 儲存結果
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
        
        output_file = f"results_iconqa_som_{name.replace(' ', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results[name], f, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_file}")

        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()