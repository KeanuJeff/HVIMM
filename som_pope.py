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
from sklearn.metrics import classification_report, accuracy_score
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- 匯入 Dataset 與 Utils ---
from dataset.pope import POPEDataset
from eval_pope.py import generate_answer_yesno
from dataset.utils import load_model_and_processor, collate_fn

# =================================================================
# 1. SoM Annotator (通用標記生成器)
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
# 2. Stage 1: 預處理 (針對 POPE 的不同 Category)
# =================================================================
def preprocess_dataset_with_som(dataset, category_name, device="cuda"):
    print(f"\n[Stage 1] Start SoM Pre-processing for POPE ({category_name})...")
    
    # 建立分類別的資料夾
    save_dir = f"./som_pope_images/{category_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        annotator = SoMAnnotator(checkpoint_path="sam_vit_l_0b3195.pth", device=device)
    except Exception as e:
        print(f"Warning: Failed to load SAM on {device}, trying CPU... ({e})")
        annotator = SoMAnnotator(checkpoint_path="sam_vit_l_0b3195.pth", device="cpu")

    processed_cache = {} 
    
    # 為了避免重複處理相同的圖片 (POPE 不同問題可能用同一張圖)，我們可以檢查檔案是否存在
    # 但為了邏輯簡單，這裡還是以 qid 為準
    
    print(f"Processing {len(dataset)} samples...")
    for i in tqdm(range(len(dataset)), desc="SAM Processing"):
        sample = dataset[i]
        img = sample['image']
        qid = str(sample['question_id'])
        
        file_name = f"{qid}.jpg"
        save_path = os.path.join(save_dir, file_name)
        
        # Check cache/disk
        if os.path.exists(save_path):
            # 這裡我們需要 num_marks，如果不想重跑 SAM，可以建立另一個 json 存 meta data
            # 這裡為了演示方便，我們假設如果檔案存在，我們重新讀取它來獲取標記並不是最快的方法
            # 簡單策略: 重新生成 (確保 num_marks 正確) 或者 讀取已存在的圖並簡單計算 (不精確)
            # **最佳策略**: 這裡演示「若檔案存在則跳過生成，但在 Stage 2 載入時才算 marks (或是設為預設值)」
            # 為了保證 Prompt 的 num_marks 正確，我們還是跑一次 apply_som 比較保險，除非我們有存 meta。
            # 考慮到 POPE 只有 500 張圖，跑一次很快。
            pass 

        try:
            marked_img, num_marks, _ = annotator.apply_som(img)
            marked_img.save(save_path)
            processed_cache[qid] = (save_path, num_marks)
        except Exception as e:
            print(f"Error processing img for qid {qid}: {e}")
            processed_cache[qid] = (None, 0)
            
    print("[Stage 1] Finished. Cleaning up SAM model...")
    del annotator
    gc.collect()
    torch.cuda.empty_cache()
    
    return processed_cache

# =================================================================
# 3. POPE 評分與解析函數 (來自 evaluate_pope.py)
# =================================================================

def score_yesno(pred, refs):
    if not pred: return 0.0
    pred = pred.strip().capitalize()
    refs = [a.strip().capitalize() for a in refs]
    return 1.0 if pred in refs else 0.0

def parse_final_answer_yesno(full_text):
    match = re.search(r"Final Answer:\s*(Yes|No)", full_text, re.IGNORECASE)
    if match: return match.group(1).capitalize()
    match = re.search(r"Answer:\s*(Yes|No)", full_text, re.IGNORECASE)
    if match: return match.group(1).capitalize()
    matches = re.findall(r"\b(Yes|No)\b", full_text, re.IGNORECASE)
    if matches: return matches[-1].capitalize()
    return ""

def format_pope_prompt(question):
    # 注意：這裡的 question 在 main 函數中會被加上 "The image is overlaid..." 前綴
    prompt = ("Based on the image, answer the following Yes/No question. You can either answer directly or with some reasoning. State your final answer in the format 'Final Answer: [Yes/No]'.\n\n")
    prompt += f"Question: {question}\n"
    prompt += ("Answer:\n")
    return prompt

# =================================================================
# 4. 主程式 (整合 SoM 與 POPE Grouped Eval)
# =================================================================

def main():
    config_dir = "configs"
    config_filename = 'pope.yaml'
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    TARGET_CATEGORIES = ["random", "popular", "adversarial"]
    SAMPLES_PER_CATEGORY = 500
    
    batch_size = cfg.get('batch_size', 4)
    print(f"Using Batch Size: {batch_size}")
    
    all_results = {}
    
    for mdl in cfg["models"]:
        name = mdl["name"]
        print(f"\n--- Evaluating Model: {name} (with SoM) ---")
        
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        mdl_obj.eval()
        
        model_category_results = {}

        generate_params = {
            "max_new_tokens": cfg.get("max_new_tokens", 300),
            "do_sample": False,
            "min_new_tokens": 1,
            "num_beams": 1
        }

        # --- 迴圈處理三種類別 ---
        for category in TARGET_CATEGORIES:
            print(f"\n--- Running Category: {category.upper()} ---")
            
            # 1. 載入 Dataset
            ds = POPEDataset(
                dataset_id=cfg["dataset_id"],
                split="test", 
                target_category=category,
                num_samples_per_category=SAMPLES_PER_CATEGORY
            )
            
            # 2. [Stage 1] 針對當前類別執行 SoM 預處理
            #    注意：因為 POPE 三個類別的圖片可能有重疊，但 Question ID 不同，
            #    所以我們針對每個類別分開跑一次處理比較單純。
            som_cache = preprocess_dataset_with_som(ds, category_name=category, device="cuda")
            
            ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
            all_true_labels, all_pred_labels = [], []
            preds_list_for_json = []

            for batch_data in tqdm(ds_loader, desc=f"{name} ({category})"):
                
                original_images = batch_data["image"]
                original_questions = batch_data["question"]
                all_answers = batch_data["answers"] 
                question_ids = batch_data["question_id"]
                
                # --- [SoM 關鍵] 替換圖片 & 修改問題 ---
                som_images = []
                som_questions = []

                for i, raw_qid in enumerate(question_ids):
                    qid = str(raw_qid)
                    
                    # 替換圖片
                    if qid in som_cache and som_cache[qid][0] is not None:
                        path, num_marks = som_cache[qid]
                        try:
                            marked_img = Image.open(path).convert("RGB")
                            som_images.append(marked_img)
                        except:
                            som_images.append(original_images[i])
                            num_marks = 0
                    else:
                        som_images.append(original_images[i])
                        num_marks = 0

                    # 修改問題 (注入 SoM 指令)
                    # 對於 POPE 這種 Yes/No 問題，加上 "locate objects" 的指令可以幫助模型
                    # 確認該物體是否存在於某個標記上
                    if num_marks > 0:
                        prefix = f"The image is overlaid with {num_marks} numeric marks. Please refer to the numeric marks in the image to locate objects when reasoning. "
                        new_q = f"{prefix}{original_questions[i]}"
                    else:
                        new_q = original_questions[i]
                    
                    som_questions.append(new_q)
                
                # --- 生成答案 ---
                ans_list = generate_answer_yesno(
                    proc, mdl_obj, mtype, 
                    som_images,     # 使用標記圖
                    som_questions,  # 使用修改後的問題
                    **generate_params
                )
                
                # --- 收集結果 ---
                for i in range(len(ans_list)):
                    pred_ans = ans_list[i]
                    true_refs = all_answers[i]
                    true_label = true_refs[0]
                    qid = str(question_ids[i])
                    
                    if not true_label or true_label == "INVALID": continue
                    
                    all_true_labels.append(true_label)
                    all_pred_labels.append(pred_ans)
                    
                    score = score_yesno(pred_ans, true_refs)
                    preds_list_for_json.append({
                        "qid": qid,
                        "category": category,
                        "question": original_questions[i], # 存原始問題
                        "pred": pred_ans,
                        "refs": true_refs,
                        "score": score,
                        "som_applied": (qid in som_cache and som_cache[qid][0] is not None)
                    })

                del som_images, som_questions, original_images
                gc.collect()
                torch.cuda.empty_cache()

            # --- 計算 Metrics ---
            if not all_true_labels:
                print(f"Skipping category {category}: No data processed.")
                continue

            mean_acc = accuracy_score(all_true_labels, all_pred_labels) * 100
            labels = sorted(list(set(all_true_labels + all_pred_labels)))
            # 避免 zero_division 警告
            report_dict = classification_report(
                all_true_labels, all_pred_labels, labels=labels, output_dict=True, zero_division=0
            )

            model_category_results[category] = {
                "count": len(all_true_labels),
                "accuracy": mean_acc,
                "metrics": report_dict,
                "predictions": preds_list_for_json
            }
            
            print(f"  Accuracy: {mean_acc:.2f}%")

        # --- 儲存該模型的所有結果 ---
        all_results[name] = model_category_results
        
        output_file = f"results_pope_som_grouped_{name.replace(' ', '_').replace('/', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results[name], f, indent=4, ensure_ascii=False)
        print(f"\n--- Grouped results for {name} saved to {output_file} ---")

        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Final POPE (SoM) Grouped Metrics Summary ===")
    for model_name, results in all_results.items():
        print(f"\nModel: {model_name}")
        for category, data in results.items():
            print(f"  {category.upper()} (N={data['count']}): Accuracy={data['accuracy']:.2f}%")

if __name__ == "__main__":
    main()