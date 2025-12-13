import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import yaml
import gc
import json
from torch.utils.data import DataLoader
from dataset.vqa import VQADataset
from tqdm import tqdm
from dataset.utils import load_model_and_processor, generate_answer, collate_fn

class SoMAnnotator:
    def __init__(self, checkpoint_path="sam_vit_l_0b3195.pth", model_type="vit_l", device="cuda"):
        print(f"Loading SAM model ({model_type})... This might take a while.")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        
        # 設定 SAM 參數：這裡稍微調高閾值以減少雜訊 mask
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,  # 忽略太小的區域
        )
        print("SAM model loaded.")

    def apply_som(self, pil_image):
        """
        輸入: 原始 PIL 圖片
        輸出: (標記後的 PIL 圖片, 標記數量, 生成的 masks 資料)
        """
        # 1. 轉為 Numpy 格式供 SAM 使用
        image_np = np.array(pil_image)
        
        # 2. 生成 Masks
        # 注意: 如果圖片太大，SAM 會消耗大量 VRAM
        with torch.no_grad():
            masks = self.mask_generator.generate(image_np)
            
        # 3. 繪製標記
        annotated_img, num_marks = self._draw_masks_and_ids(image_np, masks)
        
        return Image.fromarray(annotated_img), num_marks, masks

    def _draw_masks_and_ids(self, image, masks):
        # 複製圖片以免修改原圖
        annotated = image.copy()
        
        # 根據面積排序 (大 -> 小)，確保小物件的數字不會被大物件的 Mask 蓋住
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        
        # 限制最大標記數量，避免整張圖都是字 (例如取前 60 個顯著物體)
        sorted_masks = sorted_masks[:60]
        
        overlay = annotated.copy()
        
        # 1. 畫 Mask (半透明)
        for i, mask_data in enumerate(sorted_masks):
            color = np.random.randint(0, 255, (3), dtype=np.uint8).tolist()
            m = mask_data['segmentation']
            overlay[m] = color

        # Alpha blending
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

        # 2. 畫 ID 數字
        for i, mask_data in enumerate(sorted_masks):
            bbox = mask_data['bbox'] # [x, y, w, h]
            cx, cy = int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)
            
            label = str(i + 1)
            
            # 計算字體大小 (動態調整)
            font_scale = max(0.5, min(bbox[2], bbox[3]) / 100) 
            thickness = max(1, int(font_scale * 2))
            
            # 畫文字背景 (黑色邊框) 讓字更清楚
            cv2.putText(annotated, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (0, 0, 0), thickness + 2)
            # 畫文字本體 (白色)
            cv2.putText(annotated, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (255, 255, 255), thickness)

        return annotated, len(sorted_masks)
    
# ---------------------------------------------------------
# Stage 1: 預處理函數 (只負責跑 SAM)
# ---------------------------------------------------------
def preprocess_dataset_with_som(cfg, dataset):
    print("\n[Stage 1] Start SoM Pre-processing...")
    
    # 1. 載入 SAM
    # 因為這時候還沒有 LLaVA，我們可以放心用 GPU
    annotator = SoMAnnotator(checkpoint_path="sam_vit_l_0b3195.pth", device="cuda")
    save_dir = "./som_processed_images"
    os.makedirs(save_dir, exist_ok=True)
    
    # 用來儲存處理後圖片的字典: { question_id: (processed_image, num_marks) }
    # 1000 張 PIL 圖片在 RAM 裡大約佔 500MB~1GB，系統記憶體完全撐得住
    processed_cache = {} 
    
    # 這裡我們直接遍歷 dataset，不一定要用 dataloader，因為不需要 batch (SAM 逐張跑比較穩)
    # 但為了方便對應 ID，我們還是簡單用個迴圈
    print(f"Processing {len(dataset)} images...")
    
    for i in tqdm(range(len(dataset)), desc="SAM Processing"):
        sample = dataset[i]
        img = sample['image']
        qid = sample['question_id'] # 這是關鍵，用來對應
        
        try:
            # 產生標記圖
            # 注意：這裡我們回傳 PIL 圖片，存入 RAM
            marked_img, num_marks, _ = annotator.apply_som(img)
            file_name = f"{qid}.jpg"
            save_path = os.path.join(save_dir, file_name)
            marked_img.save(save_path)
            
            # Cache 只存路徑和標記數量，大幅節省 RAM
            processed_cache[qid] = (save_path, num_marks)
        except Exception as e:
            print(f"Error processing img for qid {qid}: {e}")
            # 出錯就存原圖，標記數設為 0
            processed_cache[qid] = (img, 0)
            
    print("[Stage 1] Finished. Cleaning up SAM model...")
    
    # 2. 銷毀 SAM 模型，釋放 GPU
    del annotator
    gc.collect()
    torch.cuda.empty_cache()
    
    return processed_cache

# ---------------------------------------------------------
# Stage 2: 主程式
# ---------------------------------------------------------
def vqa_score(pred, refs):
    pred = pred.strip().lower()
    refs = [a.strip().lower() for a in refs]
    agree = sum(r == pred for r in refs)
    return min(1.0, agree / 3.0)

def main():
    config_dir = "configs"
    config_filename = 'vqa.yaml'
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    # 初始化數據集 (兩個階段都用同一個 dataset 物件)
    ds = VQADataset(cfg["val_img_dir"], cfg["ques_json"], cfg["anns_json"], cfg["num_samples"])

    # =====================================================
    # 執行 Stage 1: 預處理
    # =====================================================
    # 這一步會跑完所有圖片並存到 som_cache 變數中
    som_cache = preprocess_dataset_with_som(cfg, ds)
    
    # =====================================================
    # 執行 Stage 2: 模型推論
    # =====================================================
    print("\n[Stage 2] Start Model Evaluation...")
    
    batch_size = cfg.get('batch_size', 1)
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_results = {}

    for mdl in cfg["models"]:
        name = mdl["name"]
        print(f"\n--- Loading Model: {name} ---")
        
        # 這裡才載入 LLaVA，這時 GPU 應該是乾淨的
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        mdl_obj.eval()
        scores, preds = [], []
        
        generate_params = {
            "max_new_tokens": cfg.get("max_new_tokens", 40),
            "do_sample": False
        }

        for batch_data in tqdm(ds_loader, desc=f"Eval {name}"):
            # 雖然 dataloader 讀了原圖，但我們不使用它
            # 我們用 question_id 去 cache 裡面撈剛剛 SAM 處理好的圖
            
            qids = batch_data["question_id"] # List of IDs
            original_questions = batch_data["question"]
            all_answers = batch_data["answers"]
            
            # 準備這一批次的資料
            batch_som_images = []
            batch_som_questions = []
            
            for i, qid in enumerate(qids):
                # 從 Cache 取出預處理好的圖
                if qid in som_cache:
                    path, num_marks = som_cache[qid]
                    marked_img = Image.open(path).convert("RGB")
                else:
                    # 理論上不會發生，除非 dataset 變動
                    marked_img = batch_data["image"][i]
                    num_marks = 0
                
                batch_som_images.append(marked_img)
                
                # 修改 Prompt
                q_text = original_questions[i]
                new_q = f"The image is overlaid with {num_marks} numeric marks. {q_text} Please refer to the numeric marks in the image to locate objects when reasoning."
                batch_som_questions.append(new_q)

            # 進行推論
            ans_list = generate_answer(
                proc, mdl_obj, mtype, 
                batch_som_images,     # <--- 使用 Cache 裡的圖
                batch_som_questions,  # <--- 使用修改過的 Prompt
                **generate_params
            )
            
            # 計分邏輯 (不變)
            for i in range(len(ans_list)):
                ans = ans_list[i]
                refs = all_answers[i]
                score = vqa_score(ans, refs)
                scores.append(score)
                preds.append({"qid": qids[i], "pred": ans, "refs": refs, "score": score})

            # 清理記憶體
            del batch_som_images, batch_som_questions
            gc.collect()

        # 統計與存檔
        if scores:
            mean_acc = sum(scores) / len(scores)
        else:
            mean_acc = 0.0
        
        all_results[name] = {"acc": mean_acc, "predictions": preds}
        
        output_file = f"results_vqa_som_sequential_{name.replace(' ', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results[name], f, indent=4, ensure_ascii=False)
            
        # 卸載 LLaVA (如果是多模型評估循環)
        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    print("\nFinal Results:")
    for k, v in all_results.items():
        print(f"{k}: {v['acc']:.3f}")

if __name__ == "__main__":
    main()