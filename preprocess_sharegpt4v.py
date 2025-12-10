import gc
import torch
import os
import sys
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# 載入模型 Class (假設你的目錄結構沒變)
from models.structural_llava_next import HybirdLlavaFlorenceModel

# ==========================================
# 設定區
# ==========================================
# 如果你的 dataset 載入的是圖片路徑字串 (例如 "coco/train2017/..."), 請設定 COCO 圖片的根目錄
# 如果 dataset 自動載入的是 PIL Image 物件，這個變數會被忽略
COCO_IMAGE_DIR = "./dataset/train2017" 
OUTPUT_DIR = "./processed_features_sharegpt4v1"

def preprocess_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model (Florence ONLY) on {device}...")
    
    # 只載入 Florence-2 來抓取物體結構
    model = HybirdLlavaFlorenceModel(
        load_llava=False, 
        load_florence=True,
        florence_model_id="./models/local_florence2" # 或 "microsoft/Florence-2-large"
    )
    
    # 載入 ShareGPT4V Dataset
    print(f"Loading ShareGPT4V Dataset...")
    # ShareGPT4V 只有 train split (102k rows)
    try:
        ds = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    processed_ids = set()
    print(f"Start Preprocessing ShareGPT4V...")
    
    for i, item in tqdm(enumerate(ds)):
        # 1. 取得 ID
        # ShareGPT4V 的 id 是 string, e.g., "000000000009"
        if 'id' in item:
            img_id = str(item['id'])
        else:
            img_id = str(i)
            
        save_path = os.path.join(OUTPUT_DIR, f"raw_{img_id}.pt")
        
        # 檢查是否已處理
        if os.path.exists(save_path):
            processed_ids.add(img_id)
            continue
        if img_id in processed_ids: 
            continue

        # 2. 處理圖片
        image_data = item['image']
        image = None

        if isinstance(image_data, Image.Image):
            # 情況 A: HuggingFace 已經轉成 PIL
            image = image_data.convert("RGB")
        elif isinstance(image_data, str):
            # 情況 B: 只是路徑字串 (通常 ShareGPT4V 是這種情況)
            # 嘗試組合路徑
            full_path = os.path.join(COCO_IMAGE_DIR, image_data)
            # 有些 dataset 路徑可能包含 'coco/train2017', 有些只有檔名，需自行調整
            if not os.path.exists(full_path):
                # 嘗試另一種常見結構
                if "train2017" in image_data:
                     filename = os.path.basename(image_data)
                     full_path = os.path.join(COCO_IMAGE_DIR, filename)
            
            if os.path.exists(full_path):
                try:
                    image = Image.open(full_path).convert("RGB")
                except:
                    print(f"[Error] Corrupt image: {full_path}")
                    continue
            else:
                # 找不到圖片，跳過 (這很正常，如果你沒有下載完整的 COCO)
                # print(f"[Warning] Image not found: {image_data}")
                continue
        else:
            continue

        if image is None:
            continue

        # 3. 執行 Florence Inference
        with torch.no_grad():
            # 我們使用 <OD> (Object Detection) 來抓取圖中所有顯著物體
            # 這樣 LLaVA 在描述整張圖時，能獲得所有物體的幾何提示
            try:
                result_dict = model.run_florence_inference(image)
                
                # 簡單驗證
                if not result_dict['labels']:
                    pass
                
                # 存檔 (存 Raw Data: Labels + Polygons)
                torch.save(result_dict, save_path)
                processed_ids.add(img_id)
                
            except Exception as e:
                print(f"Inference failed for {img_id}: {e}")

            del result_dict
        
        # 定期清理記憶體
        if i % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"Preprocessing Completed! Total processed images: {len(processed_ids)}")

if __name__ == "__main__":
    preprocess_dataset()