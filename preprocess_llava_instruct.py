import gc
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
from huggingface_hub import hf_hub_download

# 載入模型 Class
from models.structural_llava_next import HybirdLlavaFlorenceModel

# ==========================================
# 設定區
# ==========================================
# 請確保這裡指向包含 train2017 圖片的資料夾
# 例如: "./dataset/train2017/train2017" 或 "./dataset/coco/train2017"
COCO_IMAGE_DIR = "./dataset/train2017" 

# 輸出的 Feature 資料夾 (建議分開存放，避免混淆)
OUTPUT_DIR = "./processed_features_llava_instruct_20k"

# 設定要處理的資料筆數
NUM_SAMPLES = 20000

def preprocess_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model (Florence ONLY) on {device}...")
    
    # 只載入 Florence-2
    model = HybirdLlavaFlorenceModel(
        load_llava=False, 
        load_florence=True,
        florence_model_id="./models/local_florence2" # 或 "microsoft/Florence-2-large"
    )
    
    # 1. 下載並讀取 LLaVA-Instruct JSON
    print(f"Downloading/Loading LLaVA-Instruct-150K JSON...")
    try:
        repo_id = "liuhaotian/LLaVA-Instruct-150K"
        filename = "llava_instruct_150k.json"
        json_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        
        with open(json_path, 'r') as f:
            full_data = json.load(f)
            
        print(f"Original dataset size: {len(full_data)}")
        
        # 2. 【關鍵修改】只取前 20,000 筆
        target_data = full_data[:NUM_SAMPLES]
        print(f"Processing target subset: {len(target_data)} samples")
        
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    processed_ids = set()
    print(f"Start Preprocessing...")
    
    for i, item in tqdm(enumerate(target_data), total=len(target_data)):
        # LLaVA-Instruct 格式: {"id": "000000215677", "image": "000000215677.jpg", ...}
        
        # 1. 取得 ID 與 檔名
        img_id = str(item.get('id', i))
        image_filename = item.get('image', None)
        
        if not image_filename:
            continue

        save_path = os.path.join(OUTPUT_DIR, f"raw_{img_id}.pt")
        
        # 檢查是否已處理 (避免重複運算)
        if os.path.exists(save_path):
            processed_ids.add(img_id)
            continue
        if img_id in processed_ids: 
            continue

        # 2. 處理圖片路徑
        full_path = os.path.join(COCO_IMAGE_DIR, image_filename)
        
        # 如果直接組裝路徑找不到，嘗試在中間加一層 train2017 (視您的資料夾結構而定)
        if not os.path.exists(full_path):
             full_path = os.path.join(COCO_IMAGE_DIR, "train2017", image_filename)

        image = None
        if os.path.exists(full_path):
            try:
                image = Image.open(full_path).convert("RGB")
            except Exception as e:
                print(f"[Error] Corrupt image: {full_path}")
                continue
        else:
            # 找不到圖片 (可能是 COCO 資料集不完整)，跳過
            # print(f"[Warning] Image not found: {full_path}")
            continue

        # 3. 執行 Florence Inference
        with torch.no_grad():
            try:
                result_dict = model.run_florence_inference(image)
                
                # 存檔
                torch.save(result_dict, save_path)
                processed_ids.add(img_id)
                
            except Exception as e:
                print(f"Inference failed for {img_id}: {e}")

            if 'result_dict' in locals():
                del result_dict
        
        # 定期清理記憶體
        if i % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"Preprocessing Completed! Total processed features: {len(processed_ids)}")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    preprocess_dataset()