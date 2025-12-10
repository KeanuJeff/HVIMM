import gc
import torch
import os
import sys
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# 載入模型 Class
from models.structural_llava_next import HybirdLlavaFlorenceModel

def preprocess_dataset():
    output_dir = "./processed_features_raw" # 建議改名區分
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model (Florence ONLY) on {device}...")
    
    # 【關鍵修改】只載入 Florence，不載入 LLaVA
    model = HybirdLlavaFlorenceModel(
        load_llava=False, 
        load_florence=True,
        florence_model_id="./models/local_florence2" # 或 "microsoft/Florence-2-large"
    )
    
    dataset_name = "lmms-lab/RefCOCOg" 
    splits = ['val', 'test'] 
    
    processed_ids = set()
    
    print(f"Start Preprocessing {dataset_name}...")
    
    for split in splits:
        print(f"-> Processing Split: {split}")
        ds = load_dataset(dataset_name, split=split, streaming=True)
        
        # 檢查 Keys
        try:
            first_item = next(iter(ds))
            print(f"   [Check] Sample Keys: {list(first_item.keys())}")
            if 'question_id' not in first_item:
                raise KeyError(f"Dataset split '{split}' does not have 'question_id'.")
            if 'image' not in first_item:
                raise KeyError(f"Dataset split '{split}' does not have 'image' column.")
            print("   [Check] Keys confirmed. Starting loop...")
        except StopIteration:
            print(f"   [Warning] Split '{split}' is empty!")
            continue

        for i, item in tqdm(enumerate(ds)):
            img_id = str(item['question_id'])
            
            # 改為 .pt 檔存 Dict，而非 Tensor
            save_path = os.path.join(output_dir, f"raw_{img_id}.pt")
            
            if os.path.exists(save_path):
                processed_ids.add(img_id)
                continue
            if img_id in processed_ids: 
                continue

            image_obj = item['image']
            if not isinstance(image_obj, Image.Image):
                raise TypeError(f"ID {img_id}: 'image' is not PIL Image.")
            
            image = image_obj.convert("RGB")
            
            with torch.no_grad():
                # 【關鍵修改】只執行 Florence 偵測
                # 這會回傳 {'labels': [...], 'polygons': [[...], [...]]}
                # 不會進行 Fourier 計算，也不會過 Projector
                result_dict = model.run_florence_inference(image)
                
                # 簡單驗證
                if not result_dict['labels']:
                    # 沒抓到東西，存個空的以防萬一，或者選擇跳過
                    # 這裡選擇存空的，方便 Dataset 端處理
                    pass
                
                # 存檔 (存 Raw Data)
                torch.save(result_dict, save_path)
                processed_ids.add(img_id)
            del result_dict, image  # 刪除參照
            torch.cuda.empty_cache() # 歸還顯存
            gc.collect()

    print(f"Preprocessing Completed! Total unique images: {len(processed_ids)}")

if __name__ == "__main__":
    preprocess_dataset()