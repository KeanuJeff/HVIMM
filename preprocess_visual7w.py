import gc
import torch
import os
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# 載入模型 Class
from models.structural_llava_next import HybirdLlavaFlorenceModel

def preprocess_visual7w():
    output_dir = "./processed_features_visual7w" # 輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model (Florence ONLY) on {device}...")
    
    # 只載入 Florence-2
    model = HybirdLlavaFlorenceModel(
        load_llava=False, 
        load_florence=True,
        florence_model_id="./models/local_florence2" # 或 "microsoft/Florence-2-large"
    )
    
    # 設定 Visual7W
    dataset_name = "HuggingFaceM4/the_cauldron"
    subset_name = "visual7w"
    splits = ['train'] # Visual7W 在此 Dataset 中通常只有 train (或依照你實際使用的 split)
    
    processed_count = 0
    
    print(f"Start Preprocessing {subset_name}...")
    
    for split in splits:
        print(f"-> Processing Split: {split}")
        try:
            ds = load_dataset(dataset_name, subset_name, split=split, streaming=False)
        except Exception as e:
            print(f"Error loading {split}: {e}")
            continue

        # 遍歷 Dataset
        for idx, item in tqdm(enumerate(ds), total=len(ds)):
            # 【關鍵】生成唯一 ID (因為 Visual7W 沒有 global image_id)
            # 格式: visual7w_train_0, visual7w_train_1 ...
            file_id = f"visual7w_{split}_{idx}"
            
            save_path = os.path.join(output_dir, f"{file_id}.pt")
            
            # 如果已經跑過，就跳過
            if os.path.exists(save_path):
                continue

            # 取得圖片
            image_list = item.get('images', [])
            if len(image_list) == 0:
                continue # 空圖片跳過
            
            image_obj = image_list[0] # 取第一張圖
            
            if not isinstance(image_obj, Image.Image):
                continue
            
            image = image_obj.convert("RGB")
            
            # 執行 Florence Inference
            with torch.no_grad():
                # 回傳 {'labels': [...], 'polygons': [[...], [...]]}
                result_dict = model.run_florence_inference(image)
                
                # 存檔
                torch.save(result_dict, save_path)
                processed_count += 1
                
            # 清理記憶體
            del result_dict, image
            
            # 定期清理 Cache (每 100 張)
            if idx % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    print(f"Preprocessing Completed! Total processed images: {processed_count}")

if __name__ == "__main__":
    preprocess_visual7w()