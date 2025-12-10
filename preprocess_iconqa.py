import torch
import os
import sys
from tqdm import tqdm
from PIL import Image

# 設定路徑
#sys.path.append('./models')
#sys.path.append('./dataset')

# 載入模型 Class
from models.structural_llava_next import HybirdLlavaFlorenceModel
# 載入您的 Dataset
from dataset.iconqa import IconQADataset 

def preprocess_iconqa():
    output_dir = "./processed_features_iconqa_raw" # 建議區分這存放的是 raw data
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model (Florence ONLY) on {device}...")
    
    # 【關鍵設定】只載入 Florence-2，節省顯存並加速
    model = HybirdLlavaFlorenceModel(
        load_llava=False, 
        load_florence=True,
        florence_model_id="./models/local_florence2" # 或 "microsoft/Florence-2-large"
    )
    # HybirdLlavaFlorenceModel 內部會自動處理 device 移動，這裡不需要再手動 to(device)

    # 設定要處理的 Split (通常是 train, val, test)
    # 根據您的需求，這裡示範處理 'train' 和 'val' (或 test)
    splits = ['test'] 
    
    for split in splits:
        print(f"Processing IconQA Split: {split}...")
        try:
            # 假設您的 IconQADataset 支援 split 參數
            dataset = IconQADataset(split=split, num_samples=10000) 
        except Exception as e:
            print(f"Skipping split {split}: {e}")
            continue

        print(f"Total samples in {split}: {len(dataset)}")
        processed_ids = set()

        for i in tqdm(range(len(dataset))):
            try:
                item = dataset[i]
                q_id = str(item['question_id'])
                
                save_path = os.path.join(output_dir, f"raw_{q_id}.pt")
                
                if os.path.exists(save_path):
                    continue
                if q_id in processed_ids:
                    continue

                image_obj = item['image']
                # 確保是 RGB
                if isinstance(image_obj, Image.Image):
                    image = image_obj.convert("RGB")
                else:
                    # 如果不是 PIL Image (例如路徑)，需自行載入
                    continue

                with torch.no_grad():
                    # 【核心修改】只執行 Florence 偵測，回傳 Raw Dict
                    # result_dict = {'labels': [...], 'polygons': [[...], ...]}
                    result_dict = model.run_florence_inference(image)
                    
                    # 存檔
                    torch.save(result_dict, save_path)
                    processed_ids.add(q_id)
                    
                # 【記憶體管理】強制釋放
                del result_dict, image
                # 視情況加入 empty_cache，如果顯存夠大可不加，若只有 8GB 建議每 N 張加一次
                # torch.cuda.empty_cache() 

            except Exception as e:
                print(f"Error processing index {i} (ID: {q_id}): {e}")
                continue

    print("IconQA Preprocessing Completed!")

if __name__ == "__main__":
    preprocess_iconqa()