import gc
import torch
import os
import sys
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

from models.structural_llava_next import HybirdLlavaFlorenceModel

COCO_IMAGE_DIR = "./dataset/train2017" 
OUTPUT_DIR = "./processed_features_sharegpt4v1"

def preprocess_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model (Florence ONLY) on {device}...")
    
    model = HybirdLlavaFlorenceModel(
        load_llava=False, 
        load_florence=True,
        florence_model_id="./models/local_florence2"
    )
    
    print(f"Loading ShareGPT4V Dataset...")
    try:
        ds = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    processed_ids = set()
    print(f"Start Preprocessing ShareGPT4V...")
    
    for i, item in tqdm(enumerate(ds)):
        if 'id' in item:
            img_id = str(item['id'])
        else:
            img_id = str(i)
            
        save_path = os.path.join(OUTPUT_DIR, f"raw_{img_id}.pt")
        
        if os.path.exists(save_path):
            processed_ids.add(img_id)
            continue
        if img_id in processed_ids: 
            continue

        image_data = item['image']
        image = None

        if isinstance(image_data, Image.Image):
            image = image_data.convert("RGB")
        elif isinstance(image_data, str):
            full_path = os.path.join(COCO_IMAGE_DIR, image_data)
            if not os.path.exists(full_path):
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
                continue
        else:
            continue

        if image is None:
            continue

        with torch.no_grad():
            try:
                result_dict = model.run_florence_inference(image)
                
                if not result_dict['labels']:
                    pass
                
                torch.save(result_dict, save_path)
                processed_ids.add(img_id)
                
            except Exception as e:
                print(f"Inference failed for {img_id}: {e}")

            del result_dict
        
        if i % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"Preprocessing Completed! Total processed images: {len(processed_ids)}")

if __name__ == "__main__":
    preprocess_dataset()
