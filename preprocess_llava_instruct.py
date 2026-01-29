import gc
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
from huggingface_hub import hf_hub_download

from models.structural_llava_next import HybirdLlavaFlorenceModel

COCO_IMAGE_DIR = "./dataset/train2017" 

OUTPUT_DIR = "./processed_features_llava_instruct_20k"

NUM_SAMPLES = 20000

def preprocess_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model (Florence ONLY) on {device}...")
    
    model = HybirdLlavaFlorenceModel(
        load_llava=False, 
        load_florence=True,
        florence_model_id="./models/local_florence2" # 或 "microsoft/Florence-2-large"
    )
    
    print(f"Downloading/Loading LLaVA-Instruct-150K JSON...")
    try:
        repo_id = "liuhaotian/LLaVA-Instruct-150K"
        filename = "llava_instruct_150k.json"
        json_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        
        with open(json_path, 'r') as f:
            full_data = json.load(f)
            
        print(f"Original dataset size: {len(full_data)}")
        
        target_data = full_data[:NUM_SAMPLES]
        print(f"Processing target subset: {len(target_data)} samples")
        
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    processed_ids = set()
    print(f"Start Preprocessing...")
    
    for i, item in tqdm(enumerate(target_data), total=len(target_data)):
        img_id = str(item.get('id', i))
        image_filename = item.get('image', None)
        
        if not image_filename:
            continue

        save_path = os.path.join(OUTPUT_DIR, f"raw_{img_id}.pt")
        
        if os.path.exists(save_path):
            processed_ids.add(img_id)
            continue
        if img_id in processed_ids: 
            continue

        full_path = os.path.join(COCO_IMAGE_DIR, image_filename)
        
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
            # print(f"[Warning] Image not found: {full_path}")
            continue

        with torch.no_grad():
            try:
                result_dict = model.run_florence_inference(image)
                
                torch.save(result_dict, save_path)
                processed_ids.add(img_id)
                
            except Exception as e:
                print(f"Inference failed for {img_id}: {e}")

            if 'result_dict' in locals():
                del result_dict
        
        if i % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"Preprocessing Completed! Total processed features: {len(processed_ids)}")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    preprocess_dataset()
