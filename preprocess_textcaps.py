import gc
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from models.structural_llava_next_raw import HybirdLlavaFlorenceModel

BATCH_SIZE = 32 
NUM_WORKERS = 8  
OUTPUT_DIR = "./processed_features_textcaps"

# ==========================================
# Dataset Wrapper for Batch Loading
# ==========================================
class BatchTextCapsDataset(Dataset):
    def __init__(self, hf_dataset, output_dir):
        self.dataset = hf_dataset
        self.output_dir = output_dir
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return idx, self.dataset[idx]

def custom_collate_fn(batch):
    valid_batch = []
    
    for idx, item in batch:
        file_id = f"textcaps_train_{idx}"
        save_path = os.path.join(OUTPUT_DIR, f"{file_id}.pt")
        
        if os.path.exists(save_path):
            continue
            
        image_list = item.get('images', [])
        if len(image_list) == 0:
            continue
            
        image = image_list[0].convert("RGB")
        valid_batch.append((idx, image))
        
    return valid_batch

def run_batch_inference_optimized(model, batch_data):
    if not batch_data:
        return

    images = [x[1] for x in batch_data]
    indices = [x[0] for x in batch_data]
    
    device = model.florence.device
    dtype = model.florence.dtype
    processor = model.florence_processor

    # Temporary dictionary: {global_idx: {'labels': [], 'polygons': []}}
    results_buffer = {idx: {'labels': [], 'polygons': []} for idx in indices}

    # ==========================================
    # Step 1: Batch Object Detection (<OD>)
    # ==========================================
    task_od = '<OD>'
    prompts = [task_od] * len(images)
    
    try:
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device, dtype)
        
        generated_ids = model.florence.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            num_beams=3
        )
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
        
        del inputs, generated_ids
    except Exception as e:
        print(f"Batch OD generation failed: {e}")
        return

    seg_tasks = []
    task_seg = '<REFERRING_EXPRESSION_SEGMENTATION>'

    for i, (idx, image) in enumerate(zip(indices, images)):
        text = generated_texts[i]
        try:
            res_od = processor.post_process_generation(
                text, task=task_od, image_size=(image.width, image.height)
            )
            od_data = res_od.get('<OD>', {})
            raw_labels = od_data.get('labels', [])
            unique_labels = list(set(raw_labels))
            
            for label in unique_labels:
                seg_tasks.append({
                    "image": image,
                    "prompt": task_seg + label,
                    "global_idx": idx,
                    "label": label
                })
        except Exception as e:
            print(f"Error parsing OD for index {idx}: {e}")
            continue

    # ==========================================
    # Step 3: Batch Segmentation (Flattened)
    # ==========================================
    if seg_tasks:
        SEG_BATCH_SIZE = 16 
        
        for i in range(0, len(seg_tasks), SEG_BATCH_SIZE):
            batch_tasks = seg_tasks[i : i + SEG_BATCH_SIZE]
            
            batch_images = [t["image"] for t in batch_tasks]
            batch_prompts = [t["prompt"] for t in batch_tasks]
            
            try:
                inputs_seg = processor(
                    text=batch_prompts, images=batch_images, return_tensors="pt", padding=True
                ).to(device, dtype)
                
                ids_seg = model.florence.generate(
                    input_ids=inputs_seg["input_ids"], 
                    pixel_values=inputs_seg["pixel_values"], 
                    max_new_tokens=1024, 
                    do_sample=False
                )
                
                texts_seg = processor.batch_decode(ids_seg, skip_special_tokens=False)
                
                for j, t in enumerate(batch_tasks):
                    g_idx = t["global_idx"]
                    label = t["label"]
                    img_w, img_h = t["image"].width, t["image"].height
                    
                    res_seg = processor.post_process_generation(
                        texts_seg[j], task=task_seg, image_size=(img_w, img_h)
                    )
                    seg_data = res_seg.get(task_seg, {})
                    
                    if 'polygons' in seg_data:
                        poly_raw = seg_data['polygons']
                        for p in poly_raw:
                            if len(p) > 0 and isinstance(p[0], (list, np.ndarray)):
                                for sub_p in p:
                                    if len(sub_p) >= 6:
                                        results_buffer[g_idx]['labels'].append(label)
                                        results_buffer[g_idx]['polygons'].append(sub_p)
                            else:
                                if len(p) >= 6:
                                    results_buffer[g_idx]['labels'].append(label)
                                    results_buffer[g_idx]['polygons'].append(p)
                                    
                del inputs_seg, ids_seg, texts_seg
                
            except Exception as e:
                print(f"Batch Seg inference failed at step {i}: {e}")
                continue

    # ==========================================
    # Step 4: Archive
    # ==========================================
    for idx in indices:
        save_path = os.path.join(OUTPUT_DIR, f"textcaps_train_{idx}.pt")
        try:
            torch.save(results_buffer[idx], save_path)
        except Exception as e:
            print(f"Error saving index {idx}: {e}")

# ==========================================
# Main
# ==========================================
def preprocess_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model on {device}...")
    
    model = HybirdLlavaFlorenceModel(
        load_llava=False, 
        load_florence=True,
        florence_model_id="./models/local_florence2" 
    )
    
    print(f"Loading Dataset...")
    ds = load_dataset("HuggingFaceM4/the_cauldron", "textcaps", split="train", streaming=False)
    print(f"Total samples: {len(ds)}")

    batch_ds = BatchTextCapsDataset(ds, OUTPUT_DIR)
    
    dataloader = DataLoader(
        batch_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn 
    )

    print("Start Batch Preprocessing...")
    
    # 3. Batch Loop
    print("Start Batch Preprocessing...")
    
    with torch.inference_mode():
        for batch_data in tqdm(dataloader):
            if not batch_data: 
                continue
            
            run_batch_inference_optimized(model, batch_data)

    print(f"Preprocessing Completed! Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    preprocess_dataset()
