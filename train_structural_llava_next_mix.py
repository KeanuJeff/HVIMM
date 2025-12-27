import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, ConcatDataset
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import os
import sys
from PIL import Image
from huggingface_hub import hf_hub_download
from peft import LoraConfig

# 假設你的 models 資料夾在當前目錄
from models.structural_llava_next_raw import HybirdLlavaFlorenceModel, GeometricUtils

# ==============================================================================
# 全域設定 (請根據你的實際路徑修改)
# ==============================================================================

# 1. 圖片路徑
COCO_IMAGE_DIR = "./dataset/train2017" 

# 2. 特徵路徑 (Feature Paths)
FEATURE_DIR_LLAVA = "./processed_features_sharegpt4v"
FEATURE_DIR_REFCOCOG = "./processed_features_raw"
FEATURE_DIR_TEXTCAPS = "./processed_features_textcaps"

# 3. 預訓練權重路徑
PRETRAINED_ADAPTER_PATH = "./final_adapter_llava_instruct1/custom_modules.bin"

# ==============================================================================
# 工具函式
# ==============================================================================

def plot_loss_curve(log_history, output_dir):
    train_steps = []
    train_loss = []
    for entry in log_history:
        if 'loss' in entry and 'step' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])
    plt.figure(figsize=(10, 6))
    if train_loss:
        plt.plot(train_steps, train_loss, label='Training Loss', color='blue', alpha=0.7)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve (Mixed 3:1:1)')
    plt.legend()
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

class StructuralDataCollator:
    def __call__(self, features):
        batch = {}
        batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
        batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
        batch['labels'] = torch.stack([f['labels'] for f in features])
        batch['image_sizes'] = torch.stack([f['image_sizes'] for f in features])
        
        # Pixel Values Padding
        pixel_values_list = [f['pixel_values'] for f in features]
        max_patches = max([x.shape[0] for x in pixel_values_list])
        padded_pixel_values = []
        for pv in pixel_values_list:
            n_patches = pv.shape[0]
            if n_patches < max_patches:
                pad = torch.zeros((max_patches - n_patches, *pv.shape[1:]), dtype=pv.dtype)
                padded_pv = torch.cat([pv, pad], dim=0)
                padded_pixel_values.append(padded_pv)
            else:
                padded_pixel_values.append(pv)
        batch['pixel_values'] = torch.stack(padded_pixel_values)

        # Structural Feature Padding
        obj_counts = [f['structural_fourier'].shape[0] for f in features]
        if not obj_counts: max_objs = 1
        else: max_objs = max(max(obj_counts), 1)
        
        padded_fourier = []
        padded_centroids = []
        batch_labels_list = [] 
        
        for f in features:
            f_feat = f['structural_fourier']
            c_feat = f['structural_centroids']
            curr_objs = f_feat.shape[0]
            
            if curr_objs < max_objs:
                pad = torch.zeros(max_objs - curr_objs, 48, dtype=f_feat.dtype)
                f_feat = torch.cat([f_feat, pad], dim=0)
                pad_c = torch.zeros(max_objs - curr_objs, 2, dtype=c_feat.dtype)
                c_feat = torch.cat([c_feat, pad_c], dim=0)
            elif curr_objs > max_objs:
                f_feat = f_feat[:max_objs]
                c_feat = c_feat[:max_objs]
                
            padded_fourier.append(f_feat)
            padded_centroids.append(c_feat)
            batch_labels_list.append(f['structural_labels'][:max_objs])
            
        batch['structural_fourier'] = torch.stack(padded_fourier)
        batch['structural_centroids'] = torch.stack(padded_centroids)
        batch['structural_labels'] = batch_labels_list
        return batch

# ==============================================================================
# Dataset 1: LLaVA Instruct 150K (Target: Last 30,000)
# ==============================================================================
class LLaVAInstructDataset(Dataset):
    def __init__(self, hf_dataset, feature_dir, model_processor, tokenizer, target_count=30000):
        self.feature_dir = feature_dir
        self.processor = model_processor
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'right'
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'right'
        self.max_length = 3460
        self.geo_utils = GeometricUtils()

        print(f"[LLaVA] Filtering data... (Target: last {target_count})")
        self.original_data = list(hf_dataset)
        valid_candidates = []
        
        # 1. 先篩選出所有合法的資料
        for item in self.original_data:
            image_filename = item.get('image', '')
            raw_id = str(item.get('id', ''))
            
            # 策略: raw_id 或 去零後的 short_id
            feat_path_candidate_1 = os.path.join(self.feature_dir, f"raw_{raw_id}.pt")
            short_id = raw_id.lstrip('0')
            if short_id == '': short_id = '0'
            feat_path_candidate_2 = os.path.join(self.feature_dir, f"raw_{short_id}.pt")
            
            final_feat_path = None
            if os.path.exists(feat_path_candidate_1):
                final_feat_path = feat_path_candidate_1
            elif os.path.exists(feat_path_candidate_2):
                final_feat_path = feat_path_candidate_2
            
            # 檢查圖片
            image_path_full = os.path.join(COCO_IMAGE_DIR, image_filename)
            if not os.path.exists(image_path_full):
                 image_path_full = os.path.join(COCO_IMAGE_DIR, "train2017", image_filename)

            if final_feat_path and os.path.exists(image_path_full):
                valid_candidates.append({
                    "data": item,
                    "feat_path": final_feat_path,
                    "image_path": image_path_full
                })

        print(f"[LLaVA] Found {len(valid_candidates)} valid samples.")
        
        # 2. 倒排序取後面 N 筆 (取不到 N 筆則全取)
        if len(valid_candidates) > target_count:
            self.valid_data_list = valid_candidates[-target_count:]
        else:
            self.valid_data_list = valid_candidates
            
        print(f"[LLaVA] Final used samples: {len(self.valid_data_list)} (Slicing from back)")

    def __len__(self):
        return len(self.valid_data_list)

    def __getitem__(self, idx):
        record = self.valid_data_list[idx]
        item = record["data"]
        feat_path = record["feat_path"]
        image_path = record["image_path"]
        
        # 特徵讀取 (省略錯誤處理以節省篇幅，邏輯同你原本代碼)
        shape_feat_tensor = torch.zeros(0, 48)
        centroid_tensor = torch.zeros(0, 2)
        obj_labels = []
        try:
            raw_data = torch.load(feat_path)
            raw_labels = raw_data.get('labels', [])
            raw_polys = raw_data.get('polygons', [])
            fourier_list, centroid_list, valid_labels = [], [], []
            for i, poly_list in enumerate(raw_polys):
                if len(poly_list) < 6: continue
                target_poly = torch.tensor(poly_list, dtype=torch.float32).reshape(-1, 2).unsqueeze(0)
                shape_feat = self.geo_utils.fourier_shape_encoding(target_poly, num_harmonics=24)
                centroid = self.geo_utils.calculate_centroid(target_poly)
                if torch.isnan(shape_feat).any() or torch.isinf(shape_feat).any() or torch.all(shape_feat==0): continue
                fourier_list.append(shape_feat.squeeze(0))
                centroid_list.append(centroid.squeeze(0))
                valid_labels.append(raw_labels[i])
            if fourier_list:
                shape_feat_tensor = torch.stack(fourier_list)
                centroid_tensor = torch.stack(centroid_list)
                obj_labels = valid_labels
        except: pass

        # 圖片與 Prompt 處理
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = Image.new('RGB', (336, 336), (0, 0, 0))
            
        # Resize logic...
        target_max_size = 672
        w, h = image.size
        scale = min(target_max_size / w, target_max_size / h)
        if scale < 1.0:
            image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

        conversations = item.get('conversations', [])
        human_input, gpt_response = "", ""
        for turn in conversations:
            if turn['from'] == 'human': human_input = turn['value']
            elif turn['from'] == 'gpt': gpt_response = turn['value']
        
        human_input = human_input.replace("<image>", "").strip()
        prompt = f"[INST] <image>\n{human_input} [/INST]"
        full_text = prompt + " " + gpt_response

        # Tokenization
        with torch.no_grad():
            prompt_inputs = self.processor(images=image, text=prompt, return_tensors="pt", padding="do_not_pad")
        prompt_len = prompt_inputs.input_ids.shape[1]

        inputs = self.processor(images=image, text=full_text, return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True)
        input_ids = inputs.input_ids[0]
        labels = input_ids.clone()
        if prompt_len < len(labels): labels[:prompt_len] = -100
        else: labels[:] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids, "attention_mask": inputs.attention_mask[0],
            "pixel_values": inputs.pixel_values[0], "image_sizes": inputs.image_sizes[0],
            "labels": labels, "structural_fourier": shape_feat_tensor,
            "structural_centroids": centroid_tensor, "structural_labels": obj_labels
        }

# ==============================================================================
# Dataset 2: RefCOCOg (Target: Last 10,000)
# ==============================================================================
class RefCOCOgDataset(Dataset):
    def __init__(self, hf_dataset, feature_dir, model_processor, tokenizer, target_count=10000):
        self.feature_dir = feature_dir
        self.processor = model_processor
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'right'
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'right'
        self.max_length = 3460
        self.geo_utils = GeometricUtils()
        
        print(f"[RefCOCOg] Filtering data... (Target: last {target_count})")
        
        # 必須先遍歷一次以確認 Feature 存在 (為了滿足你的倒排序要求)
        self.valid_data_list = []
        original_data = list(hf_dataset)
        
        for item in original_data:
            if 'question_id' in item: image_id = str(item['question_id'])
            elif 'id' in item: image_id = str(item['id'])
            else: image_id = str(item.get('image_id', ''))
            
            feat_path = os.path.join(self.feature_dir, f"raw_{image_id}.pt")
            
            # 只有當 feature 存在時才保留
            if os.path.exists(feat_path):
                self.valid_data_list.append({
                    "data": item,
                    "feat_path": feat_path,
                    "image_id": image_id
                })
                
        print(f"[RefCOCOg] Found {len(self.valid_data_list)} valid samples.")
        
        # 倒排序取後 N 筆
        if len(self.valid_data_list) > target_count:
            self.valid_data_list = self.valid_data_list[-target_count:]
            
        print(f"[RefCOCOg] Final used samples: {len(self.valid_data_list)}")

    def __len__(self):
        return len(self.valid_data_list)

    def __getitem__(self, idx):
        record = self.valid_data_list[idx]
        item = record["data"]
        feat_path = record["feat_path"]
        
        # 讀取圖片
        image_obj = item.get('image', item.get('jpg'))
        if image_obj is None: image = Image.new('RGB', (336, 336), (0, 0, 0))
        else: image = image_obj.convert("RGB")
        
        w, h = image.size
        w_orig, h_orig = w, h
        target_max_size = 672
        scale = min(target_max_size / w, target_max_size / h)
        if scale < 1.0:
            image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            
        # 讀取 Feature
        shape_feat_tensor = torch.zeros(0, 48)
        centroid_tensor = torch.zeros(0, 2)
        obj_labels = []
        try:
            raw_data = torch.load(feat_path)
            raw_labels = raw_data.get('labels', [])
            raw_polys = raw_data.get('polygons', [])
            fourier_list, centroid_list, valid_labels = [], [], []
            for i, poly_list in enumerate(raw_polys):
                if len(poly_list) < 6: continue
                target_poly = torch.tensor(poly_list, dtype=torch.float32).reshape(-1, 2).unsqueeze(0)
                shape_feat = self.geo_utils.fourier_shape_encoding(target_poly, num_harmonics=24)
                centroid = self.geo_utils.calculate_centroid(target_poly)
                if torch.isnan(shape_feat).any() or torch.isinf(shape_feat).any() or torch.all(shape_feat==0): continue
                fourier_list.append(shape_feat.squeeze(0))
                centroid_list.append(centroid.squeeze(0))
                valid_labels.append(raw_labels[i])
            if fourier_list:
                shape_feat_tensor = torch.stack(fourier_list)
                centroid_tensor = torch.stack(centroid_list)
                obj_labels = valid_labels
        except: pass

        # Prompt
        bbox = item.get('bbox', [0, 0, 100, 100])
        caption = "object"
        if 'answer' in item and isinstance(item['answer'], list) and len(item['answer']) > 0:
            caption = item['answer'][-1]
            
        box_norm = [
            int(bbox[0]/w_orig * 100), int(bbox[1]/h_orig * 100), 
            int((bbox[0]+bbox[2])/w_orig * 100), int((bbox[1]+bbox[3])/h_orig * 100)
        ]
        box_norm = [max(0, min(100, x)) for x in box_norm]
        box_str = f"[x0={box_norm[0]}, y0={box_norm[1]}, x1={box_norm[2]}, y1={box_norm[3]}]"
        
        question = f"[INST] <image>\nDescribe the object at {box_str}, where (0, 0) is at the upper left and (100, 100) is at the lower right. [/INST]"
        full_text = question + " " + caption

        with torch.no_grad():
            prompt_inputs = self.processor(images=image, text=question, return_tensors="pt", padding="do_not_pad")
        prompt_len = prompt_inputs.input_ids.shape[1]

        inputs = self.processor(images=image, text=full_text, return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True)
        input_ids = inputs.input_ids[0]
        labels = input_ids.clone()
        if prompt_len < len(labels): labels[:prompt_len] = -100
        else: labels[:] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids, "attention_mask": inputs.attention_mask[0],
            "pixel_values": inputs.pixel_values[0], "image_sizes": inputs.image_sizes[0],
            "labels": labels, "structural_fourier": shape_feat_tensor,
            "structural_centroids": centroid_tensor, "structural_labels": obj_labels
        }

# ==============================================================================
# Dataset 3: TextCaps (Target: Last 10,000)
# ==============================================================================
class TextCapsDataset(Dataset):
    def __init__(self, hf_dataset, feature_dir, model_processor, tokenizer, target_count=10000):
        self.hf_dataset = hf_dataset
        self.feature_dir = feature_dir
        self.processor = model_processor
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'right'
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'right'
        self.max_length = 3460
        self.geo_utils = GeometricUtils()

        print(f"[TextCaps] Filtering data... (Target: last {target_count})")
        valid_indices = []
        for idx in range(len(self.hf_dataset)):
            file_id = f"textcaps_train_{idx}"
            feat_path = os.path.join(self.feature_dir, f"{file_id}.pt")
            if os.path.exists(feat_path):
                valid_indices.append(idx)
        
        print(f"[TextCaps] Found {len(valid_indices)} valid samples.")
        
        # 倒排序取後 N 筆
        if len(valid_indices) > target_count:
            self.valid_indices = valid_indices[-target_count:]
        else:
            self.valid_indices = valid_indices
            
        print(f"[TextCaps] Final used samples: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        item = self.hf_dataset[real_idx]
        
        image_list = item.get('images', [])
        if len(image_list) > 0: image = image_list[0].convert("RGB")
        else: image = Image.new('RGB', (336, 336), (0, 0, 0))
        
        target_max_size = 672 
        w, h = image.size
        scale = min(target_max_size / w, target_max_size / h)
        if scale < 1.0: image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

        file_id = f"textcaps_train_{real_idx}"
        feat_path = os.path.join(self.feature_dir, f"{file_id}.pt")
        
        shape_feat_tensor = torch.zeros(0, 48)
        centroid_tensor = torch.zeros(0, 2)
        obj_labels = []
        if os.path.exists(feat_path):
            try:
                raw_data = torch.load(feat_path) 
                raw_labels = raw_data.get('labels', [])
                raw_polys = raw_data.get('polygons', [])
                fourier_list, centroid_list, valid_labels = [], [], []
                for i, poly_list in enumerate(raw_polys):
                    if len(poly_list) < 6: continue
                    target_poly = torch.tensor(poly_list, dtype=torch.float32).reshape(-1, 2).unsqueeze(0) 
                    shape_feat = self.geo_utils.fourier_shape_encoding(target_poly, num_harmonics=24)
                    centroid = self.geo_utils.calculate_centroid(target_poly)
                    if torch.isnan(shape_feat).any() or torch.isinf(shape_feat).any() or torch.all(shape_feat==0): continue
                    fourier_list.append(shape_feat.squeeze(0))
                    centroid_list.append(centroid.squeeze(0))
                    valid_labels.append(raw_labels[i])
                if fourier_list:
                    shape_feat_tensor = torch.stack(fourier_list)
                    centroid_tensor = torch.stack(centroid_list)
                    obj_labels = valid_labels
            except: pass

        texts_list = item.get('texts', [])
        user_input, assistant_response = "", ""
        if len(texts_list) > 0:
            conv = texts_list[0]
            user_input = conv.get('user', "Describe this image.")
            assistant_response = conv.get('assistant', "")

        prompt = f"[INST] <image>\n{user_input} [/INST]"
        full_text = prompt + " " + assistant_response

        with torch.no_grad():
            prompt_inputs = self.processor(images=image, text=prompt, return_tensors="pt", padding="do_not_pad")
        prompt_len = prompt_inputs.input_ids.shape[1]

        inputs = self.processor(images=image, text=full_text, return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True)
        input_ids = inputs.input_ids[0]
        labels = input_ids.clone()
        if prompt_len < len(labels): labels[:prompt_len] = -100
        else: labels[:] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids, "attention_mask": inputs.attention_mask[0],
            "pixel_values": inputs.pixel_values[0], "image_sizes": inputs.image_sizes[0],
            "labels": labels, "structural_fourier": shape_feat_tensor,
            "structural_centroids": centroid_tensor, "structural_labels": obj_labels
        }

# ==============================================================================
# Main Training Function
# ==============================================================================
def train():
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    print("Loading Model (LLaVA ONLY)...")
    model = HybirdLlavaFlorenceModel(
        llava_model_id=model_id,
        load_llava=True,
        load_florence=False 
    )

    # LoRA Config
    peft_config = LoraConfig(
        r=128, lora_alpha=256, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", 
        target_modules=r".*multi_modal_projector.*linear.*" 
    )

    print("=== Loading Pretrained Weights (From LLaVA Instruct 1) ===")
    if os.path.exists(PRETRAINED_ADAPTER_PATH):
        print(f"Loading Adapter weights from {PRETRAINED_ADAPTER_PATH}...")
        state_dict = torch.load(PRETRAINED_ADAPTER_PATH, map_location=model.llava.device)
        model.adapter.load_state_dict(state_dict["adapter"])
        model.shape_projector.load_state_dict(state_dict["shape_projector"])
        model.label_down_projector.load_state_dict(state_dict["label_down_projector"])
    else:
        print("!!! Warning: Pretrained weights not found! !!!")

    # 設定 Gradients
    for name, param in model.named_parameters():
        if "adapter" in name or "shape_projector" in name or "label_down_projector" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")

    # =================================================
    # 載入資料集
    # =================================================
    
    # 1. LLaVA Instruct 150K
    print("--- Preparing LLaVA Instruct Dataset ---")
    try:
        repo_id = "liuhaotian/LLaVA-Instruct-150K"
        filename = "llava_instruct_150k.json"
        json_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        ds_llava_raw = load_dataset("json", data_files=json_path, split="train")
        
        train_ds_llava = LLaVAInstructDataset(
            hf_dataset=ds_llava_raw,
            feature_dir=FEATURE_DIR_LLAVA,
            model_processor=model.llava_processor,
            tokenizer=model.llava_processor.tokenizer,
            target_count=30000  # 取最後 3 萬筆
        )
    except Exception as e:
        print(f"Error loading LLaVA: {e}")
        return

    # 2. RefCOCOg
    print("--- Preparing RefCOCOg Dataset ---")
    try:
        # 為了確保足夠數量，這裡同時載入 test 和 val，如果不夠會在內部處理
        ds_ref_raw = load_dataset("lmms-lab/RefCOCOg", split="test+val", streaming=False)
        train_ds_ref = RefCOCOgDataset(
            hf_dataset=ds_ref_raw,
            feature_dir=FEATURE_DIR_REFCOCOG,
            model_processor=model.llava_processor,
            tokenizer=model.llava_processor.tokenizer,
            target_count=10000 # 取最後 1 萬筆
        )
    except Exception as e:
        print(f"Error loading RefCOCOg: {e}")
        return

    # 3. TextCaps
    print("--- Preparing TextCaps Dataset ---")
    try:
        ds_text_raw = load_dataset("HuggingFaceM4/the_cauldron", "textcaps", split="train", streaming=False)
        train_ds_text = TextCapsDataset(
            hf_dataset=ds_text_raw,
            feature_dir=FEATURE_DIR_TEXTCAPS,
            model_processor=model.llava_processor,
            tokenizer=model.llava_processor.tokenizer,
            target_count=10000 # 取最後 1 萬筆
        )
    except Exception as e:
        print(f"Error loading TextCaps: {e}")
        return

    # =================================================
    # 混合訓練
    # =================================================
    print(f"Merging Datasets: LLaVA({len(train_ds_llava)}) + RefCOCOg({len(train_ds_ref)}) + TextCaps({len(train_ds_text)})")
    
    # 使用 ConcatDataset 合併
    # 由於數量分別為 30k, 10k, 10k，Trainer Shuffle=True 時，自然會形成 3:1:1 的訓練分佈
    combined_train_dataset = ConcatDataset([train_ds_llava, train_ds_ref, train_ds_text])
    print(f"Total Combined Samples: {len(combined_train_dataset)}")

    training_args = TrainingArguments(
        output_dir="./results_mixed_training",
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=5, 
        num_train_epochs=1,
        learning_rate=1e-4, # 設定 LR
        fp16=True,              
        bf16=False,
        optim="adamw_torch",
        max_grad_norm=0.5,  # 設定 Max Grad
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=30,
        save_steps=200,     # 因為資料量變大，Save 間隔可以拉長
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=4
    )

    if hasattr(model.llava, "gradient_checkpointing_enable"):
        model.llava.gradient_checkpointing_enable() 
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_train_dataset,
        data_collator=StructuralDataCollator(),
    )

    print("Starting Mixed Training...")
    trainer.train()
    plot_loss_curve(trainer.state.log_history, "./results_mixed_training")

    print("Saving weights...")
    save_dir = "./final_adapter_mixed"
    os.makedirs(save_dir, exist_ok=True)

    custom_weights = {
        "adapter": model.adapter.state_dict(),
        "shape_projector": model.shape_projector.state_dict(),
        "label_down_projector": model.label_down_projector.state_dict(),
    }
    torch.save(custom_weights, os.path.join(save_dir, "custom_modules.bin"))
    print("Done!")

if __name__ == "__main__":
    train()