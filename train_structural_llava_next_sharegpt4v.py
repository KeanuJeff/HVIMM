import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, ConcatDataset
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import os
from transformers import AutoTokenizer
import sys
import re
from PIL import Image
# 【新增】需要 PeftModel 來載入舊權重
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# 假設你的 models 資料夾在當前目錄
from models.structural_llava_next_raw import HybirdLlavaFlorenceModel, GeometricUtils

# 設定圖片路徑
COCO_IMAGE_DIR = "./dataset/train2017/train2017" 

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
    plt.title('Training Loss Curve ShareGPT4V')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

class ShareGPT4VDataset(Dataset):
    def __init__(self, hf_dataset, feature_dir, model_processor, tokenizer):
        # --- 【新增】預先篩選有效索引 ---
        self.original_data = list(hf_dataset)
        self.feature_dir = feature_dir
        self.valid_indices = []
        for i, item in enumerate(self.original_data):
            # 取得 Image ID
            image_id = str(item.get('id', i))
            feat_path = os.path.join(self.feature_dir, f"raw_{image_id}.pt")
            
            # 檢查特徵檔案是否存在
            if os.path.exists(feat_path):
                self.valid_indices.append(i) # 記錄原始索引
            else:
                pass
        
        self.data_map = {new_idx: original_idx for new_idx, original_idx in enumerate(self.valid_indices)}
        # --- 【新增結束】 ---
        self.processor = model_processor
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'right'
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'right'
        self.max_length = 3460
        self.geo_utils = GeometricUtils()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.data_map[idx]
        item = self.original_data[original_idx]
        
        # 1. 取得 Image ID 與 載入 Feature
        image_id = str(item.get('id', original_idx))

        # 載入 Preprocess 好的 Florence 特徵
        feat_path = os.path.join(self.feature_dir, f"raw_{image_id}.pt")
        
        shape_feat_tensor = torch.zeros(0, 48)
        centroid_tensor = torch.zeros(0, 2)
        obj_labels = []

        if os.path.exists(feat_path):
            try:
                raw_data = torch.load(feat_path)
                raw_labels = raw_data.get('labels', [])
                raw_polys = raw_data.get('polygons', [])
                
                fourier_list = []
                centroid_list = []
                valid_labels = []

                for i, poly_list in enumerate(raw_polys):
                    if len(poly_list) < 6: continue
                    target_poly = torch.tensor(poly_list, dtype=torch.float32).reshape(-1, 2).unsqueeze(0)
                    
                    shape_feat = self.geo_utils.fourier_shape_encoding(target_poly, num_harmonics=24)
                    centroid = self.geo_utils.calculate_centroid(target_poly)
                    
                    # 數值安全檢查
                    if torch.isnan(shape_feat).any() or torch.isinf(shape_feat).any() or torch.all(shape_feat==0):
                        continue
                        
                    fourier_list.append(shape_feat.squeeze(0))
                    centroid_list.append(centroid.squeeze(0))
                    valid_labels.append(raw_labels[i])
                
                if fourier_list:
                    shape_feat_tensor = torch.stack(fourier_list)
                    centroid_tensor = torch.stack(centroid_list)
                    obj_labels = valid_labels
            except Exception as e:
                print(f"Error loading feats for {image_id}: {e}")

        # 2. 載入與處理圖片
        image_data = item['image']
        image = None
        if isinstance(image_data, Image.Image):
            image = image_data.convert("RGB")
        elif isinstance(image_data, str):
            full_path = os.path.join(COCO_IMAGE_DIR, image_data)
            if not os.path.exists(full_path) and "train2017" in image_data:
                 full_path = os.path.join(COCO_IMAGE_DIR, "train2017", os.path.basename(image_data))
            
            if os.path.exists(full_path):
                image = Image.open(full_path).convert("RGB")
            else:
                image = Image.new('RGB', (336, 336), (0, 0, 0))

        target_max_size = 672
        w, h = image.size
        scale = min(target_max_size / w, target_max_size / h)
        if scale < 1.0:
            image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

        # 3. 解析 Conversations
        conversations = item['conversations']
        human_input = ""
        gpt_response = ""
        
        if len(conversations) >= 2:
            if conversations[0]['from'] == 'human':
                human_input = conversations[0]['value']
            if conversations[1]['from'] == 'gpt':
                gpt_response = conversations[1]['value']
        
        human_input = human_input.replace("<image>", "").strip()
        prompt = f"[INST] <image>\n{human_input} [/INST]"
        full_text = prompt + " " + gpt_response

        # 4. 【修正】使用 Processor 計算真實 Prompt 長度 (包含 Image Tokens)
        # 這樣才能正確 Mask 掉圖片部分和 User Prompt
        with torch.no_grad():
            prompt_inputs = self.processor(
                images=image, 
                text=prompt, 
                return_tensors="pt", 
                padding="do_not_pad",
                truncation=False
            )
        prompt_len = prompt_inputs.input_ids.shape[1]

        # 5. Tokenization (Full Text)
        inputs = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        input_ids = inputs.input_ids[0]
        image_sizes = inputs.image_sizes[0]
        
        # 6. Labels Masking
        labels = input_ids.clone()
        if prompt_len < len(labels):
            labels[:prompt_len] = -100 # Mask Image Tokens + User Prompt
        else:
            labels[:] = -100 
            
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs.attention_mask[0],
            "pixel_values": inputs.pixel_values[0],
            "image_sizes": image_sizes,
            "labels": labels,
            "structural_fourier": shape_feat_tensor,
            "structural_centroids": centroid_tensor,
            "structural_labels": obj_labels
        }

# 【修正】使用支援 Pixel Value Padding 的 Collator
class StructuralDataCollator:
    def __call__(self, features):
        batch = {}
        batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
        batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
        batch['labels'] = torch.stack([f['labels'] for f in features])
        batch['image_sizes'] = torch.stack([f['image_sizes'] for f in features])
        
        # 1. 處理 pixel_values 的動態 Patch 數量 Padding
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

        # 2. 處理幾何特徵 Padding
        max_objs = max([f['structural_fourier'].shape[0] for f in features])
        max_objs = max(max_objs, 1) 
        
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

def train():
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    feature_dir = "./processed_features_sharegpt4v"
    
    print("Loading Model (LLaVA ONLY)...")
    model = HybirdLlavaFlorenceModel(
        llava_model_id=model_id,
        load_llava=True,
        load_florence=False 
    )

    # LoRA Config
    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=r".*multi_modal_projector.*linear.*" 
    )

    """
    # 權重路徑
    adapter_path = "./final_adapter_refcocog/custom_modules.bin"
    lora_path = "./final_adapter_refcocog/llava_projector_lora"

    print("=== Loading Pretrained Weights (RefCOCOg) ===")
    
    # 1. 載入 Custom Modules (Adapter, Projector)
    if os.path.exists(adapter_path):
        print(f"Loading Adapter weights from {adapter_path}...")
        state_dict = torch.load(adapter_path, map_location=model.llava.device)
        model.adapter.load_state_dict(state_dict["adapter"])
        model.shape_projector.load_state_dict(state_dict["shape_projector"])
        model.label_down_projector.load_state_dict(state_dict["label_down_projector"])
    else:
        print("!!! Warning: RefCOCOg Adapter weights not found. Initializing from scratch. !!!")

    # 2. 載入 LoRA
    # 如果舊權重存在，使用 PeftModel.from_pretrained 載入並設定為可訓練
    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}...")
        model.llava = PeftModel.from_pretrained(model.llava, lora_path, is_trainable=True)
    else:
        print("No pretrained LoRA found. Initializing new LoRA...")
        #model.llava = get_peft_model(model.llava, peft_config)
        
    #model.llava.print_trainable_parameters()
    """

    # 設定梯度
    for name, param in model.named_parameters():
        if "adapter" in name or "shape_projector" in name or "label_down_projector" in name:
            param.requires_grad = True
        else:
            # 注意：LoRA 的 requires_grad 已經由 PeftModel 處理好了，這裡只處理自定義部分
            param.requires_grad = False
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")
    print("Preparing ShareGPT4V Dataset...")
    try:
        ds = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V", split="train", streaming=False)
        # 測試時建議先跑少量數據：
        # ds = ds.select(range(500)) 
        
        train_dataset = ShareGPT4VDataset(
            hf_dataset=ds,
            feature_dir=feature_dir,
            model_processor=model.llava_processor,
            tokenizer=model.llava_processor.tokenizer
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return


    training_args = TrainingArguments(
        output_dir="./results_sharegpt4v1",
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=4, 
        num_train_epochs=1,
        learning_rate=1e-5, # Fine-tune 可以稍微調低 LR，但 2e-4 對 Projector 也可以
        fp16=True,              
        bf16=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=30,
        save_steps=100,
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
        train_dataset=train_dataset,
        data_collator=StructuralDataCollator(),
    )

    print("Starting Training...")
    trainer.train()
    plot_loss_curve(trainer.state.log_history, "./results_sharegpt4v1")

    print("Saving weights...")
    save_dir = "./final_adapter_sharegpt4v1"
    os.makedirs(save_dir, exist_ok=True)

    custom_weights = {
        "adapter": model.adapter.state_dict(),
        "shape_projector": model.shape_projector.state_dict(),
        "label_down_projector": model.label_down_projector.state_dict(),
    }
    torch.save(custom_weights, os.path.join(save_dir, "custom_modules.bin"))
    
    model.llava.save_pretrained(os.path.join(save_dir, "llava_projector_lora"))
    print("Done!")

if __name__ == "__main__":
    train()