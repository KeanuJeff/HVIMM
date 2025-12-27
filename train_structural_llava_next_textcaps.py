import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import os
from PIL import Image
from peft import LoraConfig

# 引用模型與工具
from models.structural_llava_next_raw import HybirdLlavaFlorenceModel, GeometricUtils


def plot_loss_curve(log_history, output_dir):
    """
    從 trainer.state.log_history 解析數據並繪製 Loss 曲線
    """
    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []

    for entry in log_history:
        # 收集訓練 Loss
        if 'loss' in entry and 'step' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])
        
        # 收集驗證 Loss (如果有的話)
        if 'eval_loss' in entry and 'step' in entry:
            eval_steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])

    plt.figure(figsize=(10, 6))
    
    # 畫 Training Loss
    if train_loss:
        plt.plot(train_steps, train_loss, label='Training Loss', color='blue', alpha=0.7)
    
    # 畫 Validation Loss
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label='Validation Loss', color='red', linewidth=2)

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve Textcaps')
    plt.legend()
    plt.grid(True)
    
    # 存檔
    save_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(save_path)
    print(f"Loss curve saved to {save_path}")
    plt.close()

# ==========================================
# Dataset 定義：讀取預處理特徵 + 現場計算 Fourier
# ==========================================
class TextCapsDataset(Dataset):
    def __init__(self, hf_dataset, feature_dir, model_processor, tokenizer):
        self.hf_dataset = hf_dataset
        self.feature_dir = feature_dir
        self.processor = model_processor
        self.tokenizer = tokenizer
        
        # 設定 Tokenizer
        self.tokenizer.padding_side = 'right'
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'right'
            
        self.max_length = 3460
        self.geo_utils = GeometricUtils()

        # ==========================================
        # 新增：過濾邏輯，確保只訓練有特徵檔的樣本
        # ==========================================
        print("Filtering dataset: checking for available feature files...")
        self.valid_indices = []
        for idx in range(len(self.hf_dataset)):
            # 根據你的檔名邏輯：textcaps_train_{idx}.pt
            file_id = f"textcaps_train_{idx}"
            feat_path = os.path.join(self.feature_dir, f"{file_id}.pt")
            
            if os.path.exists(feat_path):
                self.valid_indices.append(idx)
        
        print(f"Original Dataset Size: {len(self.hf_dataset)}")
        print(f"Filtered Dataset Size (Features Found): {len(self.valid_indices)}")

    def __len__(self):
        # 回傳過濾後的長度
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 1. 取得過濾後的真實索引
        real_idx = self.valid_indices[idx]
        item = self.hf_dataset[real_idx]
        
        # 2. 處理圖片
        image_list = item.get('images', [])
        if len(image_list) > 0:
            image = image_list[0].convert("RGB")
        else:
            image = Image.new('RGB', (336, 336), (0, 0, 0))

        # Resize
        target_max_size = 672 
        w, h = image.size
        scale = min(target_max_size / w, target_max_size / h)
        if scale < 1.0:
            image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

        # 3. 讀取預處理特徵 (使用 real_idx)
        file_id = f"textcaps_train_{real_idx}"
        feat_path = os.path.join(self.feature_dir, f"{file_id}.pt")

        # 預設值 (理論上經過過濾後不應進入空值，但保留結構安全)
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

                # 現場計算幾何特徵
                for i, poly_list in enumerate(raw_polys):
                    if len(poly_list) < 6: continue
                    
                    target_poly = torch.tensor(poly_list, dtype=torch.float32).reshape(-1, 2)
                    target_poly = target_poly.unsqueeze(0) 

                    shape_feat = self.geo_utils.fourier_shape_encoding(target_poly, num_harmonics=24)
                    centroid = self.geo_utils.calculate_centroid(target_poly)
                    
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
                print(f"Error loading {file_id}: {e}")

        # 4. 處理文字與 Prompt
        texts_list = item.get('texts', [])
        user_input = ""
        assistant_response = ""
        
        if len(texts_list) > 0:
            conv = texts_list[0]
            user_input = conv.get('user', "Describe this image.")
            assistant_response = conv.get('assistant', "")

        prompt = f"[INST] <image>\n{user_input} [/INST]"
        full_text = prompt + " " + assistant_response

        # 5. Tokenization & Masking
        with torch.no_grad():
            prompt_inputs = self.processor(
                images=image, text=prompt, return_tensors="pt", padding="do_not_pad"
            )
        prompt_len = prompt_inputs.input_ids.shape[1]

        inputs = self.processor(
            images=image, text=full_text, return_tensors="pt", 
            padding="max_length", max_length=self.max_length, truncation=True
        )
        
        input_ids = inputs.input_ids[0]
        labels = input_ids.clone()
        
        # Mask 掉 Prompt 部分
        if prompt_len < len(labels):
            labels[:prompt_len] = -100
        else:
            labels[:] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs.attention_mask[0],
            "pixel_values": inputs.pixel_values[0],
            "image_sizes": inputs.image_sizes[0],
            "labels": labels,
            "structural_fourier": shape_feat_tensor,
            "structural_centroids": centroid_tensor,
            "structural_labels": obj_labels
        }

# ==========================================
# Data Collator (與之前通用)
# ==========================================
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
        
        # Structural Features Padding
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

# ==========================================
# Main Training Logic
# ==========================================
def train():
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    # 設定預處理特徵的路徑
    feature_dir = "./processed_features_textcaps" 
    
    print("Loading Model (LLaVA ONLY)...")
    # 只載入 LLaVA，不載入 Florence (因為特徵已經算好了)
    model = HybirdLlavaFlorenceModel(
        llava_model_id=model_id,
        load_llava=True,
        load_florence=False 
    )

    peft_config = LoraConfig(
        r=128, lora_alpha=256, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", 
        target_modules=r".*multi_modal_projector.*linear.*" 
    )
    
    adapter_path = "./final_adapter_llava_instruct/custom_modules.bin"
    lora_path = "./final_adapter_llava_instruct/llava_projector_lora"

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
    """
    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}...")
        model.llava = PeftModel.from_pretrained(model.llava, lora_path, is_trainable=True)
    else:
        print("No pretrained LoRA found. Initializing new LoRA...")
        model.llava = get_peft_model(model.llava, peft_config)
    model.llava.print_trainable_parameters() # 這會印出有多少參數是用 LoRA 訓練的
    """
    
    
    print("Setting up gradients for custom modules...")
    for name, param in model.named_parameters():
        # A. 針對你的自定義組件 (全參數訓練)
        if "adapter" in name or "shape_projector" in name or "label_down_projector" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")

    print("Loading Dataset...")
    ds = load_dataset("HuggingFaceM4/the_cauldron", "textcaps", split="train", streaming=False)
    
    train_dataset = TextCapsDataset(
        hf_dataset=ds,
        feature_dir=feature_dir,
        model_processor=model.llava_processor,
        tokenizer=model.llava_processor.tokenizer
    )

    training_args = TrainingArguments(
        output_dir="./results_textcaps1",    # 【修改點】輸出資料夾改名
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-4,
        fp16=True,              
        bf16=False,
        optim="adamw_torch",
        #max_grad_norm=0.5,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=100, 
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=4
    )
    
    # 開啟 Checkpointing
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
    
    plot_loss_curve(trainer.state.log_history, "./results_textcaps1")

    print("Training Complete. Saving weights...")
    save_dir = "./final_adapter_textcaps1"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 儲存自定義組件 (Adapter, Shape Projector, Label Down Projector)
    #    這些是 Full Rank 的權重，我們手動打包
    custom_weights = {
        "adapter": model.adapter.state_dict(),
        "shape_projector": model.shape_projector.state_dict(),
        "label_down_projector": model.label_down_projector.state_dict(),
    }
    torch.save(custom_weights, os.path.join(save_dir, "custom_modules.bin"))
    print(f"Custom modules saved to {save_dir}/custom_modules.bin")

if __name__ == "__main__":
    train()