import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import os
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel

# 引用你的模型與工具
from models.structural_llava_next import HybirdLlavaFlorenceModel, GeometricUtils

# ==========================================
# 輔助函式：繪製 Loss 曲線
# ==========================================
def plot_loss_curve(log_history, output_dir):
    train_steps = []
    train_loss = []
    for entry in log_history:
        if 'loss' in entry and 'step' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])
    plt.figure(figsize=(10, 6))
    if train_loss:
        plt.plot(train_steps, train_loss, label='Training Loss', color='purple', alpha=0.7)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve Visual7W (Precomputed)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve_visual7w.png"))
    plt.close()

# ==========================================
# Dataset 定義：讀取預處理特徵 + Flatten
# ==========================================
class Visual7WDataset(Dataset):
    def __init__(self, hf_dataset, feature_dir, split_name, model_processor, tokenizer):
        self.processor = model_processor
        self.tokenizer = tokenizer
        self.feature_dir = feature_dir
        self.split_name = split_name # 用於重建 ID，例如 'train'
        
        self.tokenizer.padding_side = 'right'
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'right'
        
        self.max_length = 2048 
        self.geo_utils = GeometricUtils() # 初始化幾何工具

        # --- Flatten Logic (一對多) ---
        print("Flattening Visual7W dataset...")
        self.flat_samples = []
        self.hf_dataset = list(hf_dataset) 
        
        for row_idx, item in enumerate(self.hf_dataset):
            texts_list = item.get('texts', [])
            for text_idx in range(len(texts_list)):
                # 儲存 (原始 row index, 問題 index)
                self.flat_samples.append((row_idx, text_idx))
        
        print(f"Original Images: {len(self.hf_dataset)}")
        print(f"Total Training Samples: {len(self.flat_samples)}")

    def __len__(self):
        return len(self.flat_samples)

    def __getitem__(self, idx):
        # 1. 取得原始資料索引
        row_idx, text_idx = self.flat_samples[idx]
        item = self.hf_dataset[row_idx]
        
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

        # 3. 【關鍵】讀取預處理的 Florence 特徵
        # ID 必須與 preprocess 檔案中的生成邏輯一致
        file_id = f"visual7w_{self.split_name}_{row_idx}"
        feat_path = os.path.join(self.feature_dir, f"{file_id}.pt")

        shape_feat_tensor = torch.zeros(0, 48)
        centroid_tensor = torch.zeros(0, 2)
        obj_labels = []

        if os.path.exists(feat_path):
            try:
                raw_data = torch.load(feat_path) # {'labels': [], 'polygons': []}
                raw_labels = raw_data.get('labels', [])
                raw_polys = raw_data.get('polygons', [])
                
                fourier_list = []
                centroid_list = []
                valid_labels = []

                # 現場計算 Fourier (CPU 計算比 Florence 推論快很多，這步保留在 DataLoader 是 OK 的)
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
                print(f"Error loading features for {file_id}: {e}")
        else:
            # 如果檔案不存在 (可能是 preprocess 漏掉或錯誤)，就使用空的特徵，不報錯
            pass

        # 4. 處理文字
        texts_list = item.get('texts', [])
        conversation = texts_list[text_idx]
        user_input = conversation.get('user', "")
        assistant_response = conversation.get('assistant', "")
        
        prompt = f"[INST] <image>\n{user_input} [/INST]"
        full_text = prompt + " " + assistant_response

        # 5. Masking Logic
        with torch.no_grad():
            prompt_inputs = self.processor(
                images=image, 
                text=prompt, 
                return_tensors="pt", 
                padding="do_not_pad",
                truncation=False
            )
        prompt_len = prompt_inputs.input_ids.shape[1]

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
        
        labels = input_ids.clone()
        if prompt_len < len(labels):
            labels[:prompt_len] = -100 
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

# ==========================================
# Data Collator (與 RefCOCOg 通用)
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
# Main Training Function
# ==========================================
def train():
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    feature_dir = "./processed_features_visual7w" # 預處理檔案的路徑
    
    print("Loading Model (LLaVA ONLY)...")
    # 【關鍵】不載入 Florence，節省顯存
    model = HybirdLlavaFlorenceModel(
        llava_model_id=model_id,
        load_llava=True,
        load_florence=False 
    )

    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=r".*multi_modal_projector.*linear.*" 
    )

    # 1. 載入上一階段的權重 (Chart2Text)
    adapter_path = "./final_adapter_chart2text/custom_modules.bin"
    lora_path = "./final_adapter_chart2text/llava_projector_lora"

    print("=== Loading Pretrained Weights (Chart2Text) ===")
    if os.path.exists(adapter_path):
        print(f"Loading Adapter weights from {adapter_path}...")
        state_dict = torch.load(adapter_path, map_location=model.llava.device)
        model.adapter.load_state_dict(state_dict["adapter"])
        model.shape_projector.load_state_dict(state_dict["shape_projector"])
        model.label_down_projector.load_state_dict(state_dict["label_down_projector"])
    else:
        print("Warning: Custom Adapter weights not found.")

    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}...")
        model.llava = PeftModel.from_pretrained(model.llava, lora_path, is_trainable=True)
    else:
        print("No pretrained LoRA found. Initializing new LoRA...")
        model.llava = get_peft_model(model.llava, peft_config)

    model.llava.print_trainable_parameters()
    
    # 開啟梯度
    for name, param in model.named_parameters():
        if "adapter" in name or "shape_projector" in name or "label_down_projector" in name:
            param.requires_grad = True
        else:
            pass
            
    # 2. 載入 Dataset
    print("Loading The Cauldron (Visual7W) Dataset...")
    try:
        ds = load_dataset("HuggingFaceM4/the_cauldron", "visual7w", split="train", streaming=False)
        # ds = ds.select(range(100)) # Debug 用
        
        train_dataset = Visual7WDataset(
            hf_dataset=ds,
            feature_dir=feature_dir,
            split_name="train", # 傳入 split 名稱以生成正確 ID
            model_processor=model.llava_processor,
            tokenizer=model.llava_processor.tokenizer
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    use_bf16 = torch.cuda.is_bf16_supported()

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results_visual7w",
        per_device_train_batch_size=2, # 因為不需要跑 Florence，這裡甚至可以嘗試開大一點 (例如 2 或 4)
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not use_bf16,              
        bf16=use_bf16,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=500,
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
    plot_loss_curve(trainer.state.log_history, "./results_visual7w")

    print("Saving weights...")
    save_dir = "./final_adapter_visual7w"
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