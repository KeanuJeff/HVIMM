import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import os
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel

# 引用你的模型與工具
# 請確保 models 資料夾與 structural_llava_next_raw.py 在同一目錄下
from models.structural_llava_next_raw import HybirdLlavaFlorenceModel, GeometricUtils

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
        plt.plot(train_steps, train_loss, label='Training Loss', color='blue', alpha=0.7)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve IAM (Handwriting)')
    plt.legend()
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "loss_curve_iam.png"))
    plt.close()

# ==========================================
# Dataset 定義：IAM Handwriting Dataset
# ==========================================
class IAMDataset(Dataset):
    def __init__(self, hf_dataset, feature_dir, split_name, model_processor, tokenizer):
        self.processor = model_processor
        self.tokenizer = tokenizer
        self.feature_dir = feature_dir
        self.split_name = split_name 
        
        self.tokenizer.padding_side = 'right'
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'right'
        
        self.max_length = 3460 # IAM 的文字可能較長，保持足夠長度
        self.geo_utils = GeometricUtils() # 初始化幾何工具

        # --- Flatten Logic (一對多) ---
        print(f"Flattening IAM dataset ({split_name})...")
        self.flat_samples = []
        self.hf_dataset = hf_dataset
        
        for row_idx, item in enumerate(self.hf_dataset):
            texts_list = item.get('texts', [])
            # IAM 的結構通常是一張圖對應一個 OCR Q&A，但為了保險起見仍遍歷 list
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

        # 3. 【修改】不讀檔，直接生成全 0 向量
        # 即使沒有物體，我們還是給一個 "dummy object" (全0)，形狀為 (1, dim)
        # 這樣進入 model forward 時才不會因為維度是 0 而報錯
        shape_feat_tensor = torch.zeros(0, 48) 
        centroid_tensor = torch.zeros(0, 2)
        
        # 使用空列表
        obj_labels = []

        # 4. 處理文字
        texts_list = item.get('texts', [])
        conversation = texts_list[text_idx]
        user_input = conversation.get('user', "")
        assistant_response = conversation.get('assistant', "")
        
        # IAM Prompt
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
# Data Collator (通用)
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
    
    # 【修改點】預處理特徵資料夾，請確認已針對 iam 跑過 Florence 預處理
    feature_dir = "./processed_features_iam" 
    
    print("Loading Model (LLaVA ONLY)...")
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

    # 1. 載入上一階段的權重 (RefCOCOg 或 Chart2Text)
    # 假設你想延續之前的權重，路徑保持不變
    adapter_path = "./final_adapter_refcocog1/custom_modules.bin"
    lora_path = "./final_adapter_refcocog1/llava_projector_lora"

    print("=== Loading Pretrained Weights (Previous Stage) ===")
    if os.path.exists(adapter_path):
        print(f"Loading Adapter weights from {adapter_path}...")
        state_dict = torch.load(adapter_path, map_location=model.llava.device)
        model.adapter.load_state_dict(state_dict["adapter"])
        model.shape_projector.load_state_dict(state_dict["shape_projector"])
        model.label_down_projector.load_state_dict(state_dict["label_down_projector"])
    else:
        print("Warning: Custom Adapter weights not found. Using random init.")

    """
    # 如果你也想載入 LoRA 權重，請取消註解
    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}...")
        model.llava = PeftModel.from_pretrained(model.llava, lora_path, is_trainable=True)
    else:
        print("No pretrained LoRA found. Initializing new LoRA...")
        model.llava = get_peft_model(model.llava, peft_config)
    """
    
    # 開啟梯度
    for name, param in model.named_parameters():
        if "adapter" in name or "shape_projector" in name or "label_down_projector" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")

    # 2. 載入 Dataset - IAM Subset
    print("Loading The Cauldron (IAM) Dataset...")
    try:
        # 【修改點】Subset 改為 'iam'
        ds = load_dataset("HuggingFaceM4/the_cauldron", "iam", split="train", streaming=False)
        # ds = ds.select(range(50)) # Debug 用
        
        train_dataset = IAMDataset(
            hf_dataset=ds,
            feature_dir=feature_dir,
            split_name="train",
            model_processor=model.llava_processor,
            tokenizer=model.llava_processor.tokenizer
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results_iam",    # 【修改點】輸出資料夾改名
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-5,
        fp16=True,              
        bf16=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=100,                # IAM 資料量 5.6k，約 700 steps，200存一次差不多
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

    print("Starting Training on IAM...")
    trainer.train()
    plot_loss_curve(trainer.state.log_history, "./results_iam")

    print("Saving weights...")
    save_dir = "./final_adapter_iam"   # 【修改點】最終儲存路徑
    os.makedirs(save_dir, exist_ok=True)

    custom_weights = {
        "adapter": model.adapter.state_dict(),
        "shape_projector": model.shape_projector.state_dict(),
        "label_down_projector": model.label_down_projector.state_dict(),
    }
    torch.save(custom_weights, os.path.join(save_dir, "custom_modules.bin"))
    

if __name__ == "__main__":
    train()