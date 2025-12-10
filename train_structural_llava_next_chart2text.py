import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import sys
from PIL import Image
# 【新增】PeftModel 用於載入權重
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# 假設你的模型檔案在 models 資料夾下
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
        plt.plot(train_steps, train_loss, label='Training Loss', color='blue', alpha=0.7)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve Chart2Text')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

# ==========================================
# Dataset 定義：針對 The Cauldron (Chart2Text)
# ==========================================
class CauldronChartDataset(Dataset):
    def __init__(self, hf_dataset, model_processor, tokenizer):
        self.data = list(hf_dataset)
        self.processor = model_processor
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'right'
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'right'
        self.max_length = 2048 # 圖表描述通常很長

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 處理圖片
        image_list = item.get('images', [])
        if len(image_list) > 0:
            image = image_list[0].convert("RGB")
        else:
            image = Image.new('RGB', (336, 336), (0, 0, 0))

        # Resize 邏輯
        target_max_size = 672 
        w, h = image.size
        scale = min(target_max_size / w, target_max_size / h)
        if scale < 1.0:
            image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

        # 2. 處理文字 (Prompt + Answer)
        texts_list = item.get('texts', [])
        user_input = ""
        assistant_response = ""
        
        if len(texts_list) > 0:
            conversation = texts_list[0]
            user_input = conversation.get('user', "Describe this chart in detail.")
            assistant_response = conversation.get('assistant', "")
        
        # 構建 Prompt
        prompt = f"[INST] <image>\n{user_input} [/INST]"
        full_text = prompt + " " + assistant_response

        # 3. 【修正】使用 Processor 計算 Prompt 長度 (包含 Image Tokens)
        with torch.no_grad():
            prompt_inputs = self.processor(
                images=image, 
                text=prompt, 
                return_tensors="pt", 
                padding="do_not_pad",
                truncation=False
            )
        prompt_len = prompt_inputs.input_ids.shape[1]

        # 4. Tokenization (Full Text)
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
        
        # 5. Masking Labels
        labels = input_ids.clone()
        if prompt_len < len(labels):
            labels[:prompt_len] = -100 # Mask Image Tokens + User Prompt
        else:
            labels[:] = -100 
            
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # ==========================================
        # 強制回傳空的結構特徵 (Zero Objects)
        # ==========================================
        shape_feat_tensor = torch.zeros(0, 48)
        centroid_tensor = torch.zeros(0, 2)
        # 空的 labels 列表
        obj_labels = []

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
# Data Collator (修正版：支援 Pixel Padding)
# ==========================================
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
        # 對於 Chart2Text，這裡全都是 0，但我們還是要 Pad 成 (1, 48) 避免 model forward 報錯
        obj_counts = [f['structural_fourier'].shape[0] for f in features]
        if not obj_counts:
             max_objs = 1
        else:
             max_objs = max(max(obj_counts), 1)
        
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
    
    print("Loading Model (LLaVA ONLY)...")
    # load_florence=False 是正確的，這份 dataset 不需要現場跑 Florence
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

    # =========================================================
    # 【新增】 載入 Pretrained RefCOCOg 權重
    # =========================================================
    adapter_path = "./final_adapter_iconqa/custom_modules.bin"
    lora_path = "./final_adapter_iconqa/llava_projector_lora"

    print("=== Loading Pretrained Weights (RefCOCOg) ===")
    
    # 1. 載入 Custom Modules
    # 雖然這份訓練用不到它們 (輸入全是 0)，但載入權重能保持模型狀態一致，避免 random init 造成干擾
    if os.path.exists(adapter_path):
        print(f"Loading Adapter weights from {adapter_path}...")
        state_dict = torch.load(adapter_path, map_location=model.llava.device)
        model.adapter.load_state_dict(state_dict["adapter"])
        model.shape_projector.load_state_dict(state_dict["shape_projector"])
        model.label_down_projector.load_state_dict(state_dict["label_down_projector"])
    else:
        print("!!! Warning: RefCOCOg Adapter weights not found. Initializing from scratch. !!!")

    # 2. 載入 LoRA
    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}...")
        model.llava = PeftModel.from_pretrained(model.llava, lora_path, is_trainable=True)
    else:
        print("No pretrained LoRA found. Initializing new LoRA...")
        model.llava = get_peft_model(model.llava, peft_config)

    model.llava.print_trainable_parameters()
    
    # 設定梯度
    for name, param in model.named_parameters():
        if "adapter" in name or "shape_projector" in name or "label_down_projector" in name:
            param.requires_grad = True
        else:
            pass
            
    # 2. 載入 Dataset
    print("Loading The Cauldron (Chart2Text) Dataset...")
    try:
        ds = load_dataset("HuggingFaceM4/the_cauldron", "chart2text", split="train", streaming=False)
        # ds = ds.select(range(500)) # 測試用
        
        train_dataset = CauldronChartDataset(
            hf_dataset=ds,
            model_processor=model.llava_processor,
            tokenizer=model.llava_processor.tokenizer
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # BF16 Check
    use_bf16 = torch.cuda.is_bf16_supported()
    print(f"BF16 Supported: {use_bf16}")

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results_chart2text",
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not use_bf16,              
        bf16=use_bf16,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_steps=200,
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
    plot_loss_curve(trainer.state.log_history, "./results_chart2text")

    print("Saving weights...")
    save_dir = "./final_adapter_chart2text"
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