import torch
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from transformers import AutoTokenizer
from PIL import Image
# 【新增】需要 PeftModel 載入舊權重
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# 確保路徑能找到你的 models 和 dataset 資料夾
# sys.path.append('./models')
# sys.path.append('./dataset')

from models.structural_llava_next import HybirdLlavaFlorenceModel, GeometricUtils
from dataset.iconqa import IconQADataset

# =================================================================
# (A) 輔助函數 (Loss Curve & Prompts)
# =================================================================

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
    plt.title('Training Loss Curve IconQA')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(save_path)
    plt.close()

def format_mc_prompt_direct(question, choices_list):
    prompt = "Answer the following multiple-choice question with only the correct option letter (A, B, C, or D).\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Options:\n"
    for i, choice in enumerate(choices_list):
        label = chr(65 + i)
        prompt += f"{label}. {choice}\n"
    prompt += "Answer:" 
    return prompt

def format_vqa_prompt_direct(question):
    return f"Answer the following question with only the single numerical or textual answer.\n\nQuestion: {question}\nAnswer:"

# ==========================================
# 1. 訓練用的 Wrapper Dataset
# ==========================================
class IconQALlavaDataset(Dataset):
    def __init__(self, iconqa_dataset, feature_dir, model_processor):
        self.dataset = iconqa_dataset
        self.feature_dir = feature_dir
        self.processor = model_processor
        self.tokenizer = model_processor.tokenizer
        self.tokenizer.padding_side = 'right'
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'right'
        self.geo_utils = GeometricUtils()
        self.max_length = 3460
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 讀取資料
        image = item['image'].convert("RGB")
        q_id = str(item['question_id'])
        question = item['question']
        q_type = item['question_type']
        choices = item.get('choices', [])
        
        # 處理 Answer Label
        if 'answers' in item and len(item['answers']) > 0:
            answer_label = item['answers'][0]
        else:
            answer_label = "" 

        # -------------------------------------------------------
        # 讀取 Raw Features 並現場計算 Fourier
        # -------------------------------------------------------
        feat_path = os.path.join(self.feature_dir, f"raw_{q_id}.pt")
        
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

                # CPU 計算
                for i, poly_list in enumerate(raw_polys):
                    if len(poly_list) < 6: continue
                    
                    target_poly = torch.tensor(poly_list, dtype=torch.float32).reshape(-1, 2)
                    target_poly = target_poly.unsqueeze(0)

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
                print(f"[Warning] Error loading raw features for {q_id}: {e}")
        
        # -------------------------------------------------------
        # Prompt 與 Label 構建
        # -------------------------------------------------------
        if q_type == "multiple-choice":
            VQA_PROMPT = format_mc_prompt_direct(question, choices)
            label_text = answer_label.upper()
        else: 
            VQA_PROMPT = format_vqa_prompt_direct(question)
            label_text = answer_label.strip().lower()

        prompt_with_cue = f"[INST] <image>\n{VQA_PROMPT} [/INST]"
        full_text = prompt_with_cue + " " + label_text

        # -------------------------------------------------------
        # 【修正】Masking 邏輯：使用 Processor 計算 Prompt 長度
        # -------------------------------------------------------
        
        # 1. 計算 Prompt (包含 Image Tokens) 的長度
        with torch.no_grad():
            prompt_inputs = self.processor(
                images=image, 
                text=prompt_with_cue, 
                return_tensors="pt", 
                padding="do_not_pad",
                truncation=False
            )
        prompt_len = prompt_inputs.input_ids.shape[1]

        # 2. 處理完整序列
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
        
        # 3. 製作 Labels
        labels = input_ids.clone()
        
        # Mask 掉 Prompt (User + Image Tokens)
        if prompt_len < len(labels):
            labels[:prompt_len] = -100
        else:
            labels[:] = -100
            
        # Mask 掉 Padding
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs.attention_mask[0],
            "pixel_values": inputs.pixel_values[0],
            "image_sizes": image_sizes, # 必須回傳這個
            "labels": labels,
            "structural_fourier": shape_feat_tensor,
            "structural_centroids": centroid_tensor,
            "structural_labels": obj_labels
        }

# ==========================================
# 2. Data Collator (修正版)
# ==========================================
class StructuralDataCollator:
    def __call__(self, features):
        batch = {}
        batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
        batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
        batch['labels'] = torch.stack([f['labels'] for f in features])
        batch['image_sizes'] = torch.stack([f['image_sizes'] for f in features])
        
        # 1. 處理 pixel_values 的動態 Patch 數量 Padding (解決 RuntimeError)
        pixel_values_list = [f['pixel_values'] for f in features]
        max_patches = max([x.shape[0] for x in pixel_values_list])
        
        padded_pixel_values = []
        for pv in pixel_values_list:
            n_patches = pv.shape[0]
            if n_patches < max_patches:
                # 補黑圖 Patch
                pad = torch.zeros((max_patches - n_patches, *pv.shape[1:]), dtype=pv.dtype)
                padded_pv = torch.cat([pv, pad], dim=0)
                padded_pixel_values.append(padded_pv)
            else:
                padded_pixel_values.append(pv)
        batch['pixel_values'] = torch.stack(padded_pixel_values)

        # 2. 處理幾何特徵 Padding
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
# 3. 主訓練流程
# ==========================================
def train():
    feature_dir = "./processed_features_iconqa_raw"
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    try:
        temp_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        temp_tokenizer.padding_side = 'right'
        temp_tokenizer.save_pretrained(model_id) # 儲存設定，讓後續 AutoProcessor 繼承
        del temp_tokenizer
    except Exception as e:
        print(f"Warning: Could not force padding_side='right'. Relying on AutoProcessor defaults. Error: {e}")
    
    print("Loading Model (LLaVA ONLY)...")
    model = HybirdLlavaFlorenceModel(
        llava_model_id=model_id,
        load_llava=True,
        load_florence=False 
    )

    print("Preparing LLaVA for k-bit training...")

    # 定義 LoRA Config
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
    adapter_path = "./final_adapter_sharegpt4/custom_modules.bin"
    lora_path = "./final_adapter_sharegpt4/llava_projector_lora"

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
    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}...")
        model.llava = PeftModel.from_pretrained(model.llava, lora_path, is_trainable=True)
    else:
        print("No pretrained LoRA found. Initializing new LoRA...")
        model.llava = get_peft_model(model.llava, peft_config)

    model.llava.print_trainable_parameters()

    # =========================================================
    # 設定可訓練參數
    # =========================================================
    print("Setting up gradients for custom modules...")
    for name, param in model.named_parameters():
        if "adapter" in name or "shape_projector" in name or "label_down_projector" in name:
            param.requires_grad = True
        else:
            # LoRA 已經由 PeftModel 處理，這裡只需確保自定義層開啟
            pass

    print("Loading IconQA Dataset...")
    # 載入資料集
    raw_dataset = IconQADataset(split='train', num_samples=None) # 使用完整數據集
    # 測試時可以用: raw_dataset = IconQADataset(split='train_sample', num_samples=1000)
    
    train_dataset = IconQALlavaDataset(raw_dataset, feature_dir, model.llava_processor)
    print(f"Total Training Samples: {len(train_dataset)}")

    # 自動偵測 BF16
    use_bf16 = torch.cuda.is_bf16_supported()
    print(f"BF16 Supported: {use_bf16}")

    args = TrainingArguments(
        output_dir="./results_iconqa",
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not use_bf16,              
        bf16=use_bf16,
        optim="adamw_torch",
        logging_steps=1,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=4
    )

    if hasattr(model.llava, "gradient_checkpointing_enable"):
        model.llava.gradient_checkpointing_enable()
        
    model.enable_input_require_grads()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=StructuralDataCollator()
    )

    print("Starting Training...")
    trainer.train()
    plot_loss_curve(trainer.state.log_history, "./results_iconqa")
    
    # 儲存權重
    print("Training Complete. Saving weights...")
    save_dir = "./final_adapter_iconqa"
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