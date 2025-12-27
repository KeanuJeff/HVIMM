import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, ConcatDataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import sys
from PIL import Image # 記得 import PIL
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

#sys.path.append('./models')
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
    plt.title('Training and Validation Loss Curve RefCOCOg')
    plt.legend()
    plt.grid(True)
    
    # 存檔
    save_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(save_path)
    print(f"Loss curve saved to {save_path}")
    plt.close()


class CaptioningDataset(Dataset):
    def __init__(self, hf_dataset_split, feature_dir, model_processor, tokenizer):
        self.data = hf_dataset_split 
        self.feature_dir = feature_dir
        self.processor = model_processor
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'right'
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'right'
        self.max_length = 3460
        # 實例化工具
        self.geo_utils = GeometricUtils()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if 'question_id' in item:
            image_id = str(item['question_id'])
        elif 'id' in item:
            image_id = str(item['id'])
        else:
            image_id = str(item.get('image_id', idx))
            
        image_obj = item.get('image', item.get('jpg'))
        if image_obj is None:
            image = Image.new('RGB', (336, 336), (0, 0, 0))
        else:
            image = image_obj.convert("RGB")

        target_max_size = 672
        
        w, h = image.size
        w_orig, h_orig = w, h
        scale = min(target_max_size / w, target_max_size / h)
        
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.BICUBIC)
            w, h = new_w, new_h

        # 【關鍵修改】讀取 Raw Data 並現場計算 Fourier (保持不變)
        feat_path = os.path.join(self.feature_dir, f"raw_{image_id}.pt")
        
        shape_feat_tensor = torch.zeros(0, 48) # 空的
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
                    shape_feat_tensor = torch.stack(fourier_list) # (N_obj, 48)
                    centroid_tensor = torch.stack(centroid_list) # (N_obj, 2)
                    obj_labels = valid_labels

            except Exception as e:
                print(f"Error loading raw features for {image_id}: {e}")
                
        bbox = item.get('bbox', [0, 0, 100, 100]) 
        
        caption = "object" 
        
        if 'answer' in item and isinstance(item['answer'], list) and len(item['answer']) > 0:
            caption = item['answer'][-1]
        else:
            print(f"Warning: No valid 'answer' found for ID {image_id}. Using default caption.")

        w, h = max(w, 1), max(h, 1)
        
        # 正規化 BBox (0-100)
        box_norm = [
            int(bbox[0]/w_orig * 100), int(bbox[1]/h_orig * 100), 
            int((bbox[0]+bbox[2])/w_orig * 100), int((bbox[1]+bbox[3])/h_orig * 100)
        ]
        box_norm = [max(0, min(100, x)) for x in box_norm]
        box_str = f"[x0={box_norm[0]}, y0={box_norm[1]}, x1={box_norm[2]}, y1={box_norm[3]}]"

        # 構建訓練輸入
        question = f"[INST] <image>\nDescribe the object at {box_str}, where (0, 0) is at the upper left and (100, 100) is at the lower right. [/INST]"
        answer = caption
        full_text = question + " " + answer
        
        # === 【關鍵修正區域】 Masking Logic ===

        # 1. 計算包含圖片 tokens 的實際 Prompt 長度
        # 這裡 Process Prompt + Image，讓 Processor 展開圖片 Tokens，但不做 padding。
        # 這樣得到的 input_ids 長度就是 Prompt + 圖片 Tokens 的實際長度。
        with torch.no_grad():
             # 使用 do_not_pad 獲取真實長度
            prompt_inputs = self.processor(
                images=image, 
                text=question, 
                return_tensors="pt", 
                padding="do_not_pad",
                truncation=False
            )
        
        prompt_len = prompt_inputs.input_ids.shape[1]

        # 2. 處理完整的訓練序列 (Full Text)
        inputs = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        input_ids = inputs.input_ids[0]
        
        # 3. 製作 Labels (Masking)
        labels = input_ids.clone()
        
        # 將 Prompt 部分 (包含圖片展開後的 tokens) 設為 -100
        if prompt_len < len(labels):
            labels[:prompt_len] = -100
        else:
            # 如果 Prompt 已經比 Max Length 還長 (被 Truncated)，則整條資料無效
            labels[:] = -100

        # 將 Padding 部分設為 -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        

        return {
            "input_ids": input_ids,
            "attention_mask": inputs.attention_mask[0],
            "pixel_values": inputs.pixel_values[0],
            "image_sizes": inputs.image_sizes[0],
            "labels": labels,
            "structural_fourier": shape_feat_tensor,
            "structural_centroids": centroid_tensor,
            "structural_labels": obj_labels # List of strings
        }

class StructuralDataCollator:
    def __call__(self, features):
        batch = {}
        batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
        batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
        batch['labels'] = torch.stack([f['labels'] for f in features])
        batch['image_sizes'] = torch.stack([f['image_sizes'] for f in features])
        
        pixel_values_list = [f['pixel_values'] for f in features]
        # 1. 找出這個 batch 中最多的 patch 數量 (例如: max(5, 3) = 5)
        max_patches = max([x.shape[0] for x in pixel_values_list])
        
        padded_pixel_values = []
        for pv in pixel_values_list:
            # pv shape: (n_patches, 3, 336, 336)
            n_patches = pv.shape[0]
            
            if n_patches < max_patches:
                # 創建 Padding (補 0，代表全黑 patch)
                # shape: (diff, 3, 336, 336)
                pad = torch.zeros((max_patches - n_patches, *pv.shape[1:]), dtype=pv.dtype)
                # 在第 0 維度拼接
                padded_pv = torch.cat([pv, pad], dim=0)
                padded_pixel_values.append(padded_pv)
            else:
                padded_pixel_values.append(pv)
                
        batch['pixel_values'] = torch.stack(padded_pixel_values)
        # =====================================================

        # 處理變長的 structural_fourier 與 centroids
        # 找出這個 batch 中最多的物件數量
        max_objs = max([f['structural_fourier'].shape[0] for f in features])
        max_objs = max(max_objs, 1) 
        
        padded_fourier = []
        padded_centroids = []
        batch_labels_list = [] # List of List of Strings
        
        for f in features:
            f_feat = f['structural_fourier']
            c_feat = f['structural_centroids']
            
            curr_objs = f_feat.shape[0]
            
            # Padding Fourier (N, 48)
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
            
            # Labels 截斷 (如果是截斷 case) 或保持
            # 這裡簡單處理：Trainer 裡會透過 fourier 的 padding 0 來過濾
            batch_labels_list.append(f['structural_labels'][:max_objs])
            
        batch['structural_fourier'] = torch.stack(padded_fourier)
        batch['structural_centroids'] = torch.stack(padded_centroids)
        batch['structural_labels'] = batch_labels_list # 注意這不是 tensor
        
        return batch

def train():
    # 確保 tokenizer 已經被載入
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    feature_dir = "./processed_features_raw" # 對應新的路徑
    
    print("Loading Model (LLaVA ONLY)...")
    # 【關鍵修改】只載入 LLaVA，省下 Florence 顯存
    model = HybirdLlavaFlorenceModel(
        llava_model_id=model_id,
        load_llava=True,
        load_florence=False 
    )

    # 2. 定義 LoRA Config
    #    關鍵：我們利用正則表達式 (regex) 只鎖定 "multi_modal_projector" 裡面的 Linear 層
    
    peft_config = LoraConfig(
        r=128,           # LoRA Rank (可以調大一點，例如 32 或 64，因為 Projector 很重要)
        lora_alpha=256,  # Alpha 通常設為 r 的 2 倍
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM", 
        # 這是最關鍵的一行：只把 LoRA 掛在 Projector 上，不影響 LLM 的其他部分
        target_modules=r".*multi_modal_projector.*linear.*" 
    )

    # 3. 將 LLaVA 包裹成 PEFT 模型
    adapter_path = "./final_adapter_textcaps/custom_modules.bin"
    lora_path = "./final_adapter_textcaps/llava_projector_lora"

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

    print("Preparing Datasets...")
    dataset_name = "lmms-lab/RefCOCOg"
    splits = ['test', 'val']
    
    all_datasets = []
    for split in splits:
        try:
            print(f"Loading {split} split...")
            ds = load_dataset(dataset_name, split=split, streaming=False)
            pytorch_ds = CaptioningDataset(
                hf_dataset_split=ds,
                feature_dir=feature_dir,
                model_processor=model.llava_processor,
                tokenizer=model.llava_processor.tokenizer
            )
            all_datasets.append(pytorch_ds)
        except Exception as e:
            print(f"Skipping {split}: {e}")

    if not all_datasets: raise ValueError("No datasets loaded!")

    combined_train_dataset = ConcatDataset(all_datasets)
    
    training_args = TrainingArguments(
        output_dir="./results_refcocog2",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-4,
        fp16=True,             
        bf16=False,
        optim="adamw_torch",
        max_grad_norm=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False, # 依然必須是 False
        dataloader_num_workers=4     # 建議設高一點，因為現在 Fourier 在 worker 裡算
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

    print("Starting Training...")
    trainer.train()
    plot_loss_curve(trainer.state.log_history, "./results_refcocog2")

    print("Training Complete. Saving weights...")
    save_dir = "./final_adapter_refcocog2"
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