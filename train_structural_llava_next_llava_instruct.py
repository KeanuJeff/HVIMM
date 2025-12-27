import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import os
from PIL import Image
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# 假設你的 models 資料夾在當前目錄
from models.structural_llava_next_raw import HybirdLlavaFlorenceModel, GeometricUtils

# 設定圖片路徑 (請確保此路徑正確指向 COCO train2017 資料夾)
COCO_IMAGE_DIR = "./dataset/train2017" 

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
    plt.title('Training Loss Curve LLaVA-Instruct-150K')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

class LLaVAInstructDataset(Dataset):
    def __init__(self, hf_dataset, feature_dir, model_processor, tokenizer):
        self.feature_dir = feature_dir
        self.processor = model_processor
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'right'
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'right'
        self.max_length = 3460
        self.geo_utils = GeometricUtils()

        # --- 【資料篩選與索引建立】 ---
        print("Filtering valid data based on feature paths...")
        self.original_data = list(hf_dataset)
        self.valid_data_list = [] # 儲存 (原始資料, 對應到的feature路徑)
        
        for item in self.original_data:
            # LLaVA-Instruct 格式: {'id': '000000215677', 'image': '000000215677.jpg', ...}
            image_filename = item.get('image', '')
            
            # 嘗試解析 ID (有些 feature 檔名可能是純數字 id，有些可能是補零後的 id)
            # 優先使用 dataset 中的 id 欄位
            raw_id = str(item.get('id', ''))
            
            # 策略 1: 直接用 raw_id (例如 raw_000000215677.pt)
            feat_path_candidate_1 = os.path.join(self.feature_dir, f"raw_{raw_id}.pt")
            
            # 策略 2: 去除 leading zeros (例如 raw_215677.pt) - 針對 COCO 常見格式
            short_id = raw_id.lstrip('0')
            if short_id == '': short_id = '0' # 避免全0變成空字串
            feat_path_candidate_2 = os.path.join(self.feature_dir, f"raw_{short_id}.pt")
            
            final_feat_path = None
            if os.path.exists(feat_path_candidate_1):
                final_feat_path = feat_path_candidate_1
            elif os.path.exists(feat_path_candidate_2):
                final_feat_path = feat_path_candidate_2
            
            # 檢查圖片是否存在 (可選，避免訓練時報錯)
            image_path_full = os.path.join(COCO_IMAGE_DIR, image_filename)
            # 如果圖片不在根目錄，嘗試 train2017 子目錄
            if not os.path.exists(image_path_full):
                 image_path_full = os.path.join(COCO_IMAGE_DIR, "train2017", image_filename)

            # 只有當 Feature 存在 且 圖片存在 時才加入訓練列表
            if final_feat_path and os.path.exists(image_path_full):
                # 將正確的 feature path 綁定進去，避免 getitem 時還要重找
                self.valid_data_list.append({
                    "data": item,
                    "feat_path": final_feat_path,
                    "image_path": image_path_full
                })

        print(f"Total original samples: {len(self.original_data)}")
        print(f"Valid samples for training (Image & Feature found): {len(self.valid_data_list)}")
        # --- 【篩選結束】 ---

    def __len__(self):
        return len(self.valid_data_list)

    def __getitem__(self, idx):
        record = self.valid_data_list[idx]
        item = record["data"]
        feat_path = record["feat_path"]
        image_path = record["image_path"]
        
        # 1. 載入 Preprocess 好的 Florence 特徵
        shape_feat_tensor = torch.zeros(0, 48)
        centroid_tensor = torch.zeros(0, 2)
        obj_labels = []

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
            print(f"Error loading feats for {feat_path}: {e}")

        # 2. 載入圖片
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # 萬一發生讀取錯誤，生成黑圖避免 crash
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (336, 336), (0, 0, 0))

        target_max_size = 672
        w, h = image.size
        scale = min(target_max_size / w, target_max_size / h)
        if scale < 1.0:
            image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

        # 3. 解析 Conversations (LLaVA-Instruct 格式)
        conversations = item.get('conversations', [])
        human_input = ""
        gpt_response = ""
        
        # 簡單解析：通常第一句是 human, 第二句是 gpt
        for turn in conversations:
            if turn['from'] == 'human':
                human_input = turn['value']
            elif turn['from'] == 'gpt':
                gpt_response = turn['value']
        
        # 處理 <image> tag
        human_input = human_input.replace("<image>", "").strip()
        prompt = f"[INST] <image>\n{human_input} [/INST]"
        full_text = prompt + " " + gpt_response

        # 4. Processor 計算 Prompt 長度
        with torch.no_grad():
            prompt_inputs = self.processor(
                images=image, 
                text=prompt, 
                return_tensors="pt", 
                padding="do_not_pad",
                truncation=False
            )
        prompt_len = prompt_inputs.input_ids.shape[1]

        # 5. Tokenization
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

class StructuralDataCollator:
    def __call__(self, features):
        batch = {}
        batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
        batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
        batch['labels'] = torch.stack([f['labels'] for f in features])
        batch['image_sizes'] = torch.stack([f['image_sizes'] for f in features])
        
        # 1. Pixel Values Padding
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

        # 2. Structural Feature Padding
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
    # 【注意】這裡的路徑依然是你存放 Feature 的資料夾
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

    adapter_path = "./final_adapter_textcaps1/custom_modules.bin"

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

    # 設定梯度
    for name, param in model.named_parameters():
        if "adapter" in name or "shape_projector" in name or "label_down_projector" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")

    print("Preparing LLaVA-Instruct-150K Dataset...")
    try:
        # 【修正重點】：不直接 load_dataset 整個 repo，而是只下載單一 JSON 檔案
        repo_id = "liuhaotian/LLaVA-Instruct-150K"
        filename = "llava_instruct_150k.json"
        
        print(f"Downloading {filename} from {repo_id}...")
        json_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        
        print(f"Loading data from {json_path}...")
        # 使用 "json" 類型載入本地檔案，這樣最穩定
        ds = load_dataset("json", data_files=json_path, split="train")
        
        # ds = ds.select(range(500)) # 測試用
        
        train_dataset = LLaVAInstructDataset(
            hf_dataset=ds,
            feature_dir=feature_dir,
            model_processor=model.llava_processor,
            tokenizer=model.llava_processor.tokenizer
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    training_args = TrainingArguments(
        output_dir="./results_llava_instruct1",
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=4, 
        num_train_epochs=1,
        learning_rate=1e-4,
        fp16=True,              
        bf16=False,
        optim="adamw_torch",
        max_grad_norm=0.5,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
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
    plot_loss_curve(trainer.state.log_history, "./results_llava_instruct1")

    print("Saving weights...")
    save_dir = "./final_adapter_llava_instruct1"
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