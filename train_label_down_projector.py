import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LlavaNextForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
import os
from PIL import Image
from tqdm import tqdm
import glob
from functools import partial

# ==========================================
# 1. COCO 圖片 Dataset (自編碼專用)
# ==========================================
class COCOReconstructionDataset(Dataset):
    def __init__(self, img_dir, processor):
        self.img_dir = img_dir
        # 根據 image_765dc2.jpg，讀取資料夾下所有的 jpg
        self.image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
        self.processor = processor
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {img_dir}")
        print(f"Found {len(self.image_paths)} images for training.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            
            # 【使用者要求】強制 Resize 成 336x336
            #image = image.resize((336, 336), Image.BICUBIC)
            
            return image
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            # 簡單容錯：隨機回傳第一張
            return self.__getitem__(0)


def custom_collate_fn(batch_images, processor):
    # 使用 LLaVA processor 處理圖像
    inputs = processor.image_processor(images=batch_images, return_tensors="pt")
    return inputs['pixel_values']

# ==========================================
# 2. 訓練與模型邏輯
# ==========================================
def train_projector():
    # 路徑設定
    IMAGE_DIR = "C:/Users/User/Desktop/HVIMM/dataset/VQA_v2/val2014/val2014"
    MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
    SAVE_PATH = "./pretrained_projector/label_down_projector_autoencoder.bin"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading LLaVA (Frozen)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # 載入主模型
    llava = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config,
        low_cpu_mem_usage=True
    )
    llava.eval() # 絕對凍結

    # 取得維度資訊
    vision_dim = llava.vision_tower.config.hidden_size # 1024
    projector_dim = 4096 # LLaVA Projector 輸出維度
    
    # 初始化我們要訓練的 label_down_projector
    # 目標：Input(4096) -> Output(1024)
    label_down_projector = nn.Sequential(
        nn.LayerNorm(projector_dim, eps=1e-6),
        nn.Linear(projector_dim, vision_dim)
    ).to(device, dtype=torch.float32)

    label_down_projector.train()
    
    # 優化器
    optimizer = torch.optim.AdamW(label_down_projector.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    # 準備 DataLoader
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    dataset = COCOReconstructionDataset(IMAGE_DIR, processor)
    collate_func = partial(custom_collate_fn, processor=processor)

    # 4. 傳遞給 DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=4, # Windows 下建議也可以先設為 0 測試，若要用多核則必須解決 pickle 問題
        collate_fn=collate_func 
    )
    
    print("Start Training Label Down Projector...")
    
    epochs = 1
    for epoch in range(epochs):
        loop = tqdm(dataloader)
        total_loss = 0
        
        for pixel_values in loop:
            pixel_values = pixel_values.to(device, dtype=torch.float16)
            if pixel_values.ndim == 5:
                b, n, c, h, w = pixel_values.shape
                # 將 Batch 和 Patches 維度合併 -> (Batch*Patches, C, H, W)
                pixel_values = pixel_values.view(b * n, c, h, w)
            
            with torch.no_grad():
                # 1. 通過 Vision Tower 得到 Feature A (1024維) -> 這是 Ground Truth
                # output: (Batch, N_patches, 1024)
                vision_outputs = llava.vision_tower(pixel_values, output_hidden_states=True)
                feature_original_1024 = vision_outputs.hidden_states[-2]
                
                # 2. 通過 LLaVA Projector 得到 Feature B (4096維) -> 這是 Input
                feature_projected_4096 = llava.multi_modal_projector(feature_original_1024)
            
            # 3. 通過我們的 Down Projector 嘗試還原 (Feature B -> Feature A')
            # 記得轉 float32 算 loss 以求精確，或者保持 float16
            feature_restored_1024 = label_down_projector(feature_projected_4096.float())
            
            # 4. 計算 Reconstruction Loss
            loss = loss_fn(feature_restored_1024, feature_original_1024.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{epochs}")
            loop.set_postfix(loss=loss.item())
    
    # 存檔
    os.makedirs("./pretrained_projector", exist_ok=True)
    torch.save(label_down_projector.state_dict(), SAVE_PATH)
    print(f"Training Done. Weights saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_projector()