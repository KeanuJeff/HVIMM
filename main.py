import torch
import os
import json
import matplotlib.pyplot as plt
from models.structural_llava_next_raw import HybirdLlavaFlorenceModel

# ==========================================
# 新增函式：從 trainer_state.json 畫 Loss 圖
# ==========================================
def plot_loss_from_checkpoint(checkpoint_dir, output_dir):
    json_path = os.path.join(checkpoint_dir, "trainer_state.json")
    
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Skipping loss plot.")
        return

    print(f"Loading training logs from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # log_history 是一個 list，裡面存著每一步的資訊
    log_history = data.get("log_history", [])
    
    steps = []
    losses = []
    
    for entry in log_history:
        # 確保這一筆 log 包含 loss (有時候可能是 eval loss 或其他資訊)
        if 'loss' in entry and 'step' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])
            
    if not steps:
        print("No loss data found in log history.")
        return

    # 繪圖
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss', color='red', alpha=0.7)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve LLaVA Instruct')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(output_dir, "restored_loss_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")

# ==========================================
# 主函式：提取權重
# ==========================================
def export_weights_from_checkpoint():
    # 1. 設定路徑
    checkpoint_path = "./results_mixed_training/checkpoint-200" 
    output_dir = "./final_adapter_mixed"
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    print(f"Target Checkpoint: {checkpoint_path}")
    print(f"Output Directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 2. 初始化模型架構
    print("Initializing Model Architecture...")
    model = HybirdLlavaFlorenceModel(
        llava_model_id=model_id,
        load_llava=True,     # 為了讀取完整權重，建議還是載入結構
        load_florence=False 
    )

    # 3. 讀取 Checkpoint 權重
    bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    
    state_dict = None
    if os.path.exists(bin_path):
        print(f"Loading weights from {bin_path} ...")
        state_dict = torch.load(bin_path, map_location="cpu")
    elif os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        print(f"Loading weights from {safetensors_path} ...")
        state_dict = load_file(safetensors_path)
    else:
        print("Error: Could not find pytorch_model.bin or model.safetensors.")
        return

    # 載入權重
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
    
    if any("shape_projector" in k for k in missing_keys):
        print("Warning: shape_projector weights seem to be missing!")

    # 4. 儲存提取出的 Custom Weights
    print("Saving separated weights...")
    custom_weights = {
        "adapter": model.adapter.state_dict(),
        "shape_projector": model.shape_projector.state_dict(),
        "label_down_projector": model.label_down_projector.state_dict(),
    }
    torch.save(custom_weights, os.path.join(output_dir, "custom_modules.bin"))
    print(f"Saved custom modules to {os.path.join(output_dir, 'custom_modules.bin')}")

    # ==========================================
    # 5. 執行繪圖函式
    # ==========================================
    print("Extracting Loss Curve...")
    #plot_loss_from_checkpoint(checkpoint_path, "./results_llava_instruct1")

    print("Done! Weights and Loss Curve exported.")

if __name__ == "__main__":
    export_weights_from_checkpoint()