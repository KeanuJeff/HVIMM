import torch
import os
import json
import matplotlib.pyplot as plt
from models.structural_llava_next import HybirdLlavaFlorenceModel

# ================= 設定區 =================
# 請去您的資料夾看，填入數字最大的那個 checkpoint 路徑
# 例如: "./results_refcocog/checkpoint-600"
CHECKPOINT_PATH = "./results_sharegpt4v/checkpoint-300" 

# 最終輸出的資料夾
SAVE_DIR = "./final_adapter_sharegpt4v"
OUTPUT_DIR = "./results_sharegpt4v"
# =========================================

def rescue_process():
    print(f"🚀 開始從 {CHECKPOINT_PATH} 救援權重...")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ 錯誤：找不到路徑 {CHECKPOINT_PATH}，請確認資料夾名稱。")
        return

    # 1. 重新初始化模型架構 (與訓練時相同)
    print("正在初始化模型架構...")
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    model = HybirdLlavaFlorenceModel(
        llava_model_id=model_id,
        load_llava=True,
        load_florence=False 
    )

    # 2. 載入 Checkpoint 權重
    # Trainer 通常會把權重存成 pytorch_model.bin 或 model.safetensors
    bin_path = os.path.join(CHECKPOINT_PATH, "pytorch_model.bin")
    safe_path = os.path.join(CHECKPOINT_PATH, "model.safetensors")
    
    if os.path.exists(bin_path):
        print(f"正在載入權重: {bin_path}")
        state_dict = torch.load(bin_path, map_location="cpu")
    elif os.path.exists(safe_path):
        from safetensors.torch import load_file
        print(f"正在載入權重: {safe_path}")
        state_dict = load_file(safe_path)
    else:
        print("❌ 找不到權重檔案 (pytorch_model.bin 或 model.safetensors)")
        return

    # 載入權重到模型中
    # strict=False 是因為 checkpoint 可能包含優化器狀態或多餘的 keys，
    # 但只要確保我們的 adapter 相關權重有被載入即可。
    keys = model.load_state_dict(state_dict, strict=False)
    print(f"權重載入完成。未匹配的 keys (通常沒關係): {keys.missing_keys[:5]}...")

    # 3. 執行您原本的儲存邏輯
    print("正在儲存自定義組件...")
    os.makedirs(SAVE_DIR, exist_ok=True)

    custom_weights = {
        "adapter": model.adapter.state_dict(),
        "shape_projector": model.shape_projector.state_dict(),
        "label_down_projector": model.label_down_projector.state_dict(),
    }
    torch.save(custom_weights, os.path.join(SAVE_DIR, "custom_modules.bin"))
    print(f"✅ 成功！權重已救援至: {os.path.join(SAVE_DIR, 'custom_modules.bin')}")

    # 4. 補畫 Loss Curve
    print("正在嘗試繪製 Loss Curve...")
    trainer_state_path = os.path.join(CHECKPOINT_PATH, "trainer_state.json")
    
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, 'r') as f:
            data = json.load(f)
            log_history = data.get('log_history', [])
            if log_history:
                plot_loss_curve(log_history, OUTPUT_DIR)
            else:
                print("⚠️ log_history 為空，無法繪圖")
    else:
        print("⚠️ 找不到 trainer_state.json，跳過繪圖")

def plot_loss_curve(log_history, output_dir):
    # 這是您原本的繪圖函數，我加強了路徑檢查以防再次報錯
    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []

    for entry in log_history:
        if 'loss' in entry and 'step' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])
        if 'eval_loss' in entry and 'step' in entry:
            eval_steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])

    plt.figure(figsize=(10, 6))
    if train_loss:
        plt.plot(train_steps, train_loss, label='Training Loss', color='blue', alpha=0.7)
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label='Validation Loss', color='red', linewidth=2)

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve ShareGPT4V')
    plt.legend()
    plt.grid(True)
    
    # 確保目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, "loss_curve_rescued.png")
    plt.savefig(save_path)
    print(f"✅ Loss curve 已補畫並儲存至 {save_path}")
    plt.close()

if __name__ == "__main__":
    rescue_process()