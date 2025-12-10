# eval_chartqa.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import yaml
import torch
import gc
import json 
import re 
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 引入新的 Dataset ---
from dataset.chartqa import ChartQADataset

# --- 沿用現有的 utils ---
from dataset.utils import load_model_and_processor, collate_fn
from eval_iconqa import generate_answer_vqa

# =================================================================
# ChartQA 專用評分邏輯 (Relaxed Accuracy)
# =================================================================

def is_number(s):
    """判斷字串是否為數字"""
    try:
        float(s.replace('%', '').replace(',', ''))
        return True
    except ValueError:
        return False

def get_number(s):
    """提取數字數值"""
    try:
        return float(s.replace('%', '').replace(',', ''))
    except ValueError:
        return None

def score_chartqa_relaxed(pred, ref):
    """
    ChartQA 的標準評估: 
    1. 如果是純文字，做 Exact Match (轉小寫)。
    2. 如果是數字，允許 5% 的誤差範圍。
    """
    if not pred: return 0.0
    
    pred_str = str(pred).strip().lower()
    ref_str = str(ref).strip().lower()
    
    # 1. 嘗試完全匹配
    if pred_str == ref_str:
        return 1.0
        
    # 2. 數值寬鬆匹配 (Relaxed Accuracy)
    # 從字串中提取出數值部分
    val_pred = get_number(pred_str)
    val_ref = get_number(ref_str)
    
    if val_pred is not None and val_ref is not None:
        # 避免除以零
        if val_ref == 0:
            return 1.0 if abs(val_pred - val_ref) < 1e-6 else 0.0
        # 允許 5% 誤差
        if abs(val_pred - val_ref) / abs(val_ref) <= 0.05:
            return 1.0
            
    # 3. 如果包含在內 (例如 ref="red bar", pred="the red bar is higher")
    # ChartQA 有時也接受簡單的包含匹配，視你的嚴格程度而定
    # 這裡我們為了嚴謹，若非數值則堅持 Exact Match (或你可以自行放寬)
    return 0.0

# =================================================================
# 主程式
# =================================================================

def main():
    config_dir = "configs"
    config_filename = 'chartqa.yaml' # <--- 指向新的 yaml
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    # 1. 載入 ChartQA Dataset
    ds = ChartQADataset(
        dataset_id=cfg["dataset_id"],
        subset=cfg["dataset_subset"],
        split="train", 
        num_samples=cfg["num_val_samples"]
    )
    
    batch_size = cfg.get('batch_size', 1)
    print(f"Using Batch Size: {batch_size}")
    
    ds_loader = DataLoader(
        ds, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_fn 
    )
    
    all_results = {}
    
    for mdl in cfg["models"]:
        name = mdl["name"]
        print(f"\n--- Evaluating Model: {name} on ChartQA ---")
        
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        mdl_obj.eval()
        
        scores = []
        preds = []
        
        # ChartQA 主要是 VQA 格式
        generate_params = {
            "max_new_tokens": cfg.get("max_new_tokens_vqa", 200),
            "do_sample": False,
            "min_new_tokens": 1,
            "num_beams": 1
        }

        for batch_data in tqdm(ds_loader, desc=name):
            
            images = batch_data["image"]
            questions = batch_data["question"]
            all_answers = batch_data["answers"] 
            question_ids = batch_data["question_id"]
            
            # 直接使用 generate_answer_vqa (從你的 utils.py 來的，或是用 iconqa 裡的定義)
            # 注意: 如果 utils.py 沒有 generate_answer_vqa，請確保你在 eval_chartqa.py 
            # 裡保留了該函數的定義 (或是從 eval_iconqa import 進來)
            ans_list = generate_answer_vqa(
                proc, mdl_obj, mtype, 
                images, questions, 
                **generate_params
            )
            
            # 評分
            for i, ans in enumerate(ans_list):
                refs = all_answers[i] # 這是個 list, e.g. ["38.5"]
                
                # 取出第一個參考答案 (ChartQA 通常只有一個標準答案)
                ref = refs[0] if refs else ""
                
                # 使用 Relaxed Accuracy 評分
                score = score_chartqa_relaxed(ans, ref)
                scores.append(score)
                
                preds.append({
                    "qid": question_ids[i],
                    "question": questions[i],
                    "pred": ans,
                    "ref": ref,
                    "score": score
                })
            del images, questions, all_answers, ans_list
            if 'batch_data' in locals():
                del batch_data
            
            # 這是關鍵：如果 Python 物件沒死，PyTorch 就不能釋放 GPU 記憶體
            gc.collect() 
            torch.cuda.empty_cache()
        
        mean_acc = (sum(scores) / len(scores)) * 100 if scores else 0.0
        
        all_results[name] = {
            "acc_relaxed": mean_acc,
            "count": len(scores),
            "predictions": preds
        }
        
        output_file = f"results_chartqa_{name.replace(' ', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results[name], f, indent=4, ensure_ascii=False)
        print(f"Results for {name} saved to {output_file}")

        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Final ChartQA Evaluation Results ===")
    for k, v in all_results.items():
        print(f"Model: {k}")
        print(f"  Relaxed Accuracy: {v['acc_relaxed']:.2f}% (Count: {v['count']})")

if __name__ == "__main__":
    main()