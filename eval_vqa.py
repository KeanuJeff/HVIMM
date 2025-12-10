import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import yaml
import torch
import gc
import json # <--- 變動 1: 新增 json 模組
from torch.utils.data import DataLoader
from dataset.vqa import VQADataset
from tqdm import tqdm
from dataset.utils import load_model_and_processor, generate_answer, collate_fn


def vqa_score(pred, refs):
    pred = pred.strip().lower()
    # 確保 refs 是一個答案列表
    refs = [a.strip().lower() for a in refs]
    # VQA 評分標準：達到 3 個參考答案的平均分數
    agree = sum(r == pred for r in refs)
    return min(1.0, agree / 3.0)


def main():
    config_dir = "configs"
    config_filename = 'vqa.yaml'
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    # 初始化數據集
    ds = VQADataset(cfg["val_img_dir"], cfg["ques_json"], cfg["anns_json"], cfg["num_samples"])
    
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
        print(f"\n--- Evaluating Model: {name} ---")
        
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        mdl_obj.eval()
        scores, preds = [], []
        
        generate_params = {
            "max_new_tokens": cfg.get("max_new_tokens", 40),
            "do_sample": False,
            "min_new_tokens": 1,
            "num_beams": 1
        }

        for batch_data in tqdm(ds_loader, desc=name):
            
            images = batch_data["image"]
            questions = batch_data["question"]
            all_answers = batch_data["answers"]
            question_ids = batch_data["question_id"]

            ans_list = generate_answer(
                proc, mdl_obj, mtype, 
                images, 
                questions, 
                **generate_params
            )
            
            for i in range(len(ans_list)):
                ans = ans_list[i]
                q = questions[i]
                refs = all_answers[i]
                qid = question_ids[i]
                
                if not refs or not all(r.strip() for r in refs):
                    print(f"Warning: Empty or invalid reference answers for qid {qid}, skipping.")
                    continue

                score = vqa_score(ans, refs)
                scores.append(score)
                preds.append({"qid": qid, "pred": ans, "refs": refs, "score": score})
                
            del images, questions, all_answers, ans_list
            if 'batch_data' in locals():
                del batch_data
            
            # 這是關鍵：如果 Python 物件沒死，PyTorch 就不能釋放 GPU 記憶體
            gc.collect() 
            torch.cuda.empty_cache()

        # (計分邏輯保持不變)
        if scores:
            mean_acc = sum(scores) / len(scores)
        else:
            mean_acc = 0.0
            
        all_results[name] = {"acc": mean_acc, "predictions": preds}
        
        # <--- 變動 2: 新增 JSON 儲存邏輯 --->
        output_file = f"results_vqa_{name.replace(' ', '_').replace('/', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            # 確保使用 ensure_ascii=False 支援中文等非 ASCII 字符
            json.dump(all_results[name], f, indent=4, ensure_ascii=False)
        print(f"Results for {name} saved to {output_file}")
        # <--- 結束新增 --->

        # 清理
        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache() # 確保 VRAM 被釋放

    print("\nFinal Results:")
    for k, v in all_results.items():
        print(f"{k}: {v['acc']:.3f}")


if __name__ == "__main__":
    main()