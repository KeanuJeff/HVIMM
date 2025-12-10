import os
from torch.utils.data import Dataset
import datasets
from PIL import Image
import random # <--- 新增: 引入 random 模組

class POPEDataset(Dataset):
    # 修改 __init__ 接受 category_filter 和 num_samples_per_category
    def __init__(self, dataset_id="lmms-lab/POPE", split='test', 
                 target_category=None, num_samples_per_category=500):
        
        print(f"Loading POPE dataset: {dataset_id} (Split: {split}, Category: {target_category})")
        
        self.dataset = datasets.load_dataset(dataset_id, split=split)
        
        initial_count = len(self.dataset)
        
        if target_category:
            # 【關鍵過濾 1】: 只保留目標類別的樣本
            self.dataset = self.dataset.filter(
                lambda x: x['category'] == target_category
            )
            count_after_filter = len(self.dataset)
            
            # 【關鍵過濾 2】: 隨機採樣 1000 個樣本
            if count_after_filter > num_samples_per_category:
                self.dataset = self.dataset.select(range(min(num_samples_per_category, len(self.dataset))))
                #self.dataset = self.dataset.select(range(1000, min(1000 + num_samples_per_category, len(self.dataset))))
            final_count = len(self.dataset)
            print(f"Sampled {final_count} samples from category '{target_category}'.")
        else:
            final_count = initial_count
            print(f"Loaded full dataset ({final_count} samples).")
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        question = sample["question"] 
        
        # 【答案處理】: 將 "yes" / "no" 轉為 "Yes" / "No"
        answer_text = sample["answer"].strip().capitalize()
        correct_label = answer_text
        
        if image is None:
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        elif image.mode != "RGB":
            image = image.convert("RGB")

        return {
            "image": image,
            "question": question,         
            "answers": [correct_label],  
            "question_id": sample.get("question_id", f"pope-{idx}"),
            # 返回類別名稱，以便在 JSON 中記錄
            "category": sample["category"] 
        }