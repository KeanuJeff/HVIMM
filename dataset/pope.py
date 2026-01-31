import os
from torch.utils.data import Dataset
import datasets
from PIL import Image
import random # <--- 新增: 引入 random 模組

class POPEDataset(Dataset):
    def __init__(self, dataset_id="lmms-lab/POPE", split='test', 
                 target_category=None, num_samples_per_category=500):
        
        print(f"Loading POPE dataset: {dataset_id} (Split: {split}, Category: {target_category})")
        
        self.dataset = datasets.load_dataset(dataset_id, split=split)
        #self.dataset = self.dataset.shuffle(seed=42)
        
        initial_count = len(self.dataset)
        
        if target_category:
            self.dataset = self.dataset.filter(
                lambda x: x['category'] == target_category
            )
            count_after_filter = len(self.dataset)
            
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
            "category": sample["category"] 
        }
