# datasets/mcot.py
import os
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image

class M3CoTDataset(Dataset):
    def __init__(self, dataset_id, split='train', num_samples=None):
        """
        從 Hugging Face Hub 載入 M3CoT 資料集 (多選題格式)。
        """
        print(f"Loading M3CoT dataset: {dataset_id} (Split: {split})")
        
        # 載入資料集 (如果已下載，會自動使用快取)
        self.dataset = load_dataset(dataset_id, split=split)
        
        if num_samples and num_samples > 0:
            print(f"Using {num_samples} samples from the dataset.")
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        image = sample["image"]
        question = sample["question"]
        
        # 獲取選項列表和正確答案標籤
        choices_list = sample["choices"]     # e.g., ["Choice 1 text", "Choice 2 text"]
        correct_label = sample["answer"]   # e.g., "C"
        
        # 確保影像是 RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 返回多選題所需的所有欄位
        return {
            "image": image,
            "question": question,
            "choices": choices_list,         # 傳遞選項列表
            "answers": [correct_label],    # 傳遞正確答案標籤 (列表格式以兼容 collate_fn)
            "question_id": f"mcot_{sample['id']}" 
        }