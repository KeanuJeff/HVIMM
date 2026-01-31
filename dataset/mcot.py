# datasets/mcot.py
import os
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image

class M3CoTDataset(Dataset):
    def __init__(self, dataset_id, split='train', num_samples=None):
        print(f"Loading M3CoT dataset: {dataset_id} (Split: {split})")
        
        self.dataset = load_dataset(dataset_id, split=split)
        #self.dataset = self.dataset.shuffle(seed=42)
        
        if num_samples and num_samples > 0:
            print(f"Using {num_samples} samples from the dataset.")
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        image = sample["image"]
        question = sample["question"]
        
        choices_list = sample["choices"]     # e.g., ["Choice 1 text", "Choice 2 text"]
        correct_label = sample["answer"]   # e.g., "C"
        
        if image.mode != "RGB":
            image = image.convert("RGB")

        return {
            "image": image,
            "question": question,
            "choices": choices_list, 
            "answers": [correct_label], 
            "question_id": f"mcot_{sample['id']}" 
        }
