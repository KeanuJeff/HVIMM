import os
from torch.utils.data import Dataset
import datasets
from PIL import Image

class ScienceQADataset(Dataset):
    def __init__(self, dataset_id, split='train', num_samples=None):
        print(f"Loading ScienceQA dataset: {dataset_id} (Split: {split})")
        
        load_split = 'train'
        if split == 'validation':
            load_split = 'validation'
        elif split == 'test':
            load_split = 'test'
            
        self.dataset = datasets.load_dataset(dataset_id, split=load_split)
        
        initial_count = len(self.dataset)
        print(f"Original dataset size ({split}): {initial_count}")
        self.dataset = self.dataset.filter(
            lambda x: x['subject'] != 'social science'
        )
        #self.dataset = self.dataset.shuffle(seed=42)
        filtered_count = len(self.dataset)
        print(f"Filtered out {initial_count - filtered_count} 'social science' samples.")
        print(f"Final dataset size: {filtered_count}")
        
        if num_samples and num_samples > 0:
            print(f"Using {num_samples} samples from the filtered dataset.")
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        image = sample["image"]
        question = sample["question"]
        choices_list = sample["choices"]
        
        answer_index = sample["answer"] # e.g., 0, 1, 2
        correct_label = chr(65 + answer_index) 
        rationale = sample["solution"]
        
        if image is None:
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        elif image.mode != "RGB":
            image = image.convert("RGB")

        return {
            "image": image,
            "question": question,
            "choices": choices_list,
            "answers": [correct_label], 
            "rationale": rationale,
        }
