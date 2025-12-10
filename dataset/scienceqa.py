import os
from torch.utils.data import Dataset
import datasets
from PIL import Image

class ScienceQADataset(Dataset):
    def __init__(self, dataset_id, split='train', num_samples=None):
        """
        從 Hugging Face Hub 載入 ScienceQA 資料集。
        - 過濾 'social science'
        - 將答案索引 (0,1,2) 轉換為標籤 (A,B,C)
        - 處理空圖像
        """
        print(f"Loading ScienceQA dataset: {dataset_id} (Split: {split})")
        
        # 根據 split 載入
        load_split = 'train'
        if split == 'validation':
            load_split = 'validation'
        elif split == 'test':
            load_split = 'test'
            
        self.dataset = datasets.load_dataset(dataset_id, split=load_split)
        
        # 【關鍵過濾】: 移除 subject 為 'social science' 的樣本
        initial_count = len(self.dataset)
        print(f"Original dataset size ({split}): {initial_count}")
        self.dataset = self.dataset.filter(
            lambda x: x['subject'] != 'social science'
        )
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
        
        # 【關鍵轉換】: ScienceQA 的答案是索引 (int)
        answer_index = sample["answer"] # e.g., 0, 1, 2
        correct_label = chr(65 + answer_index) # 轉換為 'A', 'B', 'C'
        
        # 【關鍵】: ScienceQA 的 CoT 在 'solution' 欄位
        rationale = sample["solution"]
        
        # 【關鍵】: 處理 'image' 為 None 的情況 (純文字題)
        if image is None:
            # 建立一個空白的白色圖像，因為 LLaVA/Qwen 需要圖像輸入
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # 返回訓練和評估所需的所有欄位
        return {
            "image": image,
            "question": question,
            "choices": choices_list,
            "answers": [correct_label], # 傳遞正確答案標籤
            "rationale": rationale,     # 傳遞真實的 CoT 解釋
        }