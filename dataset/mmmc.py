import os
from torch.utils.data import Dataset
import datasets
from PIL import Image

class MMMCDataset(Dataset):
    def __init__(self, dataset_id="ustc-zhangzm/MMMC", split='validation', num_samples=None):
        """
        從 Hugging Face Hub 載入 MMMC 資料集。
        
        任務: 模態衝突檢測 (是/非)
        
        1. 過濾: 移除 conflict_type == 'relation' 的樣本。
        2. 轉換: 
           - conflict_type 為 'null' -> 答案: 'No' (無衝突)
           - conflict_type 為 其他 -> 答案: 'Yes' (有衝突)
        """
        print(f"Loading MMMC dataset: {dataset_id} (Split: {split})")
        
        # 根據 split 載入 (MMMC 似乎沒有 'validation'，我們用 'test' 代替)
        # 注意：請根據 Hugging Face 上的實際可用 split 調整 (例如 'train' 或 'test')
        load_split = 'test' 
        if split == 'train':
            load_split = 'train'
            
        self.dataset = datasets.load_dataset(dataset_id, split=load_split)
        
        initial_count = len(self.dataset)
        print(f"Original dataset size ({split}): {initial_count}")
        
        # 【關鍵過濾】: 移除 conflict_type 為 'relation' 的樣本
        self.dataset = self.dataset.filter(
            lambda x: x['conflict_type'] != 'relation'
        )
        
        filtered_count = len(self.dataset)
        print(f"Filtered out {initial_count - filtered_count} 'relation' conflict samples.")
        print(f"Dataset size after filtering: {filtered_count}")
        
        if num_samples and num_samples > 0:
            print(f"Using {num_samples} samples from the filtered dataset.")
            self.dataset = self.dataset.select(range(1000, 1000 + min(num_samples, len(self.dataset))))
            #self.dataset = self.dataset.select(range(1000, 1000 + min(num_samples, len(self.dataset))))
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        image = sample["image"]
        
        # 假設 MMMC 中的文本欄位是 'sentence'
        # 如果出現 KeyError，請將 'sentence' 更改為 'text' 或 'question'
        original_text = sample["question"] 
        
        conflict_type = sample["conflict_type"]

        # 【關鍵轉換】: 建立 "Yes" / "No" 標籤
        # conflict_type 為 None 或 'null' 字符串
        if conflict_type is None or conflict_type == 'null':
            correct_label = "No" # 無衝突
        else:
            correct_label = "Yes" # 有衝突 (因為 'relation' 已被過濾)
        
        if image is None:
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # 返回評估所需的所有欄位
        return {
            "image": image,
            "question": original_text, # 我們將原始文本放在 'question' 欄位中傳遞
            "answers": [correct_label], # 傳遞 "Yes" 或 "No"
            "question_id": sample.get("id", f"mmmc-{idx}") # 使用 'id' 或索引作為 QID
        }