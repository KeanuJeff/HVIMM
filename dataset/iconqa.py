import os
from torch.utils.data import Dataset
import datasets
from PIL import Image
import ast

class IconQADataset(Dataset):
    def __init__(self, dataset_id="lmms-lab/ICON-QA", split='validation', num_samples=None):
        """
        從 Hugging Face Hub 載入 ICON-QA 資料集。
        - 關鍵過濾: 只保留 'multiple-choice' 和 'open-ended'。
        - 關鍵過濾: 移除 'choose_img'。
        - 關鍵轉換: 根據類型返回 'A'/'B' 或 '10'/'7'。
        """
        print(f"Loading ICON-QA dataset: {dataset_id} (Split: {split})")
        
        load_split = 'train'
        if split == 'validation':
            load_split = 'val'
        elif split == 'test':
            load_split = 'test'
            
        self.dataset = datasets.load_dataset(dataset_id, "default", split=load_split)
        
        initial_count = len(self.dataset)
        print(f"Original dataset size ({split}): {initial_count}")
        
        # 2. 【過濾 2】: 移除 ques_type 為 'choose_img' 的樣本
        self.dataset = self.dataset.filter(
            lambda x: x.get('ques_type') != 'choose_img'
        )
        
        final_filtered_count = len(self.dataset)
        removed_choose_img_count = initial_count - final_filtered_count
        
        print(f"Filtered out {removed_choose_img_count} 'choose_img' samples.")
        print(f"Final dataset size (MC + VQA): {final_filtered_count}")
        
        if num_samples and num_samples > 0:
            print(f"Using {num_samples} samples from the filtered dataset.")
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        image = sample["query_image"]
        question = sample["question"]
        q_type = sample["ques_type"] # 'multiple-choice' 或 'open-ended'
        
        correct_label = ""
        choices_list = []

        if q_type == 'choose_txt':
            # --- 處理多選題 ---
            raw_choices = sample["choices"]
            
            # 檢查它是否為字串，如果是，嘗試轉回 List
            if isinstance(raw_choices, str):
                # 情況 A: 像是 "['A', 'B', 'C']" 的字串格式
                if raw_choices.startswith("[") and raw_choices.endswith("]"):
                    try:
                        choices_list = ast.literal_eval(raw_choices)
                    except:
                        # 如果解析失敗，回退到簡單分割
                         choices_list = raw_choices.strip("[]").replace("'", "").replace('"', "").split(',')
                # 情況 B: 像是 "4,6,8,7,2" 的簡單逗號分隔
                else:
                    choices_list = raw_choices.split(',')
            else:
                # 如果它原本就是 list 就不用動
                choices_list = raw_choices

            # 清理一下空白 (預防 " 4" 這種情況)
            choices_list = [str(c).strip() for c in choices_list]
            answer_text = sample["answer"]
            try:
                answer_index = choices_list.index(answer_text)
                correct_label = chr(65 + answer_index) # A, B, C...
            except ValueError:
                correct_label = "INVALID" # 答案不在選項中
        
        elif q_type == 'fill_in_blank':
            # --- 處理開放式問題 (VQA) ---
            correct_label = sample["answer"] # e.g., '10', '7'
            choices_list = [] # 開放式問題沒有選項

        # 處理圖像
        if image is None:
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        elif image.mode != "RGB":
            image = image.convert("RGB")
        
        final_type_name = "unknown"
        if q_type == 'choose_txt':
            final_type_name = "multiple-choice"  # 對應 eval_iconqa.py 的邏輯
        elif q_type == 'fill_in_blank':
            final_type_name = "open-ended"       # 對應 eval_iconqa.py 的邏輯

        return {
            "image": image,
            "question": question,
            "choices": choices_list,
            "answers": [correct_label], # 傳遞標籤 ('A') 或文本答案 ('10')
            "question_id": sample["question_id"],
            "question_type": final_type_name # <--- 關鍵: 傳遞問題類型
        }