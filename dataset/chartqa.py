# dataset/chartqa.py
import datasets
from torch.utils.data import Dataset
from PIL import Image

class ChartQADataset(Dataset):
    def __init__(self, dataset_id="HuggingFaceM4/the_cauldron", subset="chartqa", split='train', num_samples=1000):
        """
        從 Hugging Face Hub 載入 The Cauldron (ChartQA subset)。
        結構: {'images': [PIL.Image], 'texts': [{'user': 'Q', 'assistant': 'A', ...}, ...]}
        """
        print(f"Loading The Cauldron dataset: {dataset_id} (Subset: {subset}, Split: {split})")
        
        # 1. 載入資料集 (The Cauldron 只有 train split)
        self.dataset = datasets.load_dataset(dataset_id, subset, split=split)
        
        # 2. 截取前 N 筆原始資料 (依照你的需求 1000)
        if num_samples and num_samples > 0:
            print(f"Selecting first {num_samples} source samples.")
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))

        # 3. 扁平化處理 (Flattening)
        # 一筆原始資料可能包含多個 QA 對，我們需要把它們拆開變成獨立的測試樣本
        self.flat_samples = []
        
        print("Processing and flattening samples...")
        for row in self.dataset:
            # ChartQA 在 Cauldron 中通常每一列對應一張圖
            # images 欄位是 list，通常取第一個
            if len(row['images']) > 0:
                image = row['images'][0] 
            else:
                # 預防萬一沒有圖，給一張全白圖
                image = Image.new("RGB", (224, 224), (255, 255, 255))

            # 遍歷 texts 中的每一輪對話
            # 格式範例: [{'user': 'Q string', 'assistant': 'A string', 'source': 'ChartQA'}]
            for turn in row['texts']:
                question = turn.get('user', '')
                answer = turn.get('assistant', '')
                
                if question and answer:
                    self.flat_samples.append({
                        "image": image,
                        "question": question,
                        "answer": answer,
                        # 這裡沒有 ID，我們可以用 index 當作 ID
                        "question_id": f"chartqa_{len(self.flat_samples)}"
                    })
        
        print(f"Total QA pairs generated: {len(self.flat_samples)}")

    def __len__(self):
        return len(self.flat_samples)

    def __getitem__(self, idx):
        sample = self.flat_samples[idx]
        
        image = sample["image"]
        question = sample["question"]
        answer = sample["answer"]
        
        # 確保圖片是 RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return {
            "image": image,
            "question": question,
            "choices": [],       # ChartQA 通常是開放式問題，沒有選項
            "answers": [answer], # 轉成 list 配合 utils.py 格式
            "question_id": sample["question_id"],
            "question_type": "open-ended" # 標記為 VQA 類型
        }