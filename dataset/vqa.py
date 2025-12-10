import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VQADataset(Dataset):
    def __init__(self, img_dir, ques_json, anns_json, num_samples=None):
        with open(ques_json, 'r') as f:
            questions = json.load(f)['questions']
        if num_samples:
            questions = questions[:num_samples]
        with open(anns_json, 'r') as f:
            ann_data = json.load(f)
        self.questions = questions
        # 依 question_id 索引對答案
        self.ref_ans = {a["question_id"]: [ans["answer"] for ans in a["answers"]] for a in ann_data["annotations"]}
        self.img_dir = img_dir
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.questions)

    def get_image_path(self, image_id):
        return os.path.join(self.img_dir, f"COCO_val2014_{image_id:012d}.jpg")

    def __getitem__(self, idx):
        q = self.questions[idx]
        image_fp = self.get_image_path(q['image_id'])
        image = Image.open(image_fp).convert("RGB")
        question, qid = q['question'], q['question_id']
        answers = self.ref_ans.get(qid, [])
        return {
            "image": image,
            "question": question,
            "answers": answers,
            "question_id": qid
        }