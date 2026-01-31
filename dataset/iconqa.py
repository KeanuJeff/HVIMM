import os
from torch.utils.data import Dataset
import datasets
from PIL import Image
import ast

class IconQADataset(Dataset):
    def __init__(self, dataset_id="lmms-lab/ICON-QA", split='validation', num_samples=None):
        
        print(f"Loading ICON-QA dataset: {dataset_id} (Split: {split})")
        
        load_split = 'train'
        if split == 'validation':
            load_split = 'val'
        elif split == 'test':
            load_split = 'test'
            
        self.dataset = datasets.load_dataset(dataset_id, "default", split=load_split)
        
        initial_count = len(self.dataset)
        print(f"Original dataset size ({split}): {initial_count}")
        
        self.dataset = self.dataset.filter(
            lambda x: x.get('ques_type') != 'choose_img'
        )

        #self.dataset = self.dataset.shuffle(seed=42)
        
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
            raw_choices = sample["choices"]
            
            if isinstance(raw_choices, str):
                if raw_choices.startswith("[") and raw_choices.endswith("]"):
                    try:
                        choices_list = ast.literal_eval(raw_choices)
                    except:
                         choices_list = raw_choices.strip("[]").replace("'", "").replace('"', "").split(',')
 
                else:
                    choices_list = raw_choices.split(',')
            else:
                choices_list = raw_choices

            choices_list = [str(c).strip() for c in choices_list]
            answer_text = sample["answer"]
            try:
                answer_index = choices_list.index(answer_text)
                correct_label = chr(65 + answer_index) # A, B, C...
            except ValueError:
                correct_label = "INVALID" 
        
        elif q_type == 'fill_in_blank':
            correct_label = sample["answer"] # e.g., '10', '7'
            choices_list = [] 

        if image is None:
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        elif image.mode != "RGB":
            image = image.convert("RGB")
        
        final_type_name = "unknown"
        if q_type == 'choose_txt':
            final_type_name = "multiple-choice" 
        elif q_type == 'fill_in_blank':
            final_type_name = "open-ended"  

        return {
            "image": image,
            "question": question,
            "choices": choices_list,
            "answers": [correct_label],
            "question_id": sample["question_id"],
            "question_type": final_type_name
        }
