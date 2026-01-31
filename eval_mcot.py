import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import yaml
import torch
import gc
import json
import re
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.mcot import M3CoTDataset
from dataset.utils import load_model_and_processor, collate_fn

def mcot_score(pred, refs):
    if not pred: return 0.0
    pred = pred.strip().upper()
    refs = [a.strip().upper() for a in refs]
    
    return 1.0 if pred in refs else 0.0

def parse_final_answer_mc(full_text):
    match = re.search(r"Final Answer:\s*([A-D])", full_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"Answer:\s*([A-D])", full_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    match = re.search(r"Option:\s*([A-D])", full_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    matches = re.findall(r"\b([A-D])\b", full_text)
    if matches:
        return matches[-1].upper()

    return ""

def format_mc_prompt_cot(question, choices_list):
    prompt = "Answer the following multiple-choice question. First, provide your answer and then conclude with the correct option letter in the format 'Final Answer: [Letter]'.\n\n"
    prompt += "--- Example Start ---\n"
    prompt += "Question: some example question?\n"
    prompt += "Options:\n"
    prompt += "A. Option A\n"
    prompt += "B. Option B\n"
    prompt += "C. Option C\n"
    prompt += "Answer: After analyzing the question, the correct option is B.\n"
    prompt += "Final Answer: B.\n"
    prompt += "--- Example End ---\n\n"
    prompt += "--- Task Start ---\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Options:\n"
    for i, choice in enumerate(choices_list):
        label = chr(65 + i)
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt

def generate_answer_mc(proc, m, mtype, images, questions, choices_batch, **kwargs):
    batch_size = len(questions)
    
    if mtype == "gemma3":
        messages_list = []
        batch_images = [] 

        for i in range(batch_size):
            prompt_text = format_mc_prompt_cot(questions[i], choices_batch[i])
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[i]}, 
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            messages_list.append(messages)
            batch_images.append(images[i]) 

        texts = [
            proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]
        
        inputs = proc(
            text=texts,
            images=batch_images,
            padding=True,
            return_tensors="pt"
        ).to(m.device)

        kwargs.setdefault('max_new_tokens', 200)
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = proc.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        return [parse_final_answer_mc(ans) for ans in output_text]

    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_mc_prompt_cot(questions[i], choices_batch[i])
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": images[i]},
                        {"type": "text", "text": VQA_PROMPT} 
                    ]
                }
            ]
            messages_list.append(messages)

        text_prompts = [
            proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]
        
        inputs = proc(
            text=text_prompts, images=images, return_tensors="pt", padding=True
        ).to(m.device)
        
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        with torch.no_grad():
            res = m.generate(**inputs, **kwargs)
        
        responses = proc.batch_decode(res, skip_special_tokens=True)
        
        final_answers_full = []
        split_tag = "assistant\n"
        for response in responses:
            if split_tag in response:
                answer = response.split(split_tag)[-1].strip()
            else:
                answer = response.strip()
            final_answers_full.append(answer)

        return [parse_final_answer_mc(ans) for ans in final_answers_full]

    if mtype == "llava" or mtype == "dvit_llava" or mtype == "llava-next":
        prompt_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_mc_prompt_cot(questions[i], choices_batch[i])
            
            if mtype == "llava-next":
                prompt_text = f"[INST] <image>\n{VQA_PROMPT} [/INST]"
            else:
                prompt_text = f"USER: <image>\n{VQA_PROMPT} ASSISTANT:"
                
            prompt_list.append(prompt_text)
        
        inputs = proc(
            images=images, text=prompt_list, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(m.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]
        
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        
        with torch.no_grad():
            outputs = m.generate(**inputs, **kwargs)
        
        new_tokens = outputs[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)

        return [parse_final_answer_mc(ans) for ans in answers_full]

    if mtype == "instructblip":
        prompt_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_mc_prompt_cot(questions[i], choices_batch[i])
            prompt_list.append(VQA_PROMPT)

        inp = proc(
            images=images,
            text=prompt_list,
            return_tensors="pt", 
            padding=True
        ).to(m.device)
        input_len = inp['input_ids'].shape[1]
        
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        with torch.no_grad():
            out = m.generate(**inp, **kwargs)
            
        new_tokens = out[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)
        
        return [parse_final_answer_mc(ans) for ans in answers_full]

    if mtype == "qwen3-vl":
        messages_list = []
        
        for i in range(batch_size):
            prompt_text = format_mc_prompt_cot(questions[i], choices_batch[i])
            
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[i]},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            messages_list.append(msg)

        texts = [
            proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]

        inputs = proc(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(m.device)

        kwargs.setdefault('max_new_tokens', 200)
        
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = proc.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        return [parse_final_answer_mc(ans) for ans in output_text]
    if mtype == "structural_llava":
        answers = []
        for i in range(batch_size):
            q_text = format_mc_prompt_cot(questions[i], choices_batch[i])
            
            try:
                raw_response = m.generate_answer(images[i], q_text)
                
                parsed = parse_final_answer_mc(raw_response)
                
                if not parsed:
                    candidates = re.findall(r"\b([A-D])\b", raw_response.upper())
                    parsed = candidates[-1] if candidates else ""
                
                answers.append(parsed)
                
            except Exception as e:
                print(f"[Error] Sample {i}: {e}")
                answers.append("")
                
        return answers
        
    return [""] * batch_size

def main():
    config_dir = "configs"
    config_filename = 'mcot.yaml' 
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    ds = M3CoTDataset(
        dataset_id=cfg["dataset_id"],
        split="validation", 
        num_samples=cfg["num_val_samples"]
    )
    
    batch_size = cfg.get('batch_size', 1)
    print(f"Using Batch Size: {batch_size}")
    
    ds_loader = DataLoader(
        ds, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_fn 
    )
    
    all_results = {}
    
    for mdl in cfg["models"]:
        name = mdl["name"]
        print(f"\n--- Evaluating Model: {name} ---")
        
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        torch.compile(mdl_obj, mode="reduce-overhead")
        mdl_obj.eval()
        scores, preds = [], []
        
        generate_params = {
            "max_new_tokens": cfg.get("max_new_tokens", 400),
            "do_sample": False,
            "min_new_tokens": 1,
            "num_beams": 1
        }

        for batch_data in tqdm(ds_loader, desc=name):
            
            images = batch_data["image"]
            questions = batch_data["question"]
            all_answers = batch_data["answers"]
            question_ids = batch_data["question_id"]
            choices_batch = batch_data["choices"]

            ans_list = generate_answer_mc(
                proc, mdl_obj, mtype, 
                images, 
                questions, 
                choices_batch,
                **generate_params
            )
            
            for i in range(len(ans_list)):
                ans = ans_list[i]
                q = questions[i]
                refs = all_answers[i]
                qid = question_ids[i]
                
                if not refs or not all(r.strip() for r in refs):
                    continue
                
                score = mcot_score(ans, refs) 
                scores.append(score)
                preds.append({"qid": qid, "question": q, "choices": choices_batch[i], "pred": ans, "refs": refs, "score": score})
            del images, questions, all_answers, ans_list
            if 'batch_data' in locals():
                del batch_data
            
            gc.collect() 
            torch.cuda.empty_cache()

        if scores:
            mean_acc = (sum(scores) / len(scores)) * 100
        else:
            mean_acc = 0.0
            
        all_results[name] = {"acc": mean_acc, "predictions": preds}
        
        output_file = f"results_mcot_{name.replace(' ', '_').replace('/', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results[name], f, indent=4, ensure_ascii=False)
        print(f"Results for {name} saved to {output_file}")

        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Final M3CoT Evaluation Results ===")
    for k, v in all_results.items():
        print(f"{k}: {v['acc']:.2f}%")

if __name__ == "__main__":
    main()
