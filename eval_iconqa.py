import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import yaml
import torch
import gc
import json
import re
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.iconqa import IconQADataset
from dataset.utils import load_model_and_processor, collate_fn

def score_mc_acc(pred, refs):
    if not pred: return 0.0
    pred = pred.strip().upper()
    refs = [a.strip().upper() for a in refs]
    return 1.0 if pred in refs else 0.0

def parse_final_answer_mc(full_text):
    match = re.search(r"Final Answer:\s*([A-Z])", full_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"Answer:\s*([A-Z])", full_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"Option:\s*([A-Z])", full_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", full_text)
    if matches: return matches[-1].upper()
    return ""

def format_mc_prompt_cot(question, choices_list):
    prompt = "Answer the following multiple-choice question either directly or with some reasoning. Conclude the final answer with the correct option letter in the format 'Final Answer: [Letter]'.\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Options:\n"
    for i, choice in enumerate(choices_list):
        label = chr(65 + i)
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer: "
    return prompt

def generate_answer_mc(proc, m, mtype, images, questions, choices_batch, **kwargs):
    batch_size = len(questions)
    
    if mtype == "gemma3":
        messages_list = []
        batch_images = []
        for i in range(batch_size):
            VQA_PROMPT = format_mc_prompt_cot(questions[i], choices_batch[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
            batch_images.append(images[i])
            
        texts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=texts, images=batch_images, return_tensors="pt", padding=True).to(m.device)
        
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = proc.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [parse_final_answer_mc(ans) for ans in output_text]

    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_mc_prompt_cot(questions[i], choices_batch[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
        text_prompts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=text_prompts, images=images, return_tensors="pt", padding=True).to(m.device)
        
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        with torch.no_grad():
            res = m.generate(**inputs, **kwargs)
        responses = proc.batch_decode(res, skip_special_tokens=True)
        final_answers_full = [resp.split("assistant\n")[-1].strip() for resp in responses]
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
            
        inputs = proc(images=images, text=prompt_list, return_tensors="pt", padding=True)
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
            prompt_list.append(format_mc_prompt_cot(questions[i], choices_batch[i]))
        inputs = proc(images=images, text=prompt_list, return_tensors="pt", padding=True).to(m.device)
        input_len = inputs['input_ids'].shape[1]
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        
        with torch.no_grad():
            out = m.generate(**inputs, **kwargs)
        new_tokens = out[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)
        return [parse_final_answer_mc(ans) for ans in answers_full]

    if mtype == "qwen3-vl":
        messages_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_mc_prompt_cot(questions[i], choices_batch[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
            
        texts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=texts, images=images, return_tensors="pt", padding=True).to(m.device)
        
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = proc.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [parse_final_answer_mc(ans) for ans in output_text]
    if mtype == "structural_llava":
        answers = []
        for i in range(batch_size):
            q_text = format_mc_prompt_cot(questions[i], choices_batch[i])
            
            try:
                raw_ans = m.generate_answer(images[i], q_text)
                
                parsed = parse_final_answer_mc(raw_ans)
                
                if not parsed:
                    matches = re.findall(r"\b([A-E])\b", raw_ans.upper())
                    parsed = matches[-1] if matches else ""
                
                answers.append(parsed)
            except Exception as e:
                print(f"[MCQ Error] Sample {i}: {e}")
                answers.append("")
        return answers

    return [""] * batch_size

def score_vqa(pred, refs):
    if not pred: return 0.0
    pred = pred.strip().lower()
    refs = [a.strip().lower() for a in refs]
    return 1.0 if pred in refs else 0.0

def parse_final_answer_vqa(full_text):
    try:
        split_tag = "Final Answer:"
        parts = full_text.split(split_tag)
        if len(parts) < 2:
             return full_text.strip().split()[-1]
             
        final_part = parts[-1]
        
        final_answer = final_part.strip().split('\n')[0].strip().split(' ')[0].strip()
        
        final_answer = final_answer.rstrip('.,')
        
        return final_answer
    except Exception:
        return ""

def format_vqa_prompt(question):
    return f"Answer the following question either directly or with reasoning. Give the final answer in a single word like 'Final Answer: [single word]'.\n\nQuestion: {question}\nAnswer:"

def generate_answer_vqa(proc, m, mtype, images, questions, **kwargs):
    batch_size = len(questions)
    
    if mtype == "gemma3":
        messages_list = []
        batch_images = []
        for i in range(batch_size):
            VQA_PROMPT = format_vqa_prompt(questions[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
            batch_images.append(images[i])
            
        texts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=texts, images=batch_images, return_tensors="pt", padding=True).to(m.device)
        
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = proc.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [parse_final_answer_vqa(ans) for ans in output_text]

    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_vqa_prompt(questions[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
        text_prompts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=text_prompts, images=images, return_tensors="pt", padding=True).to(m.device)
        
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        with torch.no_grad():
            res = m.generate(**inputs, **kwargs)
        responses = proc.batch_decode(res, skip_special_tokens=True)
        final_answers_full = [resp.split("assistant\n")[-1].strip() for resp in responses]
        return [parse_final_answer_vqa(ans) for ans in final_answers_full]

    if mtype == "llava" or mtype == "dvit_llava" or mtype == "llava-next":
        prompt_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_vqa_prompt(questions[i])
            if mtype == "llava-next":
                prompt_text = f"[INST] <image>\n{VQA_PROMPT} [/INST]"
            else:
                prompt_text = f"USER: <image>\n{VQA_PROMPT} ASSISTANT:"
            prompt_list.append(prompt_text)
            
        inputs = proc(images=images, text=prompt_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(m.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        
        with torch.no_grad():
            outputs = m.generate(**inputs, **kwargs)
        new_tokens = outputs[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)
        return [parse_final_answer_vqa(ans) for ans in answers_full]

    if mtype == "instructblip":
        prompt_list = []
        for i in range(batch_size):
            prompt_list.append(format_vqa_prompt(questions[i]))
        inputs = proc(images=images, text=prompt_list, return_tensors="pt", padding=True).to(m.device)
        input_len = inputs['input_ids'].shape[1]
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        
        with torch.no_grad():
            out = m.generate(**inputs, **kwargs)
        new_tokens = out[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)
        return [parse_final_answer_vqa(ans) for ans in answers_full]

    if mtype == "qwen3-vl":
        messages_list = []
        for i in range(batch_size):
            VQA_PROMPT = format_vqa_prompt(questions[i])
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
            messages_list.append(messages)
        
        texts = [proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
        inputs = proc(text=texts, images=images, return_tensors="pt", padding=True).to(m.device)
        
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = proc.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [parse_final_answer_vqa(ans) for ans in output_text]
    if mtype == "structural_llava":
        answers = []
        for i in range(batch_size):
            q_text = format_vqa_prompt(questions[i])
            
            try:
                raw_ans = m.generate_answer(images[i], q_text)
                
                parsed = parse_final_answer_vqa(raw_ans)
                
                if not parsed:
                    parsed = raw_ans.strip()
                
                answers.append(parsed)
            except Exception as e:
                print(f"[VQA Error] Sample {i}: {e}")
                answers.append("")
        return answers

    return [""] * batch_size

def main():
    config_dir = "configs"
    config_filename = 'iconqa.yaml'
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    ds = IconQADataset(
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
        mdl_obj.eval()
        
        scores_mc, scores_vqa, preds = [], [], []
        
        generate_params_mc = {
            "max_new_tokens": cfg.get("max_new_tokens_mc", 200), 
            "do_sample": False,
            "min_new_tokens": 1,
            "num_beams": 1
        }
        generate_params_vqa = {
            "max_new_tokens": cfg.get("max_new_tokens_vqa", 100), 
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
            question_types = batch_data["question_type"] 

            mc_indices = [i for i, t in enumerate(question_types) if t == 'multiple-choice']
            vqa_indices = [i for i, t in enumerate(question_types) if t == 'open-ended']
            
            results = [None] * len(questions)

            if mc_indices:
                mc_images = [images[i] for i in mc_indices]
                mc_questions = [questions[i] for i in mc_indices]
                mc_choices = [choices_batch[i] for i in mc_indices]
                
                mc_ans_list = generate_answer_mc(
                    proc, mdl_obj, mtype, 
                    mc_images, mc_questions, mc_choices, 
                    **generate_params_mc
                )
                for i, ans in enumerate(mc_ans_list):
                    results[mc_indices[i]] = (ans, 'mc')

            if vqa_indices:
                vqa_images = [images[i] for i in vqa_indices]
                vqa_questions = [questions[i] for i in vqa_indices]
                
                vqa_ans_list = generate_answer_vqa(
                    proc, mdl_obj, mtype, 
                    vqa_images, vqa_questions, 
                    **generate_params_vqa
                )
                for i, ans in enumerate(vqa_ans_list):
                    results[vqa_indices[i]] = (ans, 'vqa')

            for i in range(len(results)):
                if results[i] is None: continue
                
                ans, q_type = results[i]
                refs = all_answers[i]
                
                if "INVALID" in refs or not refs or not all(r.strip() for r in refs):
                    continue
                
                score = 0.0
                if q_type == 'mc':
                    score = score_mc_acc(ans, refs) 
                    scores_mc.append(score)
                elif q_type == 'vqa':
                    score = score_vqa(ans, refs)
                    scores_vqa.append(score)
                
                preds.append({
                    "qid": question_ids[i], 
                    "q_type": q_type,
                    "question": questions[i], 
                    "choices": choices_batch[i], 
                    "pred": ans, 
                    "refs": refs, 
                    "score": score
                })
            del images, questions, all_answers
            if 'batch_data' in locals():
                del batch_data
            
            gc.collect() 
            torch.cuda.empty_cache()

        mean_acc_mc = (sum(scores_mc) / len(scores_mc)) * 100 if scores_mc else 0.0
        mean_acc_vqa = (sum(scores_vqa) / len(scores_vqa)) * 100 if scores_vqa else 0.0
        total_scores = scores_mc + scores_vqa
        mean_acc_overall = (sum(total_scores) / len(total_scores)) * 100 if total_scores else 0.0
        
        all_results[name] = {
            "acc_overall": mean_acc_overall,
            "acc_mc": mean_acc_mc,
            "acc_vqa": mean_acc_vqa,
            "counts": {"mc": len(scores_mc), "vqa": len(scores_vqa)},
            "predictions": preds
        }
        
        output_file = f"results_iconqa_{name.replace(' ', '_').replace('/', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results[name], f, indent=4, ensure_ascii=False)
        print(f"Results for {name} saved to {output_file}")

        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Final ICON-QA Evaluation Results ===")
    for k, v in all_results.items():
        print(f"Model: {k}")
        print(f"  Overall ACC: {v['acc_overall']:.2f}%")
        print(f"  MCQ ACC:     {v['acc_mc']:.2f}% (Count: {v['counts']['mc']})")
        print(f"  VQA ACC:     {v['acc_vqa']:.2f}% (Count: {v['counts']['vqa']})")

if __name__ == "__main__":
    main()
