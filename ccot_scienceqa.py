import os
import yaml
import torch
import gc
import re
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# --- 匯入 Dataset ---
# 假設 dataset/scienceqa.py 存在
from dataset.scienceqa import ScienceQADataset
from dataset.utils import load_model_and_processor, collate_fn 

# =================================================================
# 1. CCoT 提示詞模板 (針對 ScienceQA 優化)
# =================================================================

CCOT_PROMPTS = {
    # -----------------------------------------------------------------
    # Qwen-VL
    # -----------------------------------------------------------------
    "qwen-vl": {
        'stage_1_system': "You are an expert scientific image analyzer.",
        # Stage 1: 針對科學圖片生成 SG
        'stage_1_user': """For the provided science image (diagram, graph, or illustration) and the question, generate a Scene Graph in JSON format.
Include:
1. Key scientific elements (e.g., cell parts, force vectors, graph axes).
2. Attributes (values, labels, colors, directions).
3. Relationships (part-of, interacts-with, greater-than).

Question: {question}
Scene Graph:""", 

        'stage_2_system': "You are a science tutor answering questions.",
        
        # Stage 2 (Visual): 有圖 + SG
        'stage_2_user_visual': """Use the image and the following Scene Graph to reason and answer the multiple-choice question.
Scene Graph: {scene_graph}

Context: {context}
Question: {question}
Options:
{options_str}

Reason step-by-step. Finally, provide the answer as a single letter (A, B, C, etc.) in the format 'Final Answer: [Letter]'.
Answer:""",

        # Stage 2 (Text-only): 無圖，退回純文字 CoT
        'stage_2_user_text': """Answer the following multiple-choice science question based on the context.

Context: {context}
Question: {question}
Options:
{options_str}

Reason step-by-step. Finally, provide the answer as a single letter (A, B, C, etc.) in the format 'Final Answer: [Letter]'.
Answer:"""
    },
    
    # -----------------------------------------------------------------
    # LLaVA / LLaVA-NeXT
    # -----------------------------------------------------------------
    "llava": {
        'stage_1_prompt': """[INST] <image>
For the provided science image and the question, generate a Scene Graph in JSON format.
Include:
1. Key scientific elements (e.g., cell parts, force vectors, graph axes).
2. Attributes (values, labels, colors, directions).
3. Relationships (part-of, interacts-with, greater-than).

Question: {question}
ASSISTANT: Scene Graph:[/INST]""",

        'stage_2_prompt_visual': """[INST] <image>
Use the image and the following Scene Graph to reason and answer the multiple-choice question.
Scene Graph: {scene_graph}

Context: {context}
Question: {question}
Options:
{options_str}

Reason step-by-step. Finally, provide the answer as a single letter (A, B, C, etc.) in the format 'Final Answer: [Letter]'.
ASSISTANT: Answer:[/INST]""",

        'stage_2_prompt_text': """[INST] 
Answer the following multiple-choice science question based on the context.

Context: {context}
Question: {question}
Options:
{options_str}

Reason step-by-step. Finally, provide the answer as a single letter (A, B, C, etc.) in the format 'Final Answer: [Letter]'.
ASSISTANT: Answer:[/INST]"""
    }
}

# =================================================================
# 2. 輔助函數
# =================================================================

def parse_scene_graph(text):
    if text is None: return "N/A"
    try:
        match = re.search(r"Scene Graph:(.*)", text, re.DOTALL | re.IGNORECASE)
        sg = match.group(1).strip() if match else text.strip()
        if "ASSISTANT:" in sg: sg = sg.split("ASSISTANT:")[-1].strip()
        return sg if sg else "N/A"
    except:
        return "PARSE_ERROR"

def parse_final_answer_mc(text):
    match = re.search(r"Final Answer:\s*([A-E])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"Answer:\s*([A-E])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    matches = re.findall(r"\b([A-E])\b", text)
    if matches: return matches[-1].upper()
    return ""

def _format_options(choices):
    output = ""
    for i, choice in enumerate(choices):
        output += f"{chr(65+i)}. {choice}\n"
    return output.strip()

def _build_ccot_inputs(prompts, mtype, stage, images, questions, choices_batch=None, scene_graphs=None, contexts=None):
    """ 
    構建輸入。
    關鍵邏輯：如果是 Stage 2 且 image 為 None，使用 Text-only prompt。
    """
    batch_size = len(questions)
    
    # 處理可能的 None Context
    safe_contexts = [c if c else "N/A" for c in contexts]
    
    # --- Qwen-VL ---
    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            img_obj = images[i]
            
            if stage == "stage_1":
                # 如果沒有圖，Stage 1 應該被跳過，但為了防呆，這裡回傳 None
                if img_obj is None:
                    messages_list.append(None)
                else:
                    prompt_text = prompts['stage_1_user'].format(question=questions[i])
                    messages = [
                        {"role": "system", "content": prompts['stage_1_system']}, 
                        {"role": "user", "content": [{"type": "image", "image": img_obj}, {"type": "text", "text": prompt_text}]}
                    ]
                    messages_list.append(messages)
            
            elif stage == "stage_2":
                options_str = _format_options(choices_batch[i])
                
                if img_obj is not None:
                    # Visual CCoT
                    prompt_text = prompts['stage_2_user_visual'].format(
                        scene_graph=scene_graphs[i],
                        context=safe_contexts[i],
                        question=questions[i],
                        options_str=options_str
                    )
                    content = [{"type": "image", "image": img_obj}, {"type": "text", "text": prompt_text}]
                else:
                    # Text-only CoT
                    prompt_text = prompts['stage_2_user_text'].format(
                        context=safe_contexts[i],
                        question=questions[i],
                        options_str=options_str
                    )
                    content = [{"type": "text", "text": prompt_text}] # 純文字輸入

                messages = [
                    {"role": "system", "content": prompts['stage_2_system']}, 
                    {"role": "user", "content": content}
                ]
                messages_list.append(messages)
                
        return {"messages_list": messages_list}

    # --- LLaVA ---
    elif mtype == "llava" or mtype == "llava-next":
        prompt_list = []
        for i in range(batch_size):
            img_obj = images[i]
            
            if stage == "stage_1":
                if img_obj is None:
                    prompt_list.append(None)
                else:
                    prompt = prompts['stage_1_prompt'].format(question=questions[i])
                    prompt_list.append(prompt)
            
            elif stage == "stage_2":
                options_str = _format_options(choices_batch[i])
                if img_obj is not None:
                    # Visual
                    prompt = prompts['stage_2_prompt_visual'].format(
                        scene_graph=scene_graphs[i],
                        context=safe_contexts[i],
                        question=questions[i],
                        options_str=options_str
                    )
                else:
                    # Text-only (注意：這裡移除了 <image> tag)
                    prompt = prompts['stage_2_prompt_text'].format(
                        context=safe_contexts[i],
                        question=questions[i],
                        options_str=options_str
                    )
                prompt_list.append(prompt)
                
        return {"prompt_list": prompt_list}
    else:
        raise NotImplementedError

# =================================================================
# 3. 生成與評分
# =================================================================

def ccot_generate_answer(proc, m, mtype, images, questions, messages_list=None, prompt_list=None, **kwargs):
    # 支援 Batch 處理，需處理 images 列表可能含有 None 的情況
    # 大多數 Processor (如 Qwen2VLProcessor) 不喜歡 images 列表中混雜 None 和 PIL.Image
    # 所以我們可能需要分開處理，或者確保傳入 processor 的格式正確
    
    # 簡單策略：如果該 Batch 混雜，通常 processor 會報錯。
    # 最佳解法：在 main loop 中根據是否有圖拆分 batch，或者在這裡過濾。
    # 為了保持 ccot_generate_answer 的通用性，我們假設 inputs 已經構建好，
    # 但對於 Text-only 樣本，images[i] 為 None。
    
    # --- LLaVA-NeXT 範例 ---
    if mtype == "llava-next":
        # 過濾掉 None 的 prompt (針對 Stage 1 跳過的情況)
        if prompt_list is None: return [""] * len(questions)
        
        valid_indices = [i for i, p in enumerate(prompt_list) if p is not None]
        if not valid_indices: return [""] * len(questions) # 全空
        
        # 準備有效的 batch
        sub_prompts = [prompt_list[i] for i in valid_indices]
        sub_images = [images[i] for i in valid_indices]
        
        # 檢查是否全為純文字 (Text-only batch)
        # LLaVA processor 如果收到 images=None 列表可能會報錯，需視具體版本而定
        # 安全做法：如果 sub_images 全是 None，則不傳 images 參數
        
        final_prompts = []
        for p in sub_prompts:
            if "[INST]" not in p and "USER:" not in p:
                if "<image>" in p: # 有圖的 prompt
                    final_prompts.append(f"[INST] <image>\n{p} [/INST]")
                else: # 無圖的 prompt
                    final_prompts.append(f"[INST] {p} [/INST]")
            else:
                final_prompts.append(p)

        if all(img is None for img in sub_images):
            # 純文字模式
            inputs = proc(text=final_prompts, return_tensors="pt", padding=True).to(m.device)
        else:
            # 混和或純圖模式 (Processor 通常要求 images 與 prompt 中的 <image> 數量匹配)
            # 如果 batch 裡有的有圖有的沒圖，LLaVA processor 處理起來很麻煩
            # **強烈建議**：在 Main Loop 就把 batch 拆開，或者 collate_fn 裡不要混和。
            # 但這裡我們嘗試處理：
            # 替換 None 為dummy image? 不行，會影響結果。
            # 簡單起見：假設 batch 內要嘛全有圖，要嘛全沒圖 (DataLoader shuffle=False 時 ScienceQA 經常是混和的)
            # **修正策略**: 這裡簡單實作：逐個樣本生成 (效率低但穩)，或是過濾出有圖的做一次，沒圖的做一次。
            
            # 這裡演示「逐個樣本」回退機制，確保不會報錯
            pass # (下方的邏輯會被執行)

        # 這裡為了代碼簡潔，我們採用「如果 batch 混和則逐個生成」的策略
        outputs_list = [""] * len(questions)
        
        for i in range(len(questions)):
            if prompt_list[i] is None: continue
            
            p = prompt_list[i]
            img = images[i]
            
            # Format
            if "[INST]" not in p and "USER:" not in p:
                txt = f"[INST] <image>\n{p} [/INST]" if img else f"[INST] {p} [/INST]"
            else:
                txt = p
            
            if img:
                inp = proc(images=img, text=txt, return_tensors="pt").to(m.device)
            else:
                inp = proc(text=txt, return_tensors="pt").to(m.device)
                
            input_len = inp['input_ids'].shape[1]
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['messages_list', 'prompt_list']}
            with torch.no_grad():
                out = m.generate(**inp, **clean_kwargs)
            new_tok = out[:, input_len:]
            res = proc.decode(new_tok[0], skip_special_tokens=True)
            outputs_list[i] = res.strip()
            
        return outputs_list

    # --- Qwen-VL (也採用逐個處理以保證穩定) ---
    if mtype == "qwen-vl":
        outputs_list = [""] * len(questions)
        for i in range(len(questions)):
            if messages_list[i] is None: continue
            
            # apply template
            txt = proc.apply_chat_template(messages_list[i], tokenize=False, add_generation_prompt=True)
            img = images[i]
            
            if img:
                inp = proc(text=[txt], images=[img], return_tensors="pt").to(m.device)
            else:
                inp = proc(text=[txt], return_tensors="pt").to(m.device)
            
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['messages_list', 'prompt_list']}
            clean_kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
            
            with torch.no_grad():
                out = m.generate(**inp, **clean_kwargs)
            
            # decode
            # Qwen output contains input, need trim? depends on version. usually yes.
            out_decoded = proc.decode(out[0], skip_special_tokens=True)
            if "assistant\n" in out_decoded:
                outputs_list[i] = out_decoded.split("assistant\n")[-1].strip()
            else:
                outputs_list[i] = out_decoded.strip()
                
        return outputs_list

    return [""] * len(questions)

def score_scienceqa(pred, refs):
    if not pred: return 0.0
    pred = pred.strip().upper() 
    refs = [a.strip().upper() for a in refs]
    return 1.0 if pred in refs else 0.0

# =================================================================
# 4. 主程式
# =================================================================

def main():
    if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
         os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Config
    config_dir = "configs"
    config_filename = 'scienceqa.yaml'
    file_path = os.path.join(config_dir, config_filename)
    if not os.path.exists(file_path): return
    cfg = yaml.safe_load(open(file_path))
    
    print(f"Running CCoT Evaluation on ScienceQA...")
    
    # Dataset
    ds = ScienceQADataset(
        dataset_id=cfg["dataset_id"],
        split="validation", 
        num_samples=cfg.get("num_val_samples", None)
    )
    
    batch_size = cfg.get('batch_size', 1)
    # Collate function needs to handle None images
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    all_results = {}
    STAGE1_MAX_TOKENS = 300
    STAGE2_MAX_TOKENS = 200 

    for mdl in cfg["models"]:
        name = mdl["name"]
        print(f"\nProcessing model: {name}")
        
        base_type = mdl['type']
        if base_type == 'llava-next': base_type = 'llava'
        
        if base_type not in CCOT_PROMPTS: continue
        prompts = CCOT_PROMPTS[base_type]
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        mdl_obj.eval()
        
        scores = []
        preds = []
        gen_params_1 = {"max_new_tokens": STAGE1_MAX_TOKENS, "do_sample": False}
        gen_params_2 = {"max_new_tokens": STAGE2_MAX_TOKENS, "do_sample": False}

        for batch_data in tqdm(ds_loader, desc=name):
            images = batch_data["image"] # List, may contain None
            questions = batch_data["question"]
            all_answers = batch_data["answers"]
            question_ids = batch_data["question_id"]
            choices_batch = batch_data["choices"]
            # ScienceQA 的 hint/lecture
            contexts = [batch_data.get("hint", [""])[k] + " " + batch_data.get("lecture", [""])[k] for k in range(len(questions))]

            # --- STAGE 1: 生成 Scene Graph (僅針對有圖的樣本) ---
            stage1_inputs = _build_ccot_inputs(
                prompts, base_type, "stage_1", images, questions
            )
            
            # 如果 batch 裡有些是 None (無圖)，stage1_outputs 對應位置會是空字串或被處理
            stage1_outputs = ccot_generate_answer(
                proc, mdl_obj, mtype,
                images, questions,
                **stage1_inputs,
                **gen_params_1
            )
            
            generated_sgs = [parse_scene_graph(out) for out in stage1_outputs]
            
            # --- STAGE 2: 回答 (有圖用 Visual Prompt, 無圖用 Text Prompt) ---
            stage2_inputs = _build_ccot_inputs(
                prompts, base_type, "stage_2", 
                images, questions, 
                choices_batch=choices_batch, 
                scene_graphs=generated_sgs,
                contexts=contexts
            )
            
            stage2_outputs = ccot_generate_answer(
                proc, mdl_obj, mtype,
                images, questions,
                **stage2_inputs,
                **gen_params_2
            )
            
            for i in range(len(stage2_outputs)):
                raw_ans = stage2_outputs[i]
                refs = all_answers[i]
                
                parsed_ans = parse_final_answer_mc(raw_ans)
                score = score_scienceqa(parsed_ans, refs)
                scores.append(score)
                
                preds.append({
                    "qid": str(question_ids[i]),
                    "pred": parsed_ans,
                    "refs": refs,
                    "score": score,
                    "scene_graph": generated_sgs[i],
                    "has_image": (images[i] is not None)
                })

        mean_acc = sum(scores)/len(scores) if scores else 0
        all_results[name] = {"acc": mean_acc, "predictions": preds}
        
        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    print("\nFinal CCoT Results (ScienceQA):")
    for k, v in all_results.items():
        print(f"{k}: Accuracy={v['acc']:.4f}")
        
    with open("results_ccot_scienceqa.json", 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()