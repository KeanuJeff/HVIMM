# methods/ccot.py
#
# -----------------------------------------------------------------
# 【完整版】: CCoT (Compositional Chain-of-Thought) 評估腳本
# -----------------------------------------------------------------

import os
import yaml
import torch
import gc
import re
import json
from torch.utils.data import DataLoader
from dataset.vqa import VQADataset
from tqdm import tqdm
from PIL import Image

# 【依賴導入】: 只導入必要的外部函數
from dataset.utils import load_model_and_processor, collate_fn 

# =================================================================
# 1. CCoT 提示詞模板 (內部定義)
# =================================================================

CCOT_PROMPTS = {
    # -----------------------------------------------------------------
    # Qwen-VL (ChatML 格式)
    # -----------------------------------------------------------------
    "qwen-vl": {
        'stage_1_system': "You are an expert scene description assistant.",
        # 【修改點 1】: 採用 JSON 格式要求，模仿原始 sgPrompt
        'stage_1_user': """For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Object relationships that are relevant to answering the question.

Question: {question}
Scene Graph:""", # 注意：JSON 輸出應從這裡開始

        'stage_2_system': "You are a VQA assistant. Answer the question based on the provided context.",
        # 【修改點 2】: 結合 Answer Prompt 並要求單詞答案 (VQA v2)
        'stage_2_user': """Use the image and the following Scene Graph to reason and answer the question with a single word.
Scene Graph: {scene_graph}
Question: {question}
Answer:""", # 這裡省略了 "Answer the question with a single word or short phrase" 讓模型直接輸出答案標籤
    },
    
    # -----------------------------------------------------------------
    # LLaVA (USER: ... ASSISTANT: 格式)
    # -----------------------------------------------------------------
    "llava-next": {
        # 【修改點 3】: 採用 JSON 格式要求，模仿原始 sgPrompt
        'stage_1_prompt': """[INST] <image>
For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Object relationships that are relevant to answering the question.

Question: {question}
ASSISTANT: Scene Graph: [/INST]""",

        # 【修改點 4】: 結合 Answer Prompt 並要求單詞答案 (VQA v2)
        'stage_2_prompt': """[INST] <image>
Use the image and the following Scene Graph to reason and answer the question with a single word.
Scene Graph: {scene_graph}
Question: {question}
ASSISTANT: Answer: [/INST]""",
    }
}


# =================================================================
# 2. CCoT 輔助函數 (解析與輸入構建)
# =================================================================

def parse_scene_graph(text):
    """ 從 CCoT 階段 1 的輸出中解析場景圖。"""
    try:
        # 尋找 "Scene Graph:" 標記
        match = re.search(r"Scene Graph:(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            sg = match.group(1).strip()
        else:
            # 如果找不到標記，則回退到使用整個輸出
            sg = text.strip()
        
        # 簡單清理
        if sg.startswith("ASSISTANT:"):
            sg = sg.split("ASSISTANT:", 1)[-1].strip()
            
        return sg if sg else "N/A" # 確保不返回空字串
    except Exception as e:
        print(f"Error parsing Scene Graph: {e}\nOutput was: {text}")
        return "PARSE_ERROR"

def parse_final_answer(text):
    """ 從 CCoT 階段 2 的輸出中解析最終答案。"""
    try:
        # 尋找 "Answer:" 標記
        match = re.search(r"Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
        else:
            # 如果找不到標記，則回退到使用整個輸出
            answer = text.strip()
        
        # 確保只取答案的第一行
        answer = answer.split('\n')[0].strip()

        # 限制答案為單一詞彙 (VQA v2 安全網)
        if answer != "N/A":
            answer = answer.split(' ')[0].strip()
            
        return answer
    except Exception as e:
        print(f"Error parsing Final Answer: {e}\nOutput was: {text}")
        return "PARSE_ERROR"


def _build_ccot_inputs(prompts, mtype, stage, images, questions, **kwargs):
    """ 輔助函數：為 ccot_generate_answer() 構建關鍵字參數。"""
    batch_size = len(questions)
    
    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            if stage == "stage_1":
                prompt_text = prompts['stage_1_user'].format(question=questions[i])
                messages = [{"role": "system", "content": prompts['stage_1_system']}, 
                            {"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": prompt_text}]}]
            elif stage == "stage_2":
                prompt_text = prompts['stage_2_user'].format(
                    scene_graph=kwargs['scene_graphs'][i], 
                    question=questions[i]
                )
                messages = [{"role": "system", "content": prompts['stage_2_system']}, 
                            {"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": prompt_text}]}]
            messages_list.append(messages)
        return {"messages_list": messages_list}

    elif mtype == "llava-next":
        prompt_list = []
        for i in range(batch_size):
            if stage == "stage_1":
                prompt = prompts['stage_1_prompt'].format(question=questions[i])
            elif stage == "stage_2":
                prompt = prompts['stage_2_prompt'].format(
                    scene_graph=kwargs['scene_graphs'][i], 
                    question=questions[i]
                )
            prompt_list.append(prompt)
        return {"prompt_list": prompt_list}
        
    else:
        raise NotImplementedError(f"CCoT inputs not implemented for mtype: {mtype}")


# =================================================================
# 3. 【本地】: ccot_generate_answer 函數 (從 ddcot.py 複製而來)
# =================================================================
def ccot_generate_answer(proc, m, mtype, images, questions, 
                         messages_list=None, prompt_list=None, **kwargs):
    """
    從 ddcot.py 複製而來，專用於 CCoT。
    修復了 LLaVA/Qwen 的 kwargs 傳遞錯誤 (ValueError/TypeError)。
    """
    
    batch_size = len(questions)

    if mtype == "qwen-vl":
        
        if messages_list is None:
            # (標準 VQA 模式的回退)
            messages_list = []
            for i in range(batch_size):
                VQA_PROMPT = f"Question: {questions[i]}\nAnswer:"
                messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": VQA_PROMPT}]}]
                messages_list.append(messages)

        final_text_prompts = []
        final_images = []
        
        for i, msg_list in enumerate(messages_list):
            clean_msg_list = []
            for msg in msg_list:
                content = msg.get('content')
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                elif content is None:
                    content = []
                new_content = []
                for item in content:
                    if item.get('type') == 'image': new_content.append({"type": "image"}) 
                    elif item.get('type') == 'text': new_content.append(item)
                if new_content:
                    clean_msg_list.append({"role": msg['role'], "content": new_content})
            
            text_prompt = proc.apply_chat_template(clean_msg_list, tokenize=False, add_generation_prompt=True)
            final_text_prompts.append(text_prompt)
            final_images.append(images[i])

        inputs = proc(text=final_text_prompts, images=final_images, return_tensors="pt", padding=True).to(m.device)
        kwargs.pop('messages_list', None); kwargs.pop('prompt_list', None) 
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        with torch.no_grad():
            res = m.generate(**inputs, **kwargs)
        responses = proc.batch_decode(res, skip_special_tokens=True)
        final_answers = []
        split_tag = "assistant\n"
        for response in responses:
            if split_tag in response: answer = response.split(split_tag)[-1].strip()
            else: answer = response.strip()
            final_answers.append(answer)
        return final_answers

    elif mtype == "llava-next" or mtype == "dvit_llava":
        if prompt_list is None:
            prompt_list = []
            for q in questions:
                VQA_PROMPT = f"Question: {q}\nAnswer:"
                prompt_list.append(f"USER: <image>\n{VQA_PROMPT} ASSISTANT:")
        inputs = proc(images=images, text=prompt_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(m.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]
        kwargs.pop('prompt_list', None); kwargs.pop('messages_list', None)
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        outputs = m.generate(**inputs, **kwargs)
        new_tokens = outputs[:, input_len:]
        answers = proc.batch_decode(new_tokens, skip_special_tokens=True)
        return [ans.strip() for ans in answers]
        
    return [""] * batch_size


# =================================================================
# 4. VQA 計分函數 (從 evaluate_vqa.py 複製)
# =================================================================

def vqa_score(pred, refs):
    """ 從 evaluate_vqa.py 複製而來，用於計算 VQA 分數。"""
    pred = pred.strip().lower()
    refs = [a.strip().lower() for a in refs]
    agree = sum(r == pred for r in refs)
    return min(1.0, agree / 3.0)


# =================================================================
# 5. 主執行函數 (Main)
# =================================================================

def main():
    if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
         os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
         
    config_dir = "configs"
    config_filename = 'vqa.yaml'
    file_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at {file_path}")
        return
        
    cfg = yaml.safe_load(open(file_path))
    
    # --- CCoT 特定設置 ---
    STAGE1_MAX_TOKENS = 200  # 場景圖需要較長
    STAGE2_MAX_TOKENS = 20   # 答案需要較短 (VQA)
    print(f"Running CCoT (Compositional CoT) Evaluation...")
    # -------------------------

    ds = VQADataset(cfg["val_img_dir"], cfg["ques_json"], cfg["anns_json"], cfg["num_samples"])
    batch_size = cfg.get('batch_size', 1)
    print(f"Using Batch Size: {batch_size}")
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_results = {}

    for mdl in cfg["models"]:
        name = mdl["name"]
        print(f"\nProcessing model: {name} (Type: {mdl['type']})")
        
        if mdl['type'] not in CCOT_PROMPTS:
            print(f"Warning: No CCoT prompts defined for model type '{mdl['type']}'. Skipping this model.")
            continue
            
        prompts = CCOT_PROMPTS[mdl['type']]
        proc, mdl_obj, mtype = load_model_and_processor(mdl)
        mdl_obj.eval()
        scores, preds = [], []
        
        # 兩個階段使用不同的生成參數
        gen_params_stage1 = {
            "max_new_tokens": STAGE1_MAX_TOKENS,
            "do_sample": False, "min_new_tokens": 1, "num_beams": 1
        }
        gen_params_stage2 = {
            "max_new_tokens": STAGE2_MAX_TOKENS,
            "do_sample": False, "min_new_tokens": 1, "num_beams": 1
        }

        # --- CCoT 主迴圈 ---
        for batch_data in tqdm(ds_loader, desc=name):
            
            images = batch_data["image"]
            questions = batch_data["question"]
            all_answers = batch_data["answers"]
            question_ids = batch_data["question_id"]

            # --- STAGE 1: 場景圖生成 ---
            stage1_inputs = _build_ccot_inputs(prompts, mtype, "stage_1", images, questions)
            
            stage1_outputs = ccot_generate_answer(
                proc, mdl_obj, mtype,
                images, questions,
                **stage1_inputs,
                **gen_params_stage1
            )
            
            # 解析場景圖
            generated_sgs = [parse_scene_graph(out) for out in stage1_outputs]

            # --- STAGE 2: 答案提取 ---
            stage2_inputs = _build_ccot_inputs(
                prompts, mtype, "stage_2", 
                images, questions, 
                scene_graphs=generated_sgs # 傳入生成的 SG
            )
            
            stage2_outputs = ccot_generate_answer(
                proc, mdl_obj, mtype,
                images, questions,
                **stage2_inputs,
                **gen_params_stage2
            )

            # --- 解析 & 計分 ---
            for i in range(len(stage2_outputs)):
                final_output = stage2_outputs[i]
                ans = parse_final_answer(final_output)
                
                q = questions[i]
                refs = all_answers[i]
                qid = question_ids[i]
                if not refs or not all(r.strip() for r in refs): continue
                
                score = vqa_score(ans, refs)
                scores.append(score)
                preds.append({
                    "qid": qid, 
                    "pred": ans, 
                    "refs": refs, 
                    "score": score,
                    "generated_scene_graph": generated_sgs[i] # 儲存理由
                })
        
        # --- 模型計分與清理 ---
        if scores: mean_acc = sum(scores) / len(scores)
        else: mean_acc = 0.0
        all_results[name] = {"acc": mean_acc, "predictions": preds}
        
        del mdl_obj, proc
        gc.collect()
        torch.cuda.empty_cache()

    # --- 最終結果報告 ---
    print("\n" + "="*30)
    print("Final CCoT Results:")
    print("="*30)
    for k, v in all_results.items():
        print(f"{k}: {v['acc']:.4f}")

    output_filename = f"results_ccot.json"
    with open(output_filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed CCoT results (with scene graphs) saved to {output_filename}")


if __name__ == "__main__":
    main()