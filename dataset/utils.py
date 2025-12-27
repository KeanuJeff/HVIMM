import torch
from transformers import (
    AutoProcessor, LlavaForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
    VisionEncoderDecoderModel, AutoTokenizer,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    LlavaNextProcessor,
    AutoModel,
    Qwen3VLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    LlavaNextForConditionalGeneration
)
from models.llava_model import ModifiedLlavaModel
from peft import PeftModel
from PIL import Image
from models.structural_llava_next_raw import HybirdLlavaFlorenceModel
import os

# -----------------------------------------------------------------
# 載入模型 (此函數不變)
# -----------------------------------------------------------------
def parse_final_answer(full_text):
    """
    從完整的 CoT 輸出中提取 'Final Answer: [word]'
    """
    try:
        # 尋找 'Final Answer:' (不分大小寫)
        split_tag = "Final Answer:"
        parts = full_text.split(split_tag)
        if len(parts) < 2:
             # 如果找不到 'Final Answer:'，嘗試直接取最後一個詞
             return full_text.strip().split()[-1]
             
        # 取得 'Final Answer:' 後的所有內容
        final_part = parts[-1]
        
        # 清理並獲取第一個詞
        final_answer = final_part.strip().split('\n')[0].strip().split(' ')[0].strip()
        
        # 移除可能的標點符號
        final_answer = final_answer.rstrip('.,')
        
        return final_answer
    except Exception:
        return "" # 發生錯誤時返回空字串


def load_model_and_processor(cfg):
    mid, mtype = cfg["model_id"], cfg["type"]
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, # 你的兩個訓練腳本都使用 float16
        bnb_4bit_use_double_quant=True,
    )
    """
    if mtype == "gemma3":
        print(f"Loading Gemma 3 model: {mid}")
        
        # 1. 載入 Processor
        # Gemma 3 使用 AutoProcessor 來處理圖片與文字
        proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        
        # 2. 載入 Model
        # 根據 pipeline task "image-text-to-text"，使用對應的 Auto Class
        # 顯存允許的話，建議開啟 flash_attention_2
        # device_map="auto" 會自動分配 GPU
        m = Gemma3ForConditionalGeneration.from_pretrained(
            mid, device_map="auto", torch_dtype=torch.bfloat16
            ).eval()
        
        return proc, m, "gemma3"
    
    if mtype == "qwen-lora":
        print(f"Loading 4-bit base model: {mid}")
        proc = Qwen2VLProcessor.from_pretrained(mid, trust_remote_code=True)
        m = Qwen2VLForConditionalGeneration.from_pretrained(
            mid,
            #quantization_config=bnb_config, # 使用 4-bit
            device_map="auto",
            trust_remote_code=True
        )
        
        adapter_path = cfg["adapter_path"]
        print(f"Applying Qwen-VL LoRA adapter from: {adapter_path}")
        # 載入 LoRA 權重
        m = PeftModel.from_pretrained(m, adapter_path)
        # 【重要】合併權重以加速推論，並卸載 LoRA
        m = m.merge_and_unload() 
        
        m.eval()
        # 返回 "qwen-vl" 原始類型，讓 generate_answer 函數可以正常運作
        return proc, m, "qwen-vl"

    # --- 【新增】載入你訓練的 LLaVA LoRA 模型 ---
    if mtype == "llava-lora":
        print(f"Loading 4-bit base model: {mid}")
        proc = AutoProcessor.from_pretrained(mid, use_fast=True)
        m = LlavaForConditionalGeneration.from_pretrained(
            mid,
            #quantization_config=bnb_config, # 使用 4-bit
            device_map="auto",
        )
        
        adapter_path = cfg["adapter_path"]
        print(f"Applying LLaVA LoRA adapter from: {adapter_path}")
        # 載入 LoRA 權重
        m = PeftModel.from_pretrained(m, adapter_path)
        # 【重要】合併權重以加速推論，並卸載 LoRA
        m = m.merge_and_unload()
        
        m.eval()
        # 返回 "llava" 原始類型，讓 generate_answer 函數可以正常運作
        return proc, m, "llava"
    
    if mtype == "qwen-vl":
        proc = Qwen2VLProcessor.from_pretrained(mid, trust_remote_code=True)
        m = Qwen2VLForConditionalGeneration.from_pretrained(
            mid,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )
        m.eval()
        return proc, m, "qwen-vl"

    if mtype == "dvit_llava":
        proc = AutoProcessor.from_pretrained(mid, use_fast=True)
        m = ModifiedLlavaModel(model_id=mid)
        m.vision_tower.to("cuda")
        m.multi_modal_projector.to("cuda")
        return proc, m, "dvit_llava"

    if mtype == "llava":
        proc = AutoProcessor.from_pretrained(mid, use_fast=True)
        m = LlavaForConditionalGeneration.from_pretrained(mid, dtype=torch.float16, device_map="auto",)
        return proc, m, "llava"
    
    if mtype == "llava-next":
        print(f"Loading LLaVA-NeXT model: {mid}")
        proc = LlavaNextProcessor.from_pretrained(mid)
        m = LlavaNextForConditionalGeneration.from_pretrained(
            mid,
            #quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        m.eval()
        return proc, m, "llava-next"
        
    if mtype == "instructblip":
        proc = InstructBlipProcessor.from_pretrained(mid, use_fast=True)
        m = InstructBlipForConditionalGeneration.from_pretrained(mid, dtype=torch.float16, device_map="auto")
        return proc, m, "instructblip"
        
    if mtype == "qwen3-vl":
        print(f"Loading Qwen3-VL model: {mid}")
        
        # 【關鍵修正】: 載入 Tokenizer 時，強制設定 padding_side='left'
        tokenizer = AutoTokenizer.from_pretrained(mid, 
                                                 trust_remote_code=True, 
                                                 padding_side='left') # <--- 修正處
        
        proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        
        # 載入模型 (保持不變)
        try:
            m = Qwen3VLForConditionalGeneration.from_pretrained(
                mid,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            )
        except NameError:
            print("Warning: Qwen3VL class not found, falling back to Qwen2VL.")
            m = Qwen2VLForConditionalGeneration.from_pretrained(
                mid,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            )
        
        # Qwen-VL Processor 需要 Tokenizer 實例
        proc.tokenizer = tokenizer # 確保 processor 內的 tokenizer 也是左側填充
            
        m.eval()
        return proc, m, "qwen3-vl"
    if mtype == "structural_llava":
        print(f"Loading Custom Structural LLaVA: {mid}")
        
        # 1. 初始化模型
        # 注意: 推論時必須 load_florence=True，因為需要它來產生幾何特徵
        model = HybirdLlavaFlorenceModel(
            llava_model_id=mid,
            load_llava=True,
            load_florence=True
        )
        if hasattr(model, "vit_pos_embed") and model.llava is not None:
            # 將 buffer 移動到與 LLaVA 主模型相同的裝置 (通常是 cuda:0)
            target_device = model.llava.device
            model.vit_pos_embed = model.vit_pos_embed.to(target_device)
        
        # 2. 載入訓練好的權重
        custom_path = cfg.get("custom_modules_path")
        lora_path = cfg.get("adapter_path")
        
        if custom_path and os.path.exists(custom_path):
            print(f"Loading Custom Modules from {custom_path}...")
            state_dict = torch.load(custom_path, map_location=model.llava.device)
            model.adapter.load_state_dict(state_dict["adapter"])
            model.shape_projector.load_state_dict(state_dict["shape_projector"])
            model.label_down_projector.load_state_dict(state_dict["label_down_projector"])
            model.adapter.to(dtype=torch.float16)
            model.shape_projector.to(dtype=torch.float16)
            model.label_down_projector.to(dtype=torch.float16)
            model.rope_2d.to(dtype=torch.float16)
            if hasattr(model, 'output_norm'):
                model.output_norm.to(dtype=torch.float16)
        else:
            print("Warning: Custom modules path not found or empty!")

        
        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA from {lora_path}...")
            # 這裡 model.llava 是 base model，我們把 LoRA 掛上去
            model.llava = PeftModel.from_pretrained(model.llava, lora_path)
        
            
        model.eval()
        # 回傳 processor, model object, type string
        return model.llava_processor, model, "structural_llava"
        
    raise ValueError(mtype)


# -----------------------------------------------------------------
# 【修改】: 生成答案 (現在接受批次)
# -----------------------------------------------------------------
VQA_PROMPT_REASONING = (
    "Answer the following question either directly or with some reasoning. Give the final answer in a single word like 'Final Answer: [single word]'.\n"
    "Question: {question}\n"
    "Answer:"
)

def generate_answer(proc, m, mtype, images, questions, **kwargs):
    """
    【已修改】: 此函數現在使用 CoT 提示，並解析 'Final Answer:'。
    """
    
    # 'images' 是一個 PIL 圖像列表
    # 'questions' 是一個問題字串列表
    batch_size = len(questions)

    # ----------------------------------------------------
    # Qwen-VL (批次處理)
    # ----------------------------------------------------
    if mtype == "gemma3":
        messages_list = []
        batch_images = [] # 儲存 PIL 圖片

        # 1. 準備 Messages 結構並收集圖片
        for i in range(batch_size):
            # 套用 CoT Prompt
            prompt_text = VQA_PROMPT_REASONING.format(question=questions[i])
            
            # 【修改 1】：參考 Qwen-VL 邏輯，將 PIL 圖片物件直接放入 content 結構中。
            # 【修改 2】：此邏輯會為每個問題/圖片對建立一個完整的 message，以支援批次。
            messages = [
                {
                    "role": "user",
                    "content": [
                        # 將 PIL 圖片物件傳遞給 "image" 鍵，等待 Processor 處理
                        {"type": "image", "image": images[i]}, 
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            messages_list.append(messages)
            batch_images.append(images[i]) # 收集圖片以供下一步傳入 proc()

        # 2. 使用 apply_chat_template 轉成模型看得懂的 Prompt 字串
        # proc.apply_chat_template 會將 messages 轉為模型專用的 string (包含 <image> token)
        inputs = [
            proc.apply_chat_template(
                msg, 
                tokenize=True, 
                add_generation_prompt=True,
                return_dict=True,
                 padding=True,
                return_tensors="pt"
            ).to(m.device)
            for msg in messages_list
        ]

        # 4. 生成 (不變)
        kwargs.setdefault('max_new_tokens', 200)
        #input_len = inputs["input_ids"].shape[-1]
        
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)
            

        # 5. 解碼 (不變)
        # 裁切掉 input tokens，只保留生成的答案
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = proc.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        # 6. 解析 (不變)
        return [parse_final_answer(ans) for ans in output_text]
    
    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
            # <-- 【修改】使用新的 CoT 提示
            VQA_PROMPT = VQA_PROMPT_REASONING.format(question=questions[i])
            
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
            text=text_prompts, 
            images=images,
            return_tensors="pt",
            padding=True
        ).to(m.device)
        
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        
        with torch.no_grad():
            res = m.generate(**inputs, **kwargs)
        
        responses = proc.batch_decode(res, skip_special_tokens=True)
        
        # <-- 【修改】簡化解析邏輯
        # 提取 'assistant' 後的完整回答
        final_answers_full = []
        split_tag = "assistant\n"
        for response in responses:
            if split_tag in response:
                answer = response.split(split_tag)[-1].strip()
            else:
                answer = response.strip()
            final_answers_full.append(answer)

        # <-- 【修改】對每個完整回答應用 parse_final_answer
        return [parse_final_answer(ans) for ans in final_answers_full]

    # ----------------------------------------------------
    # LLaVA / dvit_llava (批次處理)
    # ----------------------------------------------------
    if mtype == "llava" or mtype == "dvit_llava" or mtype == "llava-next":
        prompt_list = []
        for q in questions:
            # 根據模型類型選擇 Prompt 格式
            if mtype == "llava-next":
                # LLaVA-NeXT (Mistral 版本) 使用 [INST] 格式
                # 您的 CoT Prompt (VQA_PROMPT_REASONING) 應該已經在傳入前格式化好了，或者在這裡格式化
                # 這裡假設 questions[i] 已經包含了您的問題文本
                # 如果您有統一的 Prompt Template (如 VQA_PROMPT_REASONING)，請在這裡套用
                
                # 假設外部傳入原始問題，這裡套用 Prompt：
                formatted_q = f"Answer the following question either directly or with some reasoning. Give the final answer in a single word like 'Final Answer: [single word]'.\nQuestion: {q}\nAnswer:"
                
                prompt_text = f"[INST] <image>\n{formatted_q} [/INST]"
                prompt_list.append(prompt_text)
            else:
                # 舊版 LLaVA v1.5
                formatted_q = f"Answer the following question either directly or with some reasoning. Give the final answer in a single word like 'Final Answer: [single word]'.\nQuestion: {q}\nAnswer:"
                prompt_list.append(f"USER: <image>\n{formatted_q} ASSISTANT:")
        
        inputs = proc(
            images=images, 
            text=prompt_list,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(m.device) for k, v in inputs.items()}
        
        # 產生回答
        input_len = inputs['input_ids'].shape[1]
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        
        with torch.no_grad():
            outputs = m.generate(**inputs, **kwargs)
        
        new_tokens = outputs[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)
        
        # 使用 parse_final_answer 提取最終答案
        # 注意：請確保您的 parse_final_answer 函數在這個檔案的前面已經定義了
        return [parse_final_answer(ans) for ans in answers_full]
        
    # ----------------------------------------------------
    # InstructBLIP (批次處理)
    # ----------------------------------------------------
    if mtype == "instructblip":
        prompt_list = []
        for q in questions:
            # <-- 【修改】使用新的 CoT 提示
            VQA_PROMPT = VQA_PROMPT_REASONING.format(question=q)
            prompt_list.append(VQA_PROMPT)

        inp = proc(
            images=images,
            text=prompt_list,
            return_tensors="pt", 
            padding=True
        ).to(m.device)
        input_len = inp['input_ids'].shape[1]
        
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        out = m.generate(
            **inp, 
            **kwargs
        )
        new_tokens = out[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)
        
        # <-- 【修改】對每個完整回答應用 parse_final_answer
        return [parse_final_answer(ans) for ans in answers_full]

    # ----------------------------------------------------
    # MiniGPT-4 (批次處理)
    # ----------------------------------------------------
    if mtype == "qwen3-vl":
        messages_list = []
        
        # 1. 準備 Messages 結構
        for i in range(batch_size):
            # 套用 CoT Prompt
            prompt_text = VQA_PROMPT_REASONING.format(question=questions[i])
            
            # Qwen-VL 標準格式: content 內包含 image 和 text
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[i]}, # 直接傳入 PIL Image
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            messages_list.append(msg)

        # 2. 使用 apply_chat_template 轉成模型看得懂的 Prompt (不包含圖片 Tensor)
        # tokenize=False 讓它只回傳字串，圖片處理交給下一步
        texts = [
            proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]

        # 3. 處理輸入 (將圖片和文字轉為 Tensor)
        # Qwen2VLProcessor 會自動處理 images 列表中的 PIL 圖片
        inputs = proc(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        
        # 移動到 GPU
        inputs = inputs.to(m.device)

        # 4. 生成
        kwargs.setdefault('max_new_tokens', 128)
        
        with torch.no_grad():
            generated_ids = m.generate(**inputs, **kwargs)

        # 5. 解碼
        # Qwen 需要裁切掉 input 的長度，只保留生成的答案
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = proc.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        # 6. 解析 Final Answer
        return [parse_final_answer(ans) for ans in output_text]
    if mtype == "structural_llava":
        answers = []
        # 因為 model.py 的 generate_answer 是針對單張圖片設計的 (包含 Florence + 幾何運算 + Embedding 替換)
        # 所以這裡我們用迴圈處理 Batch (雖然慢一點，但保證邏輯正確)
        for i in range(batch_size):
            # 呼叫 model 內部的生成函式
            ans = m.generate_answer(images[i], VQA_PROMPT_REASONING.format(question=questions[i]))
            
            # 嘗試解析答案 (如果模型輸出了長篇大論)
            parsed_ans = parse_final_answer(ans)
            # 如果解析出來是空的(代表沒有 Final Answer 標籤)，就直接用原答案
            if not parsed_ans:
                parsed_ans = ans.strip()
            
            answers.append(parsed_ans)
        return answers
    else:
        return [""] * batch_size


# -----------------------------------------------------------------
# 【修改】: Collate Function (現在支援批次)
# -----------------------------------------------------------------
def collate_fn(batch):
    """
    【已修改】: 此函數現在將一個樣本列表 (batch) 合併為一個批次字典。
    """
    
    # 'batch' 是一個列表，包含 N 個來自 VQADataset 的字典
    # e.g., [
    #   {"image": p1, "question": q1, "answers": a1, "question_id": id1},
    #   {"image": p2, "question": q2, "answers": a2, "question_id": id2}
    # ]
    
    collated_batch = {}
    keys = batch[0].keys() # 獲取所有鍵 (image, question, etc.)
    
    for key in keys:
        # 將所有樣本的 'key' 值收集到一個列表中
        collated_batch[key] = [sample[key] for sample in batch]

    # 'collated_batch' 現在看起來像這樣：
    # {
    #   "image": [p1, p2],
    #   "question": [q1, q2],
    #   "answers": [a1, a2],
    #   "question_id": [id1, id2]
    # }
    
    return collated_batch