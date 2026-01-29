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

def parse_final_answer(full_text):
    """
    從完整的 CoT 輸出中提取 'Final Answer: [word]'
    """
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


def load_model_and_processor(cfg):
    mid, mtype = cfg["model_id"], cfg["type"]

    if mtype == "gemma3":
        print(f"Loading Gemma 3 model: {mid}")
        
        proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        
        m = Gemma3ForConditionalGeneration.from_pretrained(
            mid, device_map="auto", torch_dtype=torch.bfloat16
            ).eval()
        
        return proc, m, "gemma3"
    
    if mtype == "qwen-lora":
        print(f"Loading 4-bit base model: {mid}")
        proc = Qwen2VLProcessor.from_pretrained(mid, trust_remote_code=True)
        m = Qwen2VLForConditionalGeneration.from_pretrained(
            mid,
            device_map="auto",
            trust_remote_code=True
        )
        
        adapter_path = cfg["adapter_path"]
        print(f"Applying Qwen-VL LoRA adapter from: {adapter_path}")
        m = PeftModel.from_pretrained(m, adapter_path)
        m = m.merge_and_unload() 
        
        m.eval()
        return proc, m, "qwen-vl"

    if mtype == "llava-lora":
        print(f"Loading 4-bit base model: {mid}")
        proc = AutoProcessor.from_pretrained(mid, use_fast=True)
        m = LlavaForConditionalGeneration.from_pretrained(
            mid,
            device_map="auto",
        )
        
        adapter_path = cfg["adapter_path"]
        print(f"Applying LLaVA LoRA adapter from: {adapter_path}")
        m = PeftModel.from_pretrained(m, adapter_path)
        m = m.merge_and_unload()
        
        m.eval()
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
        
        tokenizer = AutoTokenizer.from_pretrained(mid, 
                                                 trust_remote_code=True, 
                                                 padding_side='left')
        
        proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        
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
        
        proc.tokenizer = tokenizer
            
        m.eval()
        return proc, m, "qwen3-vl"

    if mtype == "structural_llava":
        print(f"Loading Custom Structural LLaVA: {mid}")
        
        model = HybirdLlavaFlorenceModel(
            llava_model_id=mid,
            load_llava=True,
            load_florence=True
        )
        if hasattr(model, "vit_pos_embed") and model.llava is not None:
            target_device = model.llava.device
            model.vit_pos_embed = model.vit_pos_embed.to(target_device)
        
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
            model.llava = PeftModel.from_pretrained(model.llava, lora_path)
        
            
        model.eval()
        return model.llava_processor, model, "structural_llava"
        
    raise ValueError(mtype)


VQA_PROMPT_REASONING = (
    "Answer the following question either directly or with some reasoning. Give the final answer in a single word like 'Final Answer: [single word]'.\n"
    "Question: {question}\n"
    "Answer:"
)

def generate_answer(proc, m, mtype, images, questions, **kwargs):
    """
    【已修改】: 此函數現在使用 CoT 提示，並解析 'Final Answer:'。
    """
    
    batch_size = len(questions)

    if mtype == "gemma3":
        messages_list = []
        batch_images = []

        for i in range(batch_size):
            prompt_text = VQA_PROMPT_REASONING.format(question=questions[i])
            
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

        return [parse_final_answer(ans) for ans in output_text]
    
    if mtype == "qwen-vl":
        messages_list = []
        for i in range(batch_size):
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
        
        final_answers_full = []
        split_tag = "assistant\n"
        for response in responses:
            if split_tag in response:
                answer = response.split(split_tag)[-1].strip()
            else:
                answer = response.strip()
            final_answers_full.append(answer)

        return [parse_final_answer(ans) for ans in final_answers_full]

    if mtype == "llava" or mtype == "dvit_llava" or mtype == "llava-next":
        prompt_list = []
        for q in questions:
            if mtype == "llava-next":
                formatted_q = f"Answer the following question either directly or with some reasoning. Give the final answer in a single word like 'Final Answer: [single word]'.\nQuestion: {q}\nAnswer:"
                
                prompt_text = f"[INST] <image>\n{formatted_q} [/INST]"
                prompt_list.append(prompt_text)
            else:
                formatted_q = f"Answer the following question either directly or with some reasoning. Give the final answer in a single word like 'Final Answer: [single word]'.\nQuestion: {q}\nAnswer:"
                prompt_list.append(f"USER: <image>\n{formatted_q} ASSISTANT:")
        
        inputs = proc(
            images=images, 
            text=prompt_list,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(m.device) for k, v in inputs.items()}
        
        input_len = inputs['input_ids'].shape[1]
        kwargs.setdefault('pad_token_id', proc.tokenizer.eos_token_id)
        
        with torch.no_grad():
            outputs = m.generate(**inputs, **kwargs)
        
        new_tokens = outputs[:, input_len:]
        answers_full = proc.batch_decode(new_tokens, skip_special_tokens=True)
        
        return [parse_final_answer(ans) for ans in answers_full]
        
    if mtype == "instructblip":
        prompt_list = []
        for q in questions:
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
        
        return [parse_final_answer(ans) for ans in answers_full]

    if mtype == "qwen3-vl":
        messages_list = []
        
        for i in range(batch_size):
            prompt_text = VQA_PROMPT_REASONING.format(question=questions[i])
            
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

        kwargs.setdefault('max_new_tokens', 128)
        
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

        return [parse_final_answer(ans) for ans in output_text]
    if mtype == "structural_llava":
        answers = []
        for i in range(batch_size):
            ans = m.generate_answer(images[i], VQA_PROMPT_REASONING.format(question=questions[i]))
            
            parsed_ans = parse_final_answer(ans)
            if not parsed_ans:
                parsed_ans = ans.strip()
            
            answers.append(parsed_ans)
        return answers
    else:
        return [""] * batch_size


def collate_fn(batch):
    
    collated_batch = {}
    keys = batch[0].keys() 
    
    for key in keys:
        collated_batch[key] = [sample[key] for sample in batch]

    return collated_batch
