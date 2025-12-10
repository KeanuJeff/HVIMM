# models/llava_model.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, LlavaForConditionalGeneration
from transformers import CLIPImageProcessor
from PIL import Image
from typing import Optional, List

# 從 models 資料夾 import
from models.dvit import CustomVisionTower, CustomProjector


def preprocess_images(raw_images, processor, device):
    image_tensors = processor(
        text=None,
        images=raw_images,
        return_tensors="pt"
    ).to(device)

    return image_tensors['pixel_values']


class ModifiedLlavaModel(nn.Module):
    def __init__(self, model_id: str, device: Optional[torch.device] = None):
        super().__init__()
        self.model_id = model_id
        self.device_override = device

        print("Loading original LLaVA model for config...")
        original_llava_model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            dtype=torch.float16, 
            device_map="auto"
        )

        if self.device_override:
            original_llava_model = original_llava_model.to(self.device_override)

        self.config = original_llava_model.config
        self.image_token_index = self.config.image_token_index
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 修正：確保 tokenizer 有 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set tokenizer.pad_token to eos_token ({self.tokenizer.eos_token})")
        # 結束修正

        self.llava_model = original_llava_model

        text_hidden_size = self.llava_model.config.text_config.hidden_size
        vision_config_object = self.config.vision_config

        print(f"Initializing custom vision components based on: {vision_config_object._name_or_path}")

        self.vision_tower = CustomVisionTower(config=vision_config_object)
        vision_hidden_size = self.vision_tower.hidden_size

        self.multi_modal_projector = CustomProjector(
            vision_hidden_size=vision_hidden_size,
            text_hidden_size=text_hidden_size
        )

        self.vision_tower.to(device=self.device, dtype=self.dtype)
        self.multi_modal_projector.to(device=self.device, dtype=self.dtype)

        print("Attempting to copy weights into CustomVisionTower ...")
        try:
            base_vision = self.llava_model.vision_tower
            self.vision_tower.load_from_pretrained(base_vision)
        except Exception as e:
            print("Error while loading vision tower weights:", e)

        print("Attempting to copy weights into CustomProjector ...")
        try:
            if hasattr(self.llava_model, "multi_modal_projector"):
                proj_state = self.llava_model.multi_modal_projector.state_dict()
                self.multi_modal_projector.load_state_dict(proj_state, strict=False)
                print("Loaded projector weights (strict=False).")
            else:
                print("original_llava_model has no attribute multi_modal_projector.")
        except Exception as e:
            print("Error while loading projector weights:", e)


        print("Monkey-patching original model with custom components...")
        self.llava_model.vision_tower = self.vision_tower
        self.llava_model.multi_modal_projector = self.multi_modal_projector
        
        if hasattr(self.llava_model, "language_model"):
            for param in self.llava_model.language_model.parameters():
                param.requires_grad = False
        else:
            print("Warning: Could not find language_model to freeze.")
            
        print("\nModifiedLlavaModel initialization complete.")

    @property
    def device(self):
        if self.device_override is not None:
            return self.device_override
        return self.llava_model.device

    @property
    def dtype(self):
        return self.llava_model.dtype

    def prepare_inputs_for_llm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None
    ):
        
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        image_features, _, _ = self.vision_tower(pixel_values, output_attentions=False)
        image_embeds = self.multi_modal_projector(image_features)
        
        # 修正：將 NaN 檢查改為拋出 ValueError
        if torch.isnan(image_embeds).any():
            raise ValueError("Error: Image embeddings contain NaN values!")
            
        num_dynamic_patches = image_embeds.shape[1]
        text_embeds = self.llava_model.get_input_embeddings()(input_ids)
        batch_size, original_seq_len, embed_dim = text_embeds.shape
        new_seq_len = -1
        final_inputs_embeds = []
        final_attention_mask = []

        for b in range(batch_size):
            image_token_indices = torch.where(input_ids[b] == self.image_token_index)[0]
            num_image_tokens_original = len(image_token_indices)
            if num_image_tokens_original > 0:
                first_image_token_pos = image_token_indices[0]
                current_new_seq_len = original_seq_len - num_image_tokens_original + num_dynamic_patches
                if new_seq_len == -1:
                    new_seq_len = current_new_seq_len
                else:
                    assert new_seq_len == current_new_seq_len, "Batch 中的序列長度不一致"

                part_A_text = text_embeds[b, :first_image_token_pos]
                part_B_image = image_embeds[b]
                part_C_text = text_embeds[b, first_image_token_pos + num_image_tokens_original :]

                b_inputs_embeds = torch.cat([part_A_text, part_B_image, part_C_text], dim=0)
                final_inputs_embeds.append(b_inputs_embeds)

                if attention_mask is not None:
                    mask_A = attention_mask[b, :first_image_token_pos]
                    mask_B = torch.ones(num_dynamic_patches, dtype=attention_mask.dtype, device=self.device)
                    mask_C = attention_mask[b, first_image_token_pos + num_image_tokens_original :]

                    b_attention_mask = torch.cat([mask_A, mask_B, mask_C], dim=0)
                    final_attention_mask.append(b_attention_mask)
            else:
                if new_seq_len == -1:
                    new_seq_len = original_seq_len
                final_inputs_embeds.append(text_embeds[b])
                if attention_mask is not None:
                    final_attention_mask.append(attention_mask[b])

        inputs_embeds = torch.stack(final_inputs_embeds, dim=0)
        new_attention_mask = None
        if attention_mask is not None:
            new_attention_mask = torch.stack(final_attention_mask, dim=0)

        return inputs_embeds, new_attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        inputs_embeds, new_attention_mask = self.prepare_inputs_for_llm(
            input_ids, pixel_values, attention_mask
        )

        new_seq_len = inputs_embeds.shape[1]

        if labels is not None:
            new_labels = torch.full((input_ids.shape[0], new_seq_len), -100, dtype=torch.long, device=self.device)
            for b in range(input_ids.shape[0]):
                image_token_indices = torch.where(input_ids[b] == self.image_token_index)[0]
                num_image_tokens_original = len(image_token_indices)

                if num_image_tokens_original > 0:
                    first_image_token_pos = image_token_indices[0]
                    original_seq_len = input_ids.shape[1]
                    num_dynamic_patches = new_seq_len - (original_seq_len - num_image_tokens_original)
                    new_labels[b, :first_image_token_pos] = labels[b, :first_image_token_pos]
                    new_labels[b, first_image_token_pos : first_image_token_pos + num_dynamic_patches] = -100
                    new_labels[b, first_image_token_pos + num_dynamic_patches :] = labels[b, first_image_token_pos + num_image_tokens_original :]
                else:
                    new_labels[b] = labels[b]
        else:
            new_labels = None

        outputs = self.llava_model(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            return_dict=True
        )
        return outputs

    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        **generate_kwargs
    ):
        
        # 1. 準備自定義的 inputs_embeds
        inputs_embeds, new_attention_mask = self.prepare_inputs_for_llm(
            input_ids, pixel_values, attention_mask
        )
        
        # 2. 確保模型在評估模式
        self.llava_model.eval()

        
        
        if 'pad_token_id' in generate_kwargs:
            del generate_kwargs['pad_token_id']
            
        generate_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        
        if 'eos_token_id' not in generate_kwargs:
              generate_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
              
        if 'do_sample' not in generate_kwargs:
            generate_kwargs['do_sample'] = True
            
        if 'temperature' not in generate_kwargs:
            generate_kwargs['temperature'] = 0.7
            
        if 'max_new_tokens' not in generate_kwargs:
            generate_kwargs['max_new_tokens'] = 128
            
        # 【新增修正】設定最小生成長度，強制模型輸出
        if 'min_new_tokens' not in generate_kwargs:
            generate_kwargs['min_new_tokens'] = 1  # 至少生成 1 個 token
        
        
        outputs = self.llava_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            **generate_kwargs
        )
        print("--- [Generate Debug Check END] ----------------")
        return outputs