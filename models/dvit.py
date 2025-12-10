# models/dvit.py
import torch
import torch.nn as nn
from transformers import CLIPVisionConfig
from typing import Optional, Tuple, List

# --- CustomCLIPAttention, CustomCLIPMLP, CustomCLIPEncoderLayer 保持不變 ---

class CustomCLIPAttention(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim {self.embed_dim} must be divisible by num_heads {self.num_heads}")

        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        bsz, seq_len, embed_dim = hidden_states.shape
        query_states = self._shape(self.q_proj(hidden_states), seq_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), seq_len, bsz)
        value_states = self._shape(self.v_proj(hidden_states), seq_len, bsz)
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        if causal_attention_mask is not None:
            attn_weights = attn_weights + causal_attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return (attn_output, attn_weights) if output_attentions else (attn_output,)


class CustomCLIPMLP(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CustomCLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.self_attn = CustomCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CustomCLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        dropout_rate = getattr(config, "dropout", 0.0)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        self_attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions
        )

        hidden_states = self_attn_outputs[0]
        attn_weights = self_attn_outputs[1] if output_attentions else None

        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, attn_weights) if output_attentions else (hidden_states,)


class CustomCLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches + 1  # +1 for [CLS] token
        self.position_embedding = nn.Embedding(self.num_positions, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))
        self.class_embedding = nn.Parameter(torch.randn(config.hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        cls_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([cls_embeds, patch_embeds], dim=1)
        position_embeds = self.position_embedding(self.position_ids)
        embeddings = embeddings + position_embeds
        return embeddings


class CustomVisionTower(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config

        self.embeddings = CustomCLIPVisionEmbeddings(self.config)
        self.encoder_layers = nn.ModuleList(
            [CustomCLIPEncoderLayer(self.config) for _ in range(self.config.num_hidden_layers)]
        )
        self.post_layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.hidden_size = self.config.hidden_size

        # 剪枝控制
        self.pruning_threshold = 0
        self.pruning_start_layer = 6
        self.pruning_end_layer = 10

    def forward(
            self,
            pixel_values: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None, # 傳入的 mask
            causal_attention_mask: Optional[torch.Tensor] = None, # 傳入的 mask
            output_attentions: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:

        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        hidden_states = self.embeddings(pixel_values)

        num_patches = self.embeddings.num_patches
        original_indices = torch.arange(num_patches + 1, device=device).unsqueeze(0).repeat(batch_size, 1)

        all_hidden_states = []
        all_attentions = []

        for i, layer in enumerate(self.encoder_layers):
            all_hidden_states.append(hidden_states)

            should_prune = self.pruning_start_layer <= i <= self.pruning_end_layer
            need_attentions_this_layer = output_attentions or should_prune

            outputs = layer(
                hidden_states,
                # 修正：不傳遞外部 attention_mask 給底層 layer，避免剪枝時的複雜錯誤
                attention_mask=None, 
                causal_attention_mask=None,
                output_attentions=need_attentions_this_layer
            )

            hidden_states = outputs[0]
            attn_weights = outputs[1] if need_attentions_this_layer else None

            if output_attentions:
                all_attentions.append(attn_weights)

            if should_prune:
                if attn_weights is None:
                    print(f"警告: Layer {i + 1} 未能獲取注意力，跳過剪枝。")
                    continue

                # 剪枝邏輯開始 (保持原樣)
                mean_att_map = attn_weights.mean(dim=1)
                patch_att_map = mean_att_map[..., 1:, 1:]
                received_attention = patch_att_map.sum(dim=1)

                min_score = received_attention.min(dim=-1, keepdim=True)[0]
                std_dev = received_attention.std(dim=-1, keepdim=True)
                threshold = min_score*0 + self.pruning_threshold * std_dev

                pruning_mask_1d = (received_attention >= threshold)

                cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
                full_mask = torch.cat([cls_mask, pruning_mask_1d], dim=1)

                hidden_states = hidden_states[
                    full_mask.unsqueeze(-1).expand_as(hidden_states)
                ].reshape(batch_size, -1, self.hidden_size)

                original_indices = original_indices[full_mask].reshape(batch_size, -1)

                # 修正：移除對 attention_mask/causal_attention_mask 的錯誤處理
                # if attention_mask is not None: ... (已移除)
                # if causal_attention_mask is not None: ... (已移除)
                # 剪枝時只處理 features 和 indices
                
        hidden_states = self.post_layernorm(hidden_states)
        all_hidden_states.append(hidden_states)
        
        # 修正：返回最終輸出，並移除 [CLS] token
        image_features = hidden_states[:, 1:, :] # 移除 [CLS] token (索引 0)

        return (image_features, all_hidden_states, all_attentions)

    # 權重載入邏輯保持不變
    def load_from_pretrained(self, base_vision_model: nn.Module):
        """
        將 base_vision_model（例如 original_llava_model.vision_tower）的權重載入到目前的 CustomVisionTower 中。
        """
        results = {}

        # 檢查 base_vision_model 是否為 LLaVA wrapper 
        inner_model = base_vision_model
        if hasattr(base_vision_model, "vision_model") and isinstance(base_vision_model.vision_model, nn.Module):
            inner_model = base_vision_model.vision_model
            print("Detected 'vision_model' wrapper, loading from inner_model.")
        else:
            print("Loading from base_vision_model directly.")

        # 1) embeddings
        try:
            res = self.embeddings.load_state_dict(inner_model.embeddings.state_dict(), strict=False)
            results['embeddings'] = res
        except Exception as e:
            results['embeddings_error'] = str(e)

        # 2) encoder layers (逐層載入)
        layer_results = {}
        
        base_layers = None
        if hasattr(inner_model, "encoder") and hasattr(inner_model.encoder, "layers"):
            base_layers = inner_model.encoder.layers
            print("Found layers in inner_model.encoder.layers")
        elif hasattr(inner_model, "encoder_layers"): # Fallback
            base_layers = inner_model.encoder_layers
            print("Found layers in inner_model.encoder_layers")
        elif hasattr(inner_model, "layers"): # Fallback 2
            base_layers = inner_model.layers

        if base_layers is None:
            layer_results['error'] = "cannot find encoder layers on inner_model"
        else:
            for i, my_layer in enumerate(self.encoder_layers):
                try:
                    base_layer = base_layers[i]
                    res = my_layer.load_state_dict(base_layer.state_dict(), strict=False)
                    layer_results[f"layer_{i}"] = res
                except Exception as e:
                    layer_results[f"layer_{i}_error"] = str(e)

        results['encoder_layers'] = layer_results

        # 3) post_layernorm
        try:
            res = self.post_layernorm.load_state_dict(inner_model.post_layernorm.state_dict(), strict=False)
            results['post_layernorm'] = res
        except Exception as e:
            results['post_layernorm_error'] = str(e)

        # 4) class_embedding
        try:
            if hasattr(inner_model.embeddings, "class_embedding") and hasattr(self.embeddings, "class_embedding"):
                with torch.no_grad():
                    self.embeddings.class_embedding.copy_(inner_model.embeddings.class_embedding.view_as(self.embeddings.class_embedding))
                results['class_embedding'] = "copied"
        except Exception as e:
            results['class_embedding_error'] = str(e)

        # 5) position embedding
        try:
            if hasattr(inner_model.embeddings, "position_embedding") and hasattr(self.embeddings, "position_embedding"):
                try:
                    res = self.embeddings.position_embedding.load_state_dict(inner_model.embeddings.position_embedding.state_dict(), strict=False)
                    results['position_embedding'] = res
                except Exception as e:
                    results['position_embedding_error'] = str(e)
        except Exception:
            pass

        print("CustomVisionTower.load_from_pretrained() results summary:")
        for k, v in results.items():
            print(f"  {k}: {v}")


class CustomProjector(nn.Module):
    # 投影層保持不變
    def __init__(self, vision_hidden_size: int, text_hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def load_weights(self, state_dict: dict):
        self.load_state_dict(state_dict)
        print("Successfully loaded weights for CustomProjector.")