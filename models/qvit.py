# custom_vision_parts.py (Final Optimized & Corrected Version)

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig
from typing import Optional, Tuple


# ====================================================================================
# 1. 底層組件定義
# ====================================================================================

class CustomCLIPAttention(nn.Module):
    """
    重現 CLIP 的 Multi-Head Self-Attention 層。
    修正版：移除布林標誌，透過 kv_states 是否存在來判斷 Cross-Attention。
    """

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

    ## 修正與優化：修正 TypeError 警告的根源 ##
    def forward(self, hidden_states: torch.Tensor, kv_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)

        if kv_states is not None:
            key_states, value_states = kv_states
            kv_len = key_states.shape[1]
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            kv_len = q_len

        query_states_reshaped = self._shape(query_states, q_len, bsz)
        key_states_reshaped = self._shape(key_states, kv_len, bsz)
        value_states_reshaped = self._shape(value_states, kv_len, bsz)

        attn_weights = torch.matmul(query_states_reshaped, key_states_reshaped.transpose(-1, -2)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states_reshaped)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, key_states, value_states


class CustomCLIPMLP(nn.Module):
    # ... (此 class 維持不變)
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CustomCLIPEncoderLayer(nn.Module):
    # ... (此 class 維持不變)
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.self_attn = CustomCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CustomCLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states_norm = self.layer_norm1(hidden_states)
        attn_output, key, value = self.self_attn(hidden_states_norm)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, key, value


class CustomCLIPVisionEmbeddings(nn.Module):
    # ... (此 class 維持不變)
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels, out_channels=config.hidden_size,
                                         kernel_size=config.patch_size, stride=config.patch_size, bias=False)
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))
        self.class_embedding = nn.Parameter(torch.randn(config.hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        bs = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        cls_embeds = self.class_embedding.expand(bs, 1, -1)
        embeddings = torch.cat([cls_embeds, patch_embeds], dim=1) + self.position_embedding(self.position_ids)
        return embeddings


class CustomVisionTower(nn.Module):
    # ... (此 class 維持不變，但依賴於上面修正過的子模組)
    def __init__(self, vision_tower_name: str):
        super().__init__()
        self.config = CLIPVisionConfig.from_pretrained(vision_tower_name)
        self.embeddings = CustomCLIPVisionEmbeddings(self.config)
        self.encoder_layers = nn.ModuleList(
            [CustomCLIPEncoderLayer(self.config) for _ in range(self.config.num_hidden_layers)])
        self.hidden_size = self.config.hidden_size
        self.learnable_queries = nn.Parameter(torch.randn(1, self.embeddings.num_patches, self.hidden_size))

    def forward(self, pixel_values: torch.Tensor, output_layer_features: int = -1) -> torch.Tensor:
        bs = pixel_values.shape[0]
        image_hidden_states = self.embeddings(pixel_values)
        query_hidden_states = self.learnable_queries.expand(bs, -1, -1)
        num_layers = len(self.encoder_layers)
        query_start_layer_index = num_layers - 6
        all_image_features = []
        for i, layer in enumerate(self.encoder_layers):
            image_hidden_states_output, image_k, image_v = layer(image_hidden_states)
            all_image_features.append(image_hidden_states_output)
            if i >= query_start_layer_index:
                q_input_norm = layer.layer_norm1(query_hidden_states)
                attn_output, _, _ = layer.self_attn(hidden_states=q_input_norm, kv_states=(image_k, image_v))
                query_hidden_states = query_hidden_states + attn_output
                mlp_output = layer.mlp(layer.layer_norm2(query_hidden_states))
                query_hidden_states = query_hidden_states + mlp_output
            image_hidden_states = image_hidden_states_output
        if output_layer_features > -1:
            return all_image_features[output_layer_features][:, 1:, :]
        return query_hidden_states

    def load_weights(self, state_dict: dict):
        new_state_dict = {}
        prefix = 'vision_model.'
        for k, v in state_dict.items(): new_state_dict[k[len(prefix):] if k.startswith(prefix) else k] = v
        self.load_state_dict(new_state_dict, strict=False)
        print("Successfully loaded weights for CustomVisionTower. 'learnable_queries' is newly initialized.")


class CustomProjector(nn.Module):
    # ... (此 class 維持不變)
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