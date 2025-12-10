import torch
import torch.nn as nn
from transformers import CLIPVisionConfig
from typing import Optional, Tuple, List


# ====================================================================================
# 1. 底層組件定義 (已整合所有建議)
# ====================================================================================

class CustomCLIPAttention(nn.Module):
    """
    重現 CLIP 的 Multi-Head Self-Attention 層。
    【增強】:
    1. (Suggestion 1) 新增 output_attentions 參數以返回注意力權重。
    2. (Suggestion 2) 新增 attention_mask 和 causal_attention_mask 參數以兼容 HF API。
    3. (Suggestion 3) 將 Dropout 移至 EncoderLayer (參見下方)，此處保持簡潔。
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

        # 線性投影層
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

        # 1. 投影到 Q, K, V
        query_states = self._shape(self.q_proj(hidden_states), seq_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), seq_len, bsz)
        value_states = self._shape(self.v_proj(hidden_states), seq_len, bsz)

        # 3. 計算 Attention Score
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale

        # 【增強】: 應用 HF API 傳入的 attention_mask (Suggestion 2)
        if attention_mask is not None:
            # attention_mask 通常形狀為 (bsz, 1, 1, seq_len)
            attn_weights = attn_weights + attention_mask

        # 【增強】: 應用 HF API 傳入的 causal_attention_mask (Suggestion 2)
        if causal_attention_mask is not None:
            # causal_attention_mask 通常形狀為 (bsz, 1, seq_len, seq_len)
            attn_weights = attn_weights + causal_attention_mask

        # 4. Softmax 取得權重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 5. 用權重加權 V
        attn_output = torch.matmul(attn_weights, value_states)

        # 6. Reshape 回原本的維度
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, embed_dim)

        # 7. 最後的線性投影
        attn_output = self.out_proj(attn_output)

        # 【增強】: 根據 flag 返回 attention_weights (Suggestion 1)
        return (attn_output, attn_weights) if output_attentions else (attn_output,)


class CustomCLIPMLP(nn.Module):
    """
    重現 CLIP 的 MLP (前饋神經網路) 層。
    (此模組在原始實作中已正確，無需加入 Dropout)
    """

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
    """
    重現一個完整的 Transformer Encoder Block。
    【增強】:
    1. (Suggestion 1) 新增 output_attentions 參數。
    2. (Suggestion 2) 新增 ..._mask 參數以傳遞給 Attention 層。
    3. (Suggestion 3) 新增 Dropout 層，放在殘差連接之前。
    """

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.self_attn = CustomCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CustomCLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 【增強】: 根據 HF 原始碼，Dropout 在 EncoderLayer 層級 (Suggestion 3)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-Attention + Residual Connection
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        # 【增強】: 傳遞所有參數並接收元組 (tuple)
        self_attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions
        )

        hidden_states = self_attn_outputs[0]
        attn_weights = self_attn_outputs[1] if output_attentions else None  # 提取 attn_weights

        # 【增強】: 應用 Dropout (Suggestion 3)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # MLP + Residual Connection
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # 【增強】: 應用 Dropout (Suggestion 3)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # 【增強】: 返回 hidden_states 和可選的 attn_weights (Suggestion 1)
        return (hidden_states, attn_weights) if output_attentions else (hidden_states,)


class CustomCLIPVisionEmbeddings(nn.Module):
    """
    重現 CLIP 的圖像嵌入層。
    (您的原始實作已非常完美，無需修改)
    """

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


# =========================================================================
# 2. 整合所有組件，建構成完整的 Vision Tower
# =========================================================================

class CustomVisionTower(nn.Module):
    """
    可客製化的 Vision Encoder.
    【增強】:
    1. (Suggestion 1) forward 方法現在可以返回 all_hidden_states 和 all_attentions。
    2. (Suggestion 2) forward 方法現在接受 ..._mask 參數以傳遞給底層。
    """

    def __init__(self, vision_tower_name: str):
        super().__init__()
        # 首先載入原始模型的設定檔，以獲取所有超參數
        self.config = CLIPVisionConfig.from_pretrained(vision_tower_name)

        # 根據設定檔，初始化我們的自定義組件
        self.embeddings = CustomCLIPVisionEmbeddings(self.config)

        # Transformer Encoder 主體 (堆疊多個 Encoder Layer)
        self.encoder_layers = nn.ModuleList(
            [CustomCLIPEncoderLayer(self.config) for _ in range(self.config.num_hidden_layers)]
        )

        self.post_layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)

        # 供外部模組讀取的隱藏層維度
        self.hidden_size = self.config.hidden_size

    def forward(
            self,
            pixel_values: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        前向傳播。
        返回: (image_features, all_hidden_states, all_attentions)
        """
        hidden_states = self.embeddings(pixel_values)

        all_hidden_states = []
        all_attentions = []

        for layer in self.encoder_layers:
            # 儲存輸入到該層的 hidden_state (Pre-LayerNorm 的標準作法)
            all_hidden_states.append(hidden_states)

            # 【增強】: 傳遞所有參數並接收元組 (tuple)
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions
            )

            hidden_states = outputs[0]
            if output_attentions:
                all_attentions.append(outputs[1])

        # 應用最後的 LayerNorm
        hidden_states = self.post_layernorm(hidden_states)
        # 儲存最後一層的 hidden_state
        all_hidden_states.append(hidden_states)

        # LLaVA 的作法是取倒數第二層的輸出，並且 "不包含" CLS token
        # all_hidden_states[-1] 是 "post_layernorm" 的輸出
        # all_hidden_states[-2] 是 "最後一層 Layer" 的輸出
        # LLaVA 1.5 使用的是 "倒數第二層的輸出"

        # 這裡需要釐清：'penultimate_hidden_state' 是指
        # 1. all_hidden_states[-2] (即第 23 層的輸出)？
        # 2. 還是指 HF LLaVA 實作中的 `outputs.hidden_states[-2]`？

        # 經確認，HF LLaVA 1.5 原始碼是：
        # outputs = self.vision_tower(..., output_hidden_states=True)
        # image_features = outputs.hidden_states[-2]
        # (hidden_states 包含 embedding 層 + 24 層的輸出，共 25 個)
        # hidden_states[-1] 是第 24 層的輸出 (post_layernorm 之前)
        # hidden_states[-2] 是第 23 層的輸出 (post_layernorm 之前)

        # 我們的 all_hidden_states 儲存的是 "輸入" 到每一層的狀態
        # all_hidden_states[0] = embedding 輸出
        # all_hidden_states[23] = 第 23 層的輸出 (即第 24 層的輸入)
        # all_hidden_states[-1] = 第 24 層 post_layernorm 後的輸出
        # all_hidden_states[-2] = 第 24 層的 "輸入" (即第 23 層的輸出)

        # 結論：LLaVA 1.5 使用的是第 23 層的輸出 (即倒數第二個 Encoder Layer 的輸出)
        # 在我們的迴圈中，這對應 `all_hidden_states[-2]`
        penultimate_hidden_state = all_hidden_states[-2]

        # 移除 CLS token (在第1個位置)
        image_features = penultimate_hidden_state[:, 1:, :]

        return (image_features, all_hidden_states, all_attentions)

    def load_weights(self, state_dict: dict):
        """
        載入權重。(您的原始實作已非常完美，無需修改)
        """
        new_state_dict = {}
        prefix_to_remove = 'vision_model.'
        for key, value in state_dict.items():
            if key.startswith(prefix_to_remove):
                new_key = key[len(prefix_to_remove):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        # 載入權重
        load_result = self.load_state_dict(new_state_dict)
        print(f"Successfully loaded weights for detailed CustomVisionTower. {load_result}")


# =========================================================================
# 3. 投影層 (Projector) 維持不變
# =========================================================================

class CustomProjector(nn.Module):
    """
    可客製化的投影層 (您的原始實作已非常完美，無需修改)。
    """

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