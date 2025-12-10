import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from transformers import LlavaNextForConditionalGeneration, AutoProcessor, AutoModelForCausalLM

# ==========================================
# 1. Signal Processing Utilities
# ==========================================

class GeometricUtils:
    """
    Utilities for geometric calculations and Fourier transforms
    """
    
    @staticmethod
    def calculate_centroid(polygon_points: torch.Tensor) -> torch.Tensor:
        """
        根據圖片三的公式計算多邊形質心 (Cx, Cy)
        
        Args:
            polygon_points: 形狀為 (B, N, 2) 的 Tensor，代表 B 個物件，每個物件 N 個座標點 (x, y)。
                           注意：假設點是按順序排列的 (Florence-2 輸出通常是有序的)。
                           如果是變長序列，建議先 padding 或針對單個樣本處理。
        Returns:
            centroids: 形狀為 (B, 2) 的 Tensor，包含 (Cx, Cy)
        """
        x = polygon_points[..., 0]
        y = polygon_points[..., 1]
        
        # 為了計算 x_i * y_{i+1}，我們需要將陣列 shift
        # x_{i+1}, y_{i+1} (最後一點連接回第一點)
        x_next = torch.roll(x, shifts=-1, dims=-1)
        y_next = torch.roll(y, shifts=-1, dims=-1)
        
        # 計算共通項: (x_i * y_{i+1} - x_{i+1} * y_i)
        cross_term = x * y_next - x_next * y
        
        # 計算面積 A (Signed Area) * 2
        # A = 0.5 * sum(cross_term) -> 2A = sum(cross_term)
        two_area = torch.sum(cross_term, dim=-1) # (B,)
        
        # 避免面積為 0 (例如退化成線或點)
        epsilon = 1e-6
        # 使用 sign 保持方向性，或取絕對值視具體座標系定義而定，通常取 Signed Area
        area_safe = torch.where(torch.abs(two_area) < epsilon, torch.ones_like(two_area) * epsilon, two_area)
        
        factor = 1.0 / (3.0 * area_safe) # 公式是 1/(6A)，因為 two_area 是 2A，所以除以 3 * (2A)
        
        # 計算 Cx = (1/6A) * sum((x_i + x_{i+1}) * cross_term)
        cx = factor * torch.sum((x + x_next) * cross_term, dim=-1)
        
        # 計算 Cy = (1/6A) * sum((y_i + y_{i+1}) * cross_term)
        cy = factor * torch.sum((y + y_next) * cross_term, dim=-1)
        
        return torch.stack([cx, cy], dim=-1) # (B, 2)

    @staticmethod
    def fourier_shape_encoding(polygon_points: torch.Tensor, num_harmonics: int = 24) -> torch.Tensor:
        """
        將多邊形轉換為傅立葉描述子特徵
        
        Args:
            polygon_points: (B, N, 2)
            num_harmonics: 取前 k 個頻率 (k=1~num_harmonics)
            
        Returns:
            features: (B, num_harmonics * 2) -> 展平的實部與虛部
        """
        B, N, _ = polygon_points.shape
        
        # 1. 轉為複數 z = x + iy
        # PyTorch 的 complex 支援
        z = torch.complex(polygon_points[..., 0], polygon_points[..., 1]) # (B, N)
        
        # 2. Discrete Fourier Transform
        # Z_k: (B, N)
        Z = torch.fft.fft(z, dim=-1)
        
        # 3. Normalization logic
        # 為了取 k=1 ~ k=24，我們需要處理頻譜索引
        # fft 輸出索引通常是 [0, 1, ..., N/2, -N/2, ..., -1]
        # 我們只取正頻率部分 1 到 num_harmonics
        
        if N <= num_harmonics:
            # 如果點數比要求的頻率少，這是個邊界情況，通常需要先插值 (Resample)
            # 這裡簡單處理：補零或報錯，實際應用建議在輸入前做 Resample
            raise ValueError(f"Polygon points N={N} is less than required harmonics {num_harmonics}. Please resample.")

        # 取 k=1 到 k=num_harmonics
        indices = torch.arange(1, num_harmonics + 1, device=polygon_points.device)
        Z_subset = Z[:, indices] # (B, K)
        
        # 取 k=1 的振幅與相位用於正規化
        Z_1 = Z[:, 1] # (B,) 基頻
        amp_1 = torch.abs(Z_1).unsqueeze(-1) # (B, 1) |Z_1|
        phase_1 = torch.angle(Z_1).unsqueeze(-1) # (B, 1) phi_1
        
        # 3.1 大小正規化 (保留大小資訊? User 說: "希望保留形狀大小，所以不要對基頻(k=1)振幅作正規化")
        # User: "Zk都能再去除以總座標點數量來抵銷取樣頻率不固定的問題"
        # 所以我們只除以 N
        Z_subset_norm = Z_subset / N
        
        # 3.2 相位修正 (解決起點問題)
        # I(k) = phase(k) - k * phase(1)
        # 取得目前的相位 phi_k
        phi_k = torch.angle(Z_subset_norm) # (B, K)
        
        # 計算 k * phi_1
        k_vec = indices.unsqueeze(0).float() # (1, K) -> [1, 2, ..., 24]
        phase_correction = k_vec * phase_1 # (B, K)
        
        I_k = phi_k - phase_correction # (B, K)
        
        # 3.3 重組特徵
        # User: "將改變後的複數轉換回x+iy的形式"
        # 這裡指的是利用修正後的相位與原始振幅重建複數係數
        magnitude = torch.abs(Z_subset_norm) # (B, K)
        
        # 重建複數 Z_feat = |Z_k|/N * e^(i * I_k)
        Z_feat = torch.polar(magnitude, I_k) # (B, K)
        
        # 4. 展平為實數特徵向量 (Real, Imag)
        # 輸出維度: B x (K * 2) -> 48 dims
        features = torch.cat([Z_feat.real, Z_feat.imag], dim=-1)
        
        return features

# ==========================================
# 2. 神經網路組件 (Modules)
# ==========================================

class ShapeFeatureProjector(nn.Module):
    """
    負責將 48維的幾何特徵投影到 4096維
    """
    def __init__(self, input_dim=48, hidden_dim=1024, output_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(), # 或 ReLU
            nn.Linear(hidden_dim, output_dim),
            # 這裡可以選擇是否再加一個 LayerNorm
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)

class CentroidRoPE(nn.Module):
    """
    基於質心 (Cx, Cy) 的 2D Rotary Positional Embedding
    參考 Qwen-VL / ViT-2D-RoPE 的概念
    """
    def __init__(self, dim: int, max_pos: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_pos = max_pos # Florence-2 輸出通常 normalize 到 1000
        
        # 頻率計算 (Inverse frequency bands)
        # 將維度分成兩半，一半給 X，一半給 Y
        self.head_dim = dim // 2 # 假設輸入已經是每個 head 的維度，或者是總維度的一半
        # 這裡為了簡化，直接對整個 hidden_dim (4096) 做操作，實際應用可能是在 Multi-head Attention 內部做
        # 但 User 說 "直接相加"，意味著是在 Embedding 層級做 RoPE 後相加
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, token_embeddings: torch.Tensor, centroids: torch.Tensor):
        """
        Args:
            token_embeddings: (B, Dim) - 物件的文字 Embedding
            centroids: (B, 2) - (Cx, Cy)
        Returns:
            rotated_embeddings: (B, Dim)
        """
        cx = centroids[:, 0]
        cy = centroids[:, 1]
        
        # 準備頻率矩陣
        # 這裡採用簡單策略：前一半維度編碼 X，後一半維度編碼 Y
        half_dim = self.dim // 2
        
        # 生成旋轉角度 theta
        # 外積: (B, 1) @ (1, Dim/4) -> (B, Dim/4) 
        # 注意: RoPE 是對 pair (d, d+1) 旋轉，所以頻率長度是 dim/2
        
        # 針對 X 的角度
        freqs_x = torch.einsum("i,j->ij", cx, self.inv_freq[:half_dim//2])
        emb_x = torch.cat((freqs_x, freqs_x), dim=-1) # (B, half_dim)
        
        # 針對 Y 的角度
        freqs_y = torch.einsum("i,j->ij", cy, self.inv_freq[:half_dim//2])
        emb_y = torch.cat((freqs_y, freqs_y), dim=-1) # (B, half_dim)
        
        # 拼接 X 和 Y 的旋轉角度
        theta = torch.cat((emb_x, emb_y), dim=-1) # (B, Dim)
        
        # 應用旋轉 (標準 RoPE 實現)
        # [x1, x2, x3, x4...] -> [-x2, x1, -x4, x3...]
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        return token_embeddings * torch.cos(theta) + rotate_half(token_embeddings) * torch.sin(theta)

class StructuralCrossAttention(nn.Module):
    """
    兩層 Attention Layer
    Query: LLaVA Image Tokens (N_img, 4096)
    Key, Value: Object Tokens (N_obj, 4096)
    """
    def __init__(self, hidden_dim=4096, num_heads=32, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        # 每個 Attention 後通常接一個 FFN 和 Norm
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2)])

    def forward(self, visual_tokens, object_tokens):
        """
        Args:
            visual_tokens: (B, Seq_img, 4096) - 作為 Query
            object_tokens: (B, Seq_obj, 4096) - 作為 Key, Value
        """
        x = visual_tokens
        
        for i, layer in enumerate(self.layers):
            # Attention Block
            # residual connection
            residual = x
            x = self.norms[i*2](x)
            
            # Cross Attention: Q=Visual, K=Object, V=Object
            attn_out, _ = layer(query=x, key=object_tokens, value=object_tokens)
            x = residual + attn_out
            
            # FFN Block
            residual = x
            x = self.norms[i*2 + 1](x)
            x = residual + self.ffns[i](x)
            
        return x

# ==========================================
# 3. 主模型 Pipeline (The Main Class)
# ==========================================

class HybirdLlavaFlorenceModel(nn.Module):
    def __init__(self, 
                 llava_model_id="llava-hf/llava-v1.6-mistral-7b-hf",
                 florence_model_id="microsoft/Florence-2-large",
                 freeze_llava=True):
        super().__init__()
        
        print("Initializing Hybrid Model...")
        
        # 1. 載入 LLaVA (這部分需要大量顯存，這裡僅示意結構)
        # 在實際 Colab 中需注意 device_map="auto"
        self.llava = LlavaNextForConditionalGeneration.from_pretrained(
            llava_model_id, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.llava_processor = AutoProcessor.from_pretrained(llava_model_id)
        
        # 凍結 LLaVA
        if freeze_llava:
            for param in self.llava.parameters():
                param.requires_grad = False
            print("LLaVA weights frozen.")

        # 2. 載入 Florence-2 (用於推論獲取 Masks)
        self.florence = AutoModelForCausalLM.from_pretrained(
            florence_model_id, 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        self.florence_processor = AutoProcessor.from_pretrained(florence_model_id, trust_remote_code=True)
        # 凍結 Florence
        for param in self.florence.parameters():
            param.requires_grad = False

        # 3. 新增的組件 (Trainable Parts)
        self.embedding_dim = 4096 # Mistral 7B standard
        
        # 形狀投影器 (48 -> 1024 -> 4096)
        self.shape_projector = ShapeFeatureProjector(input_dim=48, output_dim=self.embedding_dim)
        
        # 2D RoPE
        self.rope_2d = CentroidRoPE(dim=self.embedding_dim)
        
        # Cross Attention Adapter
        self.adapter = StructuralCrossAttention(hidden_dim=self.embedding_dim)
        
        # 工具類別
        self.geo_utils = GeometricUtils()

    def get_object_text_embedding(self, label_text: str):
        """
        利用 LLaVA 的 embedding layer 將物件名稱轉為 Vector。
        注意：這裡簡化為取 label token 的平均或最後一個 token，
        實際上應該將 label token 序列作為整體。
        """
        inputs = self.llava_processor.tokenizer(label_text, return_tensors="pt").to(self.llava.device)
        with torch.no_grad():
            # 取得 Embedding Layer (通常是 model.model.embed_tokens)
            # 不同版本的 transformers 結構可能不同，需確認
            if hasattr(self.llava.language_model, 'model'):
                embed_layer = self.llava.language_model.model.embed_tokens
            else:
                embed_layer = self.llava.language_model.get_input_embeddings()
                
            tokens = embed_layer(inputs.input_ids) # (1, Seq, 4096)
            
        # 簡單起見，這裡做 Mean Pooling 得到代表該物件的單一向量 (1, 4096)
        # 更好的做法是保留序列長度，但這會讓後續計算變複雜
        return torch.mean(tokens, dim=1) 

    def forward(self, input_ids, attention_mask, pixel_values, structural_kv, labels=None):
        """
        支援 Trainer 訓練的 Forward 函數
        """
        # 1. 取得 LLaVA 原始 Vision Features
        with torch.no_grad():
            image_outputs = self.llava.vision_tower(pixel_values, output_hidden_states=True)
            selected_image_feature = image_outputs.hidden_states[-2]
            vision_tokens = self.llava.multi_modal_projector(selected_image_feature)

        # 2. 注入結構化特徵 (Trainable Part)
        # 這裡 structural_kv 是從 Dataset 傳進來的 (B, N_obj, 4096)
        enhanced_vision_tokens = self.adapter(vision_tokens, structural_kv)

        # 3. 準備 LLM 輸入
        # 獲取 Text Embeddings
        # 這裡簡化處理：假設 input_ids 已經包含了 prompt + labels
        # 實際 LLaVA 處理會更複雜（要處理 <image> token 替換），建議參考 LLaVA 官方的 prepare_inputs_labels_for_multimodal
        # 為了跑通 Demo，我們做最簡單的拼接： [Image, Text]
        
        inputs_embeds = self.llava.language_model.get_input_embeddings()(input_ids)
        
        # 簡單拼接：先把 Image 放前面
        # 注意：這會導致 positional embedding 偏移，正規做法需要重算 position_ids
        # 但對於 Adapter 微調，模型通常能適應
        combined_embeds = torch.cat([enhanced_vision_tokens, inputs_embeds], dim=1)
        
        # 調整 Labels 與 Attention Mask 以匹配拼接後的長度
        # Image 部分的 Label 設為 -100 (不計算 Loss)
        if labels is not None:
            image_labels = torch.ones((labels.shape[0], enhanced_vision_tokens.shape[1]), dtype=torch.long).to(labels.device) * -100
            combined_labels = torch.cat([image_labels, labels], dim=1)
            
            image_mask = torch.ones((attention_mask.shape[0], enhanced_vision_tokens.shape[1]), dtype=torch.long).to(attention_mask.device)
            combined_mask = torch.cat([image_mask, attention_mask], dim=1)
        
        # 4. 輸入 LLM 計算 Loss
        outputs = self.llava.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask if labels is not None else None,
            labels=combined_labels if labels is not None else None,
            return_dict=True
        )
        
        # Trainer 需要回傳 (loss, logits) 或 dict
        return {"loss": outputs.loss, "logits": outputs.logits}
    
    def forward_pipeline(self, image, text_prompt):
        """
        完整的推論/訓練流程
        image: PIL Image
        text_prompt: User question
        """
        device = self.llava.device
        
        # --- Step 1: Florence-2 提取物件與 Masks ---
        # 這裡簡化呼叫流程，假設已封裝好上面你提供的 detect_and_segment_all
        # 實際整合需將 detect_and_segment_all 放入此 class 或外部呼叫
        # 假設我們得到 objects_data: List[Dict] -> [{'label': 'car', 'polygons': [[x,y...]]}, ...]
        
        # 模擬 Florence 輸出 (為了讓 code 可執行，這裡寫死邏輯，實際要跑真正的 inference)
        # 注意: 訓練時通常是 pre-compute 好的 mask 資訊
        print("Step 1: Running Florence-2 for object detection...")
        # objects_data = run_florence_inference(self.florence, self.florence_processor, image)
        
        # 模擬數據 (B=1, 2 objects)
        mock_polygons = [torch.randn(100, 2).abs() * 1000 for _ in range(2)] # 兩個物件，各100點
        mock_labels = ["person", "bicycle"]
        
        object_tokens_list = []
        
        for i, (poly, label) in enumerate(zip(mock_polygons, mock_labels)):
            poly = poly.to(device) # (N, 2)
            
            # --- Step 2: 文字 Token 處理 ---
            # 取得原始文字 Embedding
            text_emb = self.get_object_text_embedding(label) # (1, 4096)
            
            # --- Step 3: 計算幾何資訊 ---
            # 計算質心
            centroid = self.geo_utils.calculate_centroid(poly.unsqueeze(0)) # (1, 2)
            
            # 計算形狀特徵 (Fourier)
            # 需要先 resample 到固定點數，這裡假設已處理
            shape_feat = self.geo_utils.fourier_shape_encoding(poly.unsqueeze(0), num_harmonics=24) # (1, 48)
            
            # --- Step 4: 編碼與融合 ---
            # 4.1 位置編碼 (2D RoPE on Text Token)
            
            # 4.2 形狀編碼 (Projection)
            shape_emb = self.shape_projector(shape_feat) # (1, 4096)
            
            # 4.3 融合 (直接相加)
            # Resulting Structural Token
            text_emb = text_emb + shape_emb # (1, 4096)

            combined_token = self.rope_2d(text_emb, centroid)

            object_tokens_list.append(combined_token)
            
        # 堆疊所有物件 tokens -> (1, Num_Objects, 4096)
        if len(object_tokens_list) > 0:
            structural_kv = torch.cat(object_tokens_list, dim=1)
        else:
            # 如果沒偵測到物件，用全零或 dummy token 避免 crash
            structural_kv = torch.zeros(1, 1, self.embedding_dim).to(device)

        # --- Step 5: LLaVA Vision Encoding (Query) ---
        inputs = self.llava_processor(text=text_prompt, images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # 提取 Image Features
            # 呼叫 vision_tower
            pixel_values = inputs.pixel_values
            image_outputs = self.llava.vision_tower(pixel_values, output_hidden_states=True)
            selected_image_feature = image_outputs.hidden_states[-2] # LLaVA 通常取倒數第二層
            
            # 經過 LLaVA 的 Projector 映射到 LLM 維度
            vision_tokens = self.llava.multi_modal_projector(selected_image_feature) # (B, Seq_img, 4096)

        # --- Step 6: Cross Attention (Injecting Structure) ---
        # Q = Vision Tokens, K,V = Structural Tokens
        # 這部分是此架構的核心改良
        enhanced_vision_tokens = self.adapter(vision_tokens, structural_kv)
        
        # --- Step 7: 輸入 LLM 生成答案 ---
        # 這裡需要將 enhanced_vision_tokens 替換回原本的 embedding sequence 中
        # 因為 transformers 的 generate 介面封裝較深，通常需要 override model.forward
        # 或者使用 inputs_embeds 參數
        
        # 構建 inputs_embeds
        # 1. 取得 Prompt 的 text embeddings
        prompt_embeds = self.llava.language_model.get_input_embeddings()(inputs.input_ids)
        
        # 2. 拼接 (Vision + Text)
        # 注意：這裡省略了 LLaVA 複雜的 <image> token 替換邏輯，實作需參考 LLaVA source code
        # 簡單示意: [Enhanced Vision, Text Prompt]
        combined_embeds = torch.cat([enhanced_vision_tokens, prompt_embeds], dim=1)
        
        # 生成
        print("Step 7: Generating response from LLM...")
        # 這裡需要 attention_mask 對齊
        outputs = self.llava.language_model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=100
        )
        
        return self.llava_processor.decode(outputs[0], skip_special_tokens=True)

# ---------------------------------------------------------
# 使用範例 (Mock Execution)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Code syntax check passed.")
    # 實際執行需要 GPU 和下載模型，這裡僅展示架構定義
    # model = HybirdLlavaFlorenceModel()
    # output = model.forward_pipeline(image_pil, "Describe the image.")