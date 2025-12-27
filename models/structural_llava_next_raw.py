import torch
import torch.nn as nn
import numpy as np
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, CLIPVisionModel
import warnings
import torch.nn.functional as F
from peft import PeftModel
import os

# 忽略 LLaVA 載入時的 dtype 棄用警告
warnings.filterwarnings("ignore", category=FutureWarning, message="`torch_dtype` is deprecated! Use `dtype` instead!")

# ==========================================
# 1. 幾何與訊號處理工具
# ==========================================

class GeometricUtils:
    @staticmethod
    def calculate_centroid(polygon_points: torch.Tensor) -> torch.Tensor:
        polygon_points = polygon_points.float()
        x = polygon_points[..., 0]
        y = polygon_points[..., 1]
        x_next = torch.roll(x, shifts=-1, dims=-1)
        y_next = torch.roll(y, shifts=-1, dims=-1)
        cross_term = x * y_next - x_next * y
        two_area = torch.sum(cross_term, dim=-1) 
        
        epsilon = 1e-3
        # 避免面積為 0
        area_safe = torch.where(torch.abs(two_area) < epsilon, torch.ones_like(two_area) * epsilon, two_area)
        
        factor = 1.0 / (3.0 * area_safe)
        cx = factor * torch.sum((x + x_next) * cross_term, dim=-1)
        cy = factor * torch.sum((y + y_next) * cross_term, dim=-1)
        
        return torch.stack([cx, cy], dim=-1)

    @staticmethod
    def fourier_shape_encoding(polygon_points: torch.Tensor, num_harmonics: int = 24) -> torch.Tensor:
        polygon_points = polygon_points.float()
        B, N, _ = polygon_points.shape
        device = polygon_points.device
        
        z = torch.complex(polygon_points[..., 0], polygon_points[..., 1])
        
        required_len = num_harmonics + 1
        
        if N < required_len:
            pad_len = required_len - N
            z = torch.cat([z, torch.zeros(B, pad_len, device=device, dtype=z.dtype)], dim=-1)
            N = z.shape[-1]

        Z = torch.fft.fft(z, dim=-1)
        
        indices = torch.arange(1, num_harmonics + 1, device=device)
        Z_subset = Z[:, indices]
        
        Z_subset_norm = Z_subset / N
        
        Z_1 = Z[:, 1]
        phase_1 = torch.angle(Z_1).unsqueeze(-1)
        
        phi_k = torch.angle(Z_subset_norm)
        k_vec = indices.unsqueeze(0).float()
        
        phase_correction = k_vec * phase_1
        I_k = phi_k - phase_correction
        
        magnitude = torch.abs(Z_subset_norm)
        Z_feat = torch.polar(magnitude, I_k)
        
        features = torch.cat([Z_feat.real, Z_feat.imag], dim=-1)
        features = torch.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
        return features

# ==========================================
# 2. 神經網路組件
# ==========================================

class ShapeFeatureProjector(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=1024, output_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    def forward(self, x): return self.mlp(x)

class CentroidRoPE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, token_embeddings, centroids):
        cx, cy = centroids[:, 0], centroids[:, 1]
        half_dim = self.dim // 2
        
        freqs_x = torch.einsum("i,j->ij", cx, self.inv_freq[:half_dim//2])
        freqs_y = torch.einsum("i,j->ij", cy, self.inv_freq[:half_dim//2])
        emb_x = torch.cat((freqs_x, freqs_x), dim=-1)
        emb_y = torch.cat((freqs_y, freqs_y), dim=-1)
        theta = torch.cat((emb_x, emb_y), dim=-1)
        
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        return token_embeddings * torch.cos(theta) + rotate_half(token_embeddings) * torch.sin(theta)

class StructuralCrossAttention(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=16, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 1),
                nn.GELU(),
                nn.Linear(hidden_dim * 1, hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2+1)])

    def forward(self, visual_tokens, object_tokens):
        x = visual_tokens
        object_tokens = self.norms[0](object_tokens)
        for i, layer in enumerate(self.layers):
            residual = x
            attn_out, _ = layer(query=x, key=object_tokens, value=object_tokens)
            x = residual + attn_out
            x = self.norms[i*2+1](x)
            
            residual = x
            x = residual + self.ffns[i](x)
            x = self.norms[i*2 + 2](x)
        return x

# ==========================================
# 3. 主模型 (整合 Florence-2 與 LLaVA)
# ==========================================

class HybirdLlavaFlorenceModel(nn.Module):
    def __init__(self, 
                 llava_model_id="llava-hf/llava-v1.6-mistral-7b-hf",
                 florence_model_id="./models/local_florence2",
                 freeze_llava=True,
                 load_llava=True,       # 新增控制開關
                 load_florence=True,
                 max_new_token=200):
        super().__init__()
        
        self.embedding_dim = 4096
        self.vision_dim = 1024
        self.max_new_token = max_new_token
        
        # 1. 載入 LLaVA
        if load_llava:
            print(f"Loading LLaVA from {llava_model_id}...")
            vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
            pos_embeddings = vision_tower.vision_model.embeddings.position_embedding.weight
            self.register_buffer("vit_pos_embed", pos_embeddings[1:].detach())
            """
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                )
            
            self.llava = LlavaNextForConditionalGeneration.from_pretrained(
                llava_model_id,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
                device_map="auto" 
                )
            """
            
            self.llava = LlavaNextForConditionalGeneration.from_pretrained(
                llava_model_id, 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True,
                device_map="auto" 
            )
            
            self.llava_processor = AutoProcessor.from_pretrained(llava_model_id)
            pretrained_path = "./pretrained_projector/label_down_projector_autoencoder.bin"

            
            # 載入權重
            # 注意：如果維度不對 (例如 4096->1024 寫反)，這裡會報錯，剛好當作檢查
            self.label_down_projector = nn.Sequential(
                nn.LayerNorm(self.embedding_dim, eps=1e-6),
                nn.Linear(self.embedding_dim, self.vision_dim)
                )
            if os.path.exists(pretrained_path):
                print(f"Loading pretrained label_down_projector from {pretrained_path}...")
                try:
                    state_dict = torch.load(pretrained_path, map_location=self.llava.device)
                    self.label_down_projector.load_state_dict(state_dict)
                    print("Successfully loaded pretrained projector weights.")
                except Exception as e:
                    print(f"[Warning] Failed to load weights: {e}. Using Random Init.")
            else:
                print(f"[Info] No pretrained projector found at {pretrained_path}.")
                print(">>> Using RANDOM INITIALIZATION for label_down_projector (End-to-End Training).")
                
            self.label_down_projector = self.label_down_projector.to(self.llava.device)
                
            # Freeze LLaVA
            self.llava.requires_grad_(False)
            if hasattr(self.llava.language_model, 'final_logits_bias') and self.llava.language_model.final_logits_bias is not None:
                self.llava.language_model.final_logits_bias = self.llava.language_model.final_logits_bias.to('cpu')
        else:
            self.llava = None
            self.llava_processor = None

        # 2. 載入 Florence-2
        if load_florence:
            print(f"Loading Florence-2 from {florence_model_id}...")
            # Florence 通常不大，可以自動分配或放在特定 GPU
            self.florence = AutoModelForCausalLM.from_pretrained(
                florence_model_id, 
                torch_dtype=torch.float16, 
                trust_remote_code=True,
            ).eval()
            self.florence_processor = AutoProcessor.from_pretrained(florence_model_id, trust_remote_code=True)
            self.florence.requires_grad_(False)
            
            # 手動移到 GPU (如果沒用 device_map)
            if torch.cuda.is_available():
                self.florence = self.florence.to("cuda")
        else:
            self.florence = None
            self.florence_processor = None

        # 3. 新增組件
        self.shape_projector = ShapeFeatureProjector(
            input_dim=48, 
            output_dim=self.vision_dim
        )
        if self.florence_processor is not None:
             self.pad_token_id = self.florence_processor.tokenizer.pad_token_id
        self.rope_2d = CentroidRoPE(dim=self.vision_dim)
        self.adapter = StructuralCrossAttention(
            hidden_dim=self.vision_dim,
            num_heads=16
        )
        self.geo_utils = GeometricUtils()
        self.dummy_grad_input = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.output_norm = nn.LayerNorm(self.vision_dim)
        
        # =================================================================
        # 【修正】判斷 target_device 的邏輯
        # =================================================================
        target_device = None
        
        if self.llava is not None:
            target_device = self.llava.device
        elif self.florence is not None:
            target_device = self.florence.device
        else:
            # 如果都沒載入 (理論上不該發生)，預設 CPU 或 CUDA
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # 手動移動自定義組件到目標設備
        if target_device.type == 'cuda':
            print(f"Manual device move for Adapters to {target_device}...")
            
            # 如果 Florence 存在且不在目標設備上，移動它
            if self.florence is not None:
                self.florence = self.florence.to(target_device)
            
            self.output_norm = self.output_norm.to(target_device)
            
            self.shape_projector = self.shape_projector.to(target_device)
            self.rope_2d = self.rope_2d.to(target_device)
            self.adapter = self.adapter.to(target_device)

        if self.llava_processor is not None:
            self.image_token_id = self.llava_processor.tokenizer.convert_tokens_to_ids("<image>")
            if self.image_token_id is None:
                 self.image_token_id = 32000
        else:
            self.image_token_id = 32000

    def get_object_text_embedding(self, label_text: str):
        """取得物件名稱的 Text Embedding"""
        inputs = self.llava_processor.tokenizer(label_text, return_tensors="pt").to(self.llava.device)
        with torch.no_grad():
            embed_layer = self.llava.get_input_embeddings()
            tokens = embed_layer(inputs.input_ids)
        return torch.mean(tokens, dim=1).to(torch.float16)
    

    def get_input_embeddings(self):
        """用於兼容 Hugging Face Trainer 內部機制。"""
        return self.llava.get_input_embeddings()

    def enable_input_require_grads(self):
        """
        強制 LLaVA 的輸入 Embedding 追蹤梯度，以避免 Checkpointing 警告。
        這會修改 LLM embedding 層的行為。
        """
        def make_inputs_require_grad(module, input, output):
            # 確保輸出結果追蹤梯度
            output.requires_grad_(True)
        
        # 覆寫 LLM 的 input embedding 層的 forward hook
        # 注意: 這裡的 get_input_embeddings 是 LLM 的文本 Embedding 層
        self.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        # 確保 Vision Tower 輸出也追蹤梯度 (如果 Vision Tower 參與 Checkpointing)
        # 由於 Vision Tower 是 frozen 的，這個 Hook 可能會產生另一個 Warning，但我們需要嘗試
        # if hasattr(self.llava, 'vision_tower'):
        #     self.llava.vision_tower.register_forward_hook(make_inputs_require_grad)
        
        # 如果 LLaVA 模型內建 enable_input_require_grads，則呼叫它
        if hasattr(self.llava, 'enable_input_require_grads'):
            try:
                self.llava.enable_input_require_grads()
            except Exception as e:
                print(f"Warning: Failed to call internal enable_input_require_grads: {e}")


    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Trainer 會呼叫這個函數來開啟省顯存模式。
        我們將其轉發給 LLaVA 主模型。
        """
        if self.llava is not None:
            self.llava.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        

    def run_florence_inference(self, image):
        """
        第一階段：負責跑 Florence-2 產生 Label 和 Polygon 座標
        【已修改】加入信心分數過濾機制 (Confidence Filtering)
        """
        if self.florence is None:
            raise RuntimeError("Florence-2 model is not loaded.")

        device = self.florence.device
        dtype = self.florence.dtype

        # 準備回傳結構
        result = {
            "labels": [],
            "polygons": [] 
        }

        with torch.no_grad():
            # ==========================================
            # Step 1: Object Detection (<OD>) with Scores
            # ==========================================
            task_od = '<OD>'
            inputs = self.florence_processor(text=task_od, images=image, return_tensors="pt").to(device, dtype)
            
            # 啟用 output_scores 與 return_dict_in_generate
            outputs = self.florence.generate(
                input_ids=inputs["input_ids"], 
                pixel_values=inputs["pixel_values"], 
                max_new_tokens=256, 
                do_sample=False,
                num_beams=3
            )
            
            # 取得生成的文字
            if hasattr(outputs, "sequences"):
                generated_ids = outputs.sequences
            else:
                generated_ids = outputs
            generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            # 標準解析 (取得所有偵測到的 Labels)
            res_od = self.florence_processor.post_process_generation(generated_text, task=task_od, image_size=(image.width, image.height))
            od_data = res_od.get('<OD>', {})
            raw_labels = od_data.get('labels', [])

            if not raw_labels:
                return result
            
            filtered_labels = list(set(raw_labels))

            if not filtered_labels:
                return result

            # ==========================================
            # Step 3: Segmentation (只跑過濾後的標籤)
            # ==========================================
            task_seg = '<REFERRING_EXPRESSION_SEGMENTATION>'
            BATCH_SIZE = 5
            
            for i in range(0, len(filtered_labels), BATCH_SIZE):
                batch_labels = filtered_labels[i : i + BATCH_SIZE]
                batch_prompts = [task_seg + label for label in batch_labels]
                batch_images = [image for _ in batch_labels]

                inputs_seg = self.florence_processor(
                    text=batch_prompts, images=batch_images, return_tensors="pt", padding=True
                ).to(device, dtype)

                ids_seg = self.florence.generate(
                    input_ids=inputs_seg["input_ids"], pixel_values=inputs_seg["pixel_values"], max_new_tokens=1024, do_sample=False
                )
                
                texts_seg = self.florence_processor.batch_decode(ids_seg, skip_special_tokens=False)
                
                for j, label in enumerate(batch_labels):
                    res_seg_single = self.florence_processor.post_process_generation(
                        texts_seg[j], task=task_seg, image_size=(image.width, image.height)
                    )
                    seg_data = res_seg_single.get(task_seg, {})
                    if 'polygons' in seg_data:
                        poly_raw = seg_data['polygons']
                        # 處理多重多邊形 (Nested Lists)
                        for p in poly_raw:
                            if len(p) > 0 and isinstance(p[0], (list, np.ndarray, torch.Tensor)):
                                for sub_p in p:
                                    if len(sub_p) >= 6:
                                        result['labels'].append(label)
                                        result['polygons'].append(sub_p)
                            else:
                                if len(p) >= 6:
                                    result['labels'].append(label)
                                    result['polygons'].append(p)
                del inputs_seg, ids_seg
                                    
            return result

    def encode_structural_features(self, labels, fourier_features, centroids):
        """
        第二階段：將 Fourier 特徵 + 標籤文字 -> 最終 Embedding
        Training 階段在 forward 中呼叫這個。
        """
        fourier_features = torch.nan_to_num(fourier_features, nan=0.0)
        is_nan_mask = torch.isnan(centroids).any(dim=-1) # Shape: (N_objs,)
        is_valid_mask = ~is_nan_mask # 有效的位置標記為 True
        centroids = torch.nan_to_num(centroids, nan=0.0)
        device = self.shape_projector.mlp[0].weight.device
        dtype = self.shape_projector.mlp[0].weight.dtype
        
        # 確保輸入在正確的 device
        fourier_features = fourier_features.to(device, dtype=dtype)
        centroids = centroids.to(device, dtype=dtype)
        
        tokens_list = []

        shape_emb_fp32 = self.shape_projector(fourier_features) # (N, 1024)
        
        # 3. [關鍵] 輸出 Clamp 並轉回 System Dtype (通常是 FP16)
        # 這是為了防止 Shape Projector 初始化不好導致數值過大
        
        # 取得目標 dtype (通常是 float16)
        target_dtype = self.llava.dtype 
        shape_emb = shape_emb_fp32.to(dtype=target_dtype)
        
        # 因為 labels 是 list of strings，且 batch 裡每個 sample 物件數不同，
        # 這裡通常是一個 sample 的處理迴圈 (如果是 batch processing 需要改寫，這裡假設輸入是一個 sample 的所有物件)
        # 為了支援 Batch Forward，這裡我們假設輸入的 fourier_features 是 (N_objs, 48)
        
        if len(labels) == 0:
             return torch.zeros(1, 1, self.vision_dim, device=device, dtype=dtype)

        
        # 計算 Text Embeddings (這一步比較慢，可以優化成 batch)
        # 這裡我們將所有 unique label 收集起來一次編碼，然後 gather 回來
        # 簡單起見，先逐個處理 (推論時物件不會太多)
        text_embs = []
        for label in labels:
            text_embs.append(self.get_object_text_embedding(label))
        
        text_emb_tensor = torch.cat(text_embs, dim=0) # (N_objs, 4096)
        down_proj_dtype = self.label_down_projector[1].weight.dtype
        
        # 轉型後再輸入
        input_to_proj = text_emb_tensor.to(dtype=down_proj_dtype)
        out = self.label_down_projector(input_to_proj)
        
        # 如果需要數值穩定，可以在運算完後轉回 fp32 做加法，或者直接用當前 dtype
        
        text_feat = out.to(dtype=target_dtype)
        
        final_tokens = text_feat + shape_emb
        grid_size = 24
        # 使用 clamp 防止極端值超出 23
        grid_x = (centroids[..., 0] * grid_size).long().clamp(0, grid_size - 1)
        grid_y = (centroids[..., 1] * grid_size).long().clamp(0, grid_size - 1)
        
        # 2. 計算 1D 索引 (Row-Major: y * 24 + x)
        pos_indices = grid_y * grid_size + grid_x  # Shape: (N_objs,)
        
        # 3. 取出對應的 Embedding
        vit_pos_feat = F.embedding(pos_indices, self.vit_pos_embed)
        
        # 4. 轉型並相加
        vit_pos_feat = vit_pos_feat.to(dtype=final_tokens.dtype)
        final_tokens = final_tokens + (vit_pos_feat * is_valid_mask.unsqueeze(-1))
        final_tokens = final_tokens.to(self.llava.dtype)
        
        return final_tokens.unsqueeze(0) # (1, N_objs, 1024) - Batch size 1 for this context

    def extract_structural_context(self, image):
        """
        整合函式：Inference 時使用 (End-to-End)
        """
        # 1. Florence 偵測
        raw_data = self.run_florence_inference(image)
        labels = raw_data['labels']
        polygons = raw_data['polygons']
        
        if not labels:
            return torch.zeros(1, 1, self.embedding_dim, device=self.llava.device, dtype=self.llava.dtype)
            
        # 2. 幾何運算 (Math) - 這裡現場算
        fourier_list = []
        centroid_list = []
        valid_labels = []
        
        device = self.llava.device
        img_w, img_h = image.size
        
        for idx, poly_list in enumerate(polygons):
            try:
                target_poly = torch.tensor(poly_list, dtype=torch.float16, device=device).reshape(-1, 2)
                target_poly = target_poly.unsqueeze(0)
                
                shape_feat = self.geo_utils.fourier_shape_encoding(target_poly, num_harmonics=24)
                """
                if shape_feat.norm(p=2, dim=-1).max() > 0:
                    shape_feat = shape_feat / (shape_feat.norm(p=2, dim=-1, keepdim=True) + 1e-6)
                """
                centroid = self.geo_utils.calculate_centroid(target_poly)
                norm_centroid = centroid.clone()
                norm_centroid[..., 0] = norm_centroid[..., 0] / img_w
                norm_centroid[..., 1] = norm_centroid[..., 1] / img_h
                
                # 簡單的邊界檢查 (Clamp 到 0~1)
                norm_centroid = torch.clamp(norm_centroid, 0.0, 1.0)
                
                if torch.isnan(shape_feat).any() or torch.isinf(shape_feat).any(): continue
                if torch.all(shape_feat == 0): continue
                
                fourier_list.append(shape_feat.squeeze(0))
                centroid_list.append(norm_centroid.squeeze(0))
                valid_labels.append(labels[idx])
            except:
                continue
                
        if not valid_labels:
            return torch.zeros(1, 1, self.embedding_dim, device=device, dtype=self.llava.dtype)

        fourier_tensor = torch.stack(fourier_list)
        centroid_tensor = torch.stack(centroid_list)
        
        # 3. 編碼 (Projector + RoPE)
        return self.encode_structural_features(valid_labels, fourier_tensor, centroid_tensor)

    def merge_embeddings(self, input_ids, vision_embeds, text_embeds):
        batch_size = input_ids.shape[0]
        final_embeds_list = []
        
        for i in range(batch_size):
            image_inds = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
            
            if len(image_inds) == 0:
                final_embeds_list.append(text_embeds[i])
                continue
                
            idx = image_inds[0]
            
            combined = torch.cat([
                text_embeds[i, :idx], 
                vision_embeds[i], 
                text_embeds[i, idx+1:]
            ], dim=0)
            
            final_embeds_list.append(combined)
            
        return torch.stack(final_embeds_list)

    # (forward 和 generate_answer 保持原樣，此處略過以節省篇幅)

    def forward(self, input_ids=None, pixel_values=None, image_sizes=None, attention_mask=None, labels=None, 
                # 新增參數：支援訓練時傳入預處理好的幾何特徵
                structural_fourier=None, structural_centroids=None, structural_labels=None,
                # 舊參數兼容
                structural_kv=None, images=None):
        
        # 1. 決定 structural_kv 的來源
        # Case A: 已經傳入了最終的 kv tensor (極少用，除非測試)
        if structural_kv is not None:
            pass # 已經有了
        # Case B: 訓練模式 - 傳入 Raw Fourier Features + Labels
        elif structural_fourier is not None and structural_labels is not None:
            # 這裡需要對 Batch 裡的每個 sample 進行 encode_structural_features
            # structural_fourier: (B, N_max, 48)
            # structural_labels: List of List of Strings (Batch size)
            
            batch_kv_list = []
            batch_size = structural_fourier.shape[0]
            
            for i in range(batch_size):
                # 拿出有效的數據 (去除 Padding，假設 Dataset 端做了 padding 補 0)
                # 簡單判斷：如果是全 0 的 row 視為 padding
                curr_fourier = structural_fourier[i]
                curr_centroids = structural_centroids[i]
                curr_labels = structural_labels[i] # 這是 list
                
                # 這裡有個長度匹配問題，Dataset colllate 會 pad tensor，但 labels list 長度可能不一致
                # 我們只取前 len(curr_labels) 個 tensor
                num_valid = len(curr_labels)
                if num_valid == 0:
                    # 萬一空的
                    kv = torch.zeros(1, self.vision_dim, device=self.llava.device, dtype=self.llava.dtype)
                else:
                    valid_fourier = curr_fourier[:num_valid]
                    valid_centroids = curr_centroids[:num_valid]
                    # 呼叫編碼器 (Projector + RoPE)
                    kv = self.encode_structural_features(curr_labels, valid_fourier, valid_centroids)
                    kv = kv.squeeze(0) # encode 回傳 (1, N, D), 這裡要 (N, D)
                    if self.training and (torch.isnan(kv).any() or kv.abs().max() > 10000):
                        print(f"[Model Debug] kv_after_encode explodes! Max: {kv.abs().max()}")
                
                batch_kv_list.append(kv)
            
            # Pad 這裡產生的 batch_kv_list 以便堆疊
            # 但因為後面的 Adapter 是 CrossAttention，KV 長度可以不一致嗎？
            # PyTorch MHA 支援 key_padding_mask，但這裡為了簡化，我們 pad 成 tensor
            max_len = max([k.shape[0] for k in batch_kv_list])
            padded_kvs = []
            for k in batch_kv_list:
                pad_len = max_len - k.shape[0]
                if pad_len > 0:
                    pad = torch.zeros(pad_len, self.vision_dim, device=k.device, dtype=k.dtype)
                    k = torch.cat([k, pad], dim=0)
                padded_kvs.append(k)
            structural_kv = torch.stack(padded_kvs) # (B, N_max, 4096)
            if self.training and (torch.isnan(structural_kv).any() or structural_kv.abs().max() > 10000):
                print(f"[Model Debug] kv_preprocess explodes! Max: {structural_kv.abs().max()}")

        # Case C: 推論模式 - 傳入 raw images，現場跑 Florence
        elif images is not None:
            str_kv_list = []
            if isinstance(images, list):
                for img in images:
                    str_kv_list.append(self.extract_structural_context(img))
                structural_kv = torch.cat(str_kv_list, dim=0)
            else:
                structural_kv = self.extract_structural_context(images)
            if self.training and (torch.isnan(structural_kv).any() or structural_kv.abs().max() > 10000):
                print(f"[Model Debug] kv_inference explodes! Max: {structural_kv.abs().max()}")
        
        else:
            # 什麼都沒有，回傳空的
             structural_kv = None
        
        # 1. 拆解 5D 維度 (Batch, Patches, Channels, Height, Width)
        b, n_patches, c, h, w = pixel_values.shape 
        
        # 2. 攤平為 4D (Batch * Patches, C, H, W) 讓 CLIP 讀取
        flat_pixel_values = pixel_values.view(b * n_patches, c, h, w)
        
        # 3. 取得特徵
        with torch.no_grad():
            vision_outputs = self.llava.vision_tower(flat_pixel_values, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]

        b, n_patches, c, h, w = pixel_values.shape
        # image_features 目前是 (b*n_patches, num_tokens, 1024)
        # 我們先不經過 multi_modal_projector

        # 注意：這裡的 reshape 邏輯取決於 LLaVA 版本，如果是 LLaVA-1.6 (LLaVA-NeXT)，
        # 它的 image_features 處理比較複雜 (有 unpad, reshape 等)，
        # 為了簡化，我們假設這裡是標準的特徵提取：
        
        # 為了讓代碼能跑，我們需要模擬 LLaVA 內部取出 feature 的行為，但不做 projection
        # 簡單做法：直接對 image_features 做 Adapter
        
        if self.training:
            image_features.requires_grad_(True)

        # 3. 在 1024 維度上做 Structural Adapter
        if structural_kv is not None:
            # structural_kv 必須是 (B, N_objs, 1024)
            # image_features 必須是 (B*N_patches, Num_tokens, 1024) -> 需要調整 batch 對齊
            
            # 注意：這裡有個 batch size 對齊問題
            # pixel_values 是 (Batch, N_patches, C, H, W)，經過 vision tower 後變成 (Batch * N_patches, ...)
            # 但 structural_kv 是 (Batch, ...)
            # 我們需要把 structural_kv 擴展來對齊 image_features
            
            # 擴展 structural_kv: (B, N_obj, 1024) -> (B, N_patches, N_obj, 1024) -> (B*N_patches, N_obj, 1024)
            structural_kv_expanded = structural_kv.unsqueeze(1).repeat(1, n_patches, 1, 1)
            structural_kv_expanded = structural_kv_expanded.view(b * n_patches, -1, self.vision_dim)
            if self.training and (torch.isnan(structural_kv_expanded).any() or structural_kv_expanded.abs().max() > 10000):
                print(f"[Model Debug] kv Output explodes! Max: {structural_kv_expanded.abs().max()}")
            if self.training and (torch.isnan(image_features).any() or image_features.abs().max() > 10000):
                print(f"[Model Debug] image_feature Output explodes! Max: {image_features.abs().max()}")
            
            
             
            enhanced_features = self.adapter(image_features, structural_kv_expanded)
        else:
            enhanced_features = image_features

        # 4. 現在才通過原本的 Projector (1024 -> 4096)
        # LLaVA 的 projector 通常是 self.llava.multi_modal_projector

        enhanced_features = self.output_norm(enhanced_features)
        enhanced_features = enhanced_features.to(self.llava.dtype)

        vision_tokens_flat = self.llava.multi_modal_projector(enhanced_features)

        # 5. Reshape 為最終輸入 (Batch, Total_Tokens, 4096)
        enhanced_vision_tokens = vision_tokens_flat.view(b, -1, vision_tokens_flat.shape[-1])
        

        # 4. Text Embeddings
        with torch.enable_grad():
            inputs_embeds = self.llava.get_input_embeddings()(input_ids)
            
            # 5. Merge
            merged_embeds = self.merge_embeddings(input_ids, enhanced_vision_tokens, inputs_embeds)
            
        
        
        # 6. Mask Logic (同前)
        new_mask_list = []
        new_labels_list = []
        batch_size = input_ids.shape[0]
        vision_len = enhanced_vision_tokens.shape[1]

        
        for i in range(batch_size):
            image_inds = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(image_inds) == 0:
                new_mask_list.append(attention_mask[i])
                if labels is not None: new_labels_list.append(labels[i])
                continue
            
            idx = image_inds[0]
            vis_mask = torch.ones(vision_len, device=attention_mask.device, dtype=attention_mask.dtype)
            new_mask = torch.cat([attention_mask[i, :idx], vis_mask, attention_mask[i, idx+1:]])
            new_mask_list.append(new_mask)
            
            if labels is not None:
                vis_labels = torch.ones(vision_len, device=labels.device, dtype=labels.dtype) * -100
                new_label = torch.cat([labels[i, :idx], vis_labels, labels[i, idx+1:]])
                new_labels_list.append(new_label)

        final_attention_mask = torch.stack(new_mask_list)
        final_labels = torch.stack(new_labels_list) if labels is not None else None

        # 7. LLM Forward
        outputs = self.llava.language_model(
            inputs_embeds=merged_embeds,
            attention_mask=final_attention_mask,
            labels=final_labels,
            return_dict=True
        )

        if not hasattr(outputs, "loss") or outputs.loss is None:
            # A. 提取 Hidden States (BaseModelOutputWithPast 通常把 hidden state 放在第一位或 .last_hidden_state)
            if hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs[0]

            # B. 尋找 LM Head (通常在 language_model.lm_head)
            if hasattr(self.llava.language_model, "lm_head"):
                lm_head = self.llava.language_model.lm_head
            elif hasattr(self.llava, "lm_head"):
                lm_head = self.llava.lm_head
            else:
                # 如果找不到 head，這是一個嚴重錯誤，但通常 mistral 結構都有 .lm_head
                raise AttributeError("Could not find 'lm_head' to compute logits!")

            # C. 手動計算 Logits
            logits = lm_head(hidden_states)
            logits = logits.float() # 確保轉回 float32 防止溢出

            # D. 手動計算 Loss (CrossEntropy)
            loss = None
            if final_labels is not None:
                # Shift logits and labels (Causal LM 標準操作: 預測下一個 token)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = final_labels[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss()
                # Flatten tokens
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # 【雙重保險】加入 dummy grad 以解決 "None of the inputs have requires_grad" 問題
                if self.training:
                    loss = loss + 0.0 * self.adapter.parameters().__next__().sum()
                    loss = loss + 0.0 * self.shape_projector.mlp[0].weight.sum()

            return {"loss": loss, "logits": logits}

        # 如果模型正常回傳 (Fallback)
        return {"loss": outputs.loss, "logits": outputs.logits}
    

    def generate_answer(self, image, question):
        """
        Inference 入口: 自動處理所有流程
        """

        """
        save_dir="./final_adapter"
        print(f"Loading fusion weights from {save_dir}...")
        custom_path = os.path.join(save_dir, "custom_modules.bin")
        state_dict = torch.load(custom_path, map_location=self.llava.device)
        
        self.adapter.load_state_dict(state_dict["adapter"])
        self.shape_projector.load_state_dict(state_dict["shape_projector"])
        self.label_down_projector.load_state_dict(state_dict["label_down_projector"])
        lora_path = os.path.join(save_dir, "llava_projector_lora")
        self.llava = PeftModel.from_pretrained(self.llava, lora_path)
        
        """
        
        # 1. 準備輸入
        # LLaVA-1.6 Prompt 格式: "USER: <image>\nQuestion ASSISTANT:"
        if "<image>" not in question:
            prompt = f"[INST] <image>\n{question} [/INST]"
        else:
            prompt = f"[INST] {question} [/INST]"
            
        inputs = self.llava_processor(text=prompt, images=image, return_tensors="pt").to(self.llava.device)
        
        # 2. 呼叫 Forward (不傳 Labels, 傳 raw images 以觸發 Florence)
        # 這裡我們不直接調用 self.forward，因為 generate 需要 KV cache 等複雜機制
        # 我們必須手動做 Embedding 替換然後 call model.generate
        
        with torch.no_grad():
            # A. 取得 Structural KV (1, N_objs, 1024)
            structural_kv = self.extract_structural_context(image)
            
            # B. 取得 Vision Features (Raw from Vision Tower)
            pixel_values = inputs.pixel_values
            b, n_patches, c, h, w = pixel_values.shape
            
            # 1. 攤平輸入給 Vision Tower
            flat_pixel_values = pixel_values.view(b * n_patches, c, h, w)
            
            # 2. 取得 Vision Hidden States (尚未 Project)
            # Output: (Batch * N_patches, Num_Tokens, 1024)
            vis_out = self.llava.vision_tower(flat_pixel_values, output_hidden_states=True)
            vis_feat_raw = vis_out.hidden_states[-2] 
            del vis_out
            
            # C. Early Fusion Adapter (在 1024 維度進行)
            # 關鍵：對齊 Batch Size
            # vis_feat_raw 是 (B * N_patches, ...)
            # structural_kv 是 (B, ...) 也就是 (1, ...)
            # 我們需要把 structural_kv 複製 N_patches 份
            
            if structural_kv is not None:
                # Expand: (1, N_objs, 1024) -> (1, N_patches, N_objs, 1024)
                structural_kv_expanded = structural_kv.unsqueeze(1).repeat(1, n_patches, 1, 1)
                # Flatten: -> (N_patches, N_objs, 1024)
                structural_kv_expanded = structural_kv_expanded.view(b * n_patches, -1, self.vision_dim)
                adapter_dtype = self.adapter.layers[0].in_proj_weight.dtype
                
                # 確保輸入符合 adapter dtype
                vis_feat_raw = vis_feat_raw.to(dtype=adapter_dtype)
                structural_kv_expanded = structural_kv_expanded.to(dtype=adapter_dtype)
                
                # Adapter Attention
                enhanced_vis_raw = self.adapter(vis_feat_raw, structural_kv_expanded)
            else:
                enhanced_vis_raw = vis_feat_raw

            # D. Projector (1024 -> 4096)
            # 現在才過 LLaVA 原本的 Projector
            vis_feat_flat = self.llava.multi_modal_projector(enhanced_vis_raw)
            
            # E. 重塑回 (Batch, Total_Tokens, 4096)
            enhanced_vis = vis_feat_flat.view(b, -1, vis_feat_flat.shape[-1])
            
            # D. Merge with Prompt
            txt_emb = self.llava.get_input_embeddings()(inputs.input_ids)
            merged_emb = self.merge_embeddings(inputs.input_ids, enhanced_vis, txt_emb)
            
            # E. Generate
            # 注意: generate 不接受 inputs_embeds 作為主要輸入的同時自動處理 attention mask 擴展
            # 我們需要手動擴展 attention mask (同 forward 邏輯)
            
            batch_size = inputs.input_ids.shape[0]
            vision_len = enhanced_vis.shape[1]
            idx = (inputs.input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0][0]
            
            vis_mask = torch.ones(vision_len, device=inputs.attention_mask.device, dtype=inputs.attention_mask.dtype)
            final_mask = torch.cat([inputs.attention_mask[0, :idx], vis_mask, inputs.attention_mask[0, idx+1:]]).unsqueeze(0)
            
            out_ids = self.llava.generate(
                inputs_embeds=merged_emb,
                attention_mask=final_mask,
                max_new_tokens=self.max_new_token,
                use_cache=True,
                pad_token_id=self.llava_processor.tokenizer.pad_token_id,
                eos_token_id=self.llava_processor.tokenizer.eos_token_id
            )
            
        return self.llava_processor.decode(out_ids[0], skip_special_tokens=True)