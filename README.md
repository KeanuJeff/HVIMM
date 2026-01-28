# SHARP: Packing Invariant Shape Harmonics for Robust Visual Reasoning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/xxxx.xxxxx) 這份儲存庫包含了論文 **"SHARP: Packing Invariant Shape Harmonics for Robust Visual Reasoning"** 的官方 PyTorch 實作。

## 🌟 摘要 (Abstract)

現有的多模態大型語言模型 (MLLMs) 主要依賴於 Patch-based 的標記化 (tokenization)，這個過程不可避免地破壞了連續的幾何邊界並削弱了以物件為中心的完整性。為了解決這種「結構性失憶 (Structural Amnesia)」，我們提出了 **SHARP (Shape Harmonics for Accurate Reasoning and Perception)**。這是一個新穎的框架，透過顯式地將幾何先驗「打包 (packing)」進視覺潛在空間中。

我們方法的核心是 **傅立葉形狀編碼 (Fourier Shape Encoding, FSE)**，這是一種數學上嚴謹的描述符，將物件輪廓映射到頻域，並證明了其對尺度、平移、採樣密度和初始點選擇的不變性。透過交叉注意力機制將這些頻譜特徵與語義和位置嵌入合成，SHARP 在不改變預訓練骨幹網路的情況下，恢復了破碎視覺 Token 的拓撲連續性。在 IconQA 等五個基準測試上的評估顯示，我們的方法顯著減少了物件幻覺 (Object Hallucination)，並將幾何推理準確率提高了 **15.8%**。

<div align="center">
  <img src="assets/framework.png" width="800px" /> <br>
  <em>SHARP 架構概覽：結合語義上下文、幾何接地編碼與複合結構先驗融合。</em>
</div>

## 📢 更新 (News)
* **[2026/01/28]** 釋出訓練與評估程式碼。
* **[TBD]** 將釋出預訓練權重與資料集。

## 🛠️ 安裝 (Installation)

1. **複製專案**
    ```bash
    git clone [https://github.com/KeanuJeff/HVIMM.git](https://github.com/KeanuJeff/HVIMM.git)
    cd HVIMM
    ```

2. **建立環境**
    ```bash
    conda create -n sharp python=3.10 -y
    conda activate sharp
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *(註：請確保您的環境中有安裝 PyTorch, Transformers 以及 LLaVA-NeXT 和 Florence-2 所需的依賴)*

## 📂 資料準備 (Data Preparation)

請依照下列步驟下載並預處理資料集。原始圖片請放置於 `dataset/` 目錄下。

我們提供了預處理腳本來生成所需的標註與 Mask 資料 (使用 Florence-2)：

| 資料集 | 預處理腳本 | 說明 |
| :--- | :--- | :--- |
| **LLaVA-Instruct** | `python preprocess_llava_instruct.py` | 處理指令微調資料 |
| **RefCOCOg** | `python preprocess_refcocog.py` | 處理 Grounding 資料 |
| **ShareGPT4V** | `python preprocess_sharegpt4v.py` | 處理高詳細度 Caption 資料 |
| **TextCaps** | `python preprocess_textcaps.py` | 處理 OCR 相關資料 |
| **Florence-2 Fix** | `python fix_florence.py` | 用於修正或補全 Mask 生成 |

資料目錄結構範例：
