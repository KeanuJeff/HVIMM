# SHARP: Packing Invariant Shape Harmonics for Robust Visual Reasoning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/xxxx.xxxxx)

This repository contains the official PyTorch implementation of the paper **"SHARP: Packing Invariant Shape Harmonics for Robust Visual Reasoning"**.

## 🌟 Abstract

Existing Multimodal Large Language Models (MLLMs) predominantly rely on patch-based tokenization, a process that inevitably disrupts continuous geometric boundaries and diminishes object-centric integrity. To address this "Structural Amnesia," we propose **SHARP (Shape Harmonics for Accurate Reasoning and Perception)**, a novel framework that explicitly "packs" geometric priors into the visual latent space.

At the core of our method is **Fourier Shape Encoding (FSE)**, a mathematically rigorous descriptor that maps object contours to the frequency domain, demonstrating invariance to scale, translation, sampling density, and starting point selection. By synthesizing these spectral features with semantic and positional embeddings via a cross-attention mechanism, SHARP restores the topological continuity of fragmented visual tokens without altering the pre-trained backbone. Evaluations across five benchmarks, including IconQA, demonstrate that our method significantly reduces object hallucination and improves geometric reasoning accuracy by **15.8%**.

<div align="center">
  <img src="dataset/architecture.jpg" width="800px" /> <br>
  <em>Overview of the SHARP architecture: integrating semantic context, geometric grounding encoding, and composite structural prior fusion.</em>
</div>

## 📢 News
* **[2026/01/28]** Released training and evaluation code.
* **[TBD]** Pre-trained weights and datasets will be released soon.

## 🛠️ Installation

1. **Clone the repository**
    ```bash
    git clone [[https://github.com/xxxxx](https://github.com/xxxxx)]
    cd HVIMM
    ```

2. **Create environment**
    ```bash
    conda create -n sharp python=3.10 -y
    conda activate sharp
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## 📂 Data Preparation

Please follow the steps below to download and preprocess the datasets. Place original images in the `dataset/` directory.

We provide preprocessing scripts to generate the necessary annotations and mask data (using Florence-2):

| Dataset | Preprocessing Script | Description |
| :--- | :--- | :--- |
| **LLaVA-Instruct** | `python preprocess_llava_instruct.py` | Processes instruction tuning data |
| **RefCOCOg** | `python preprocess_refcocog.py` | Processes grounding data |
| **ShareGPT4V** | `python preprocess_sharegpt4v.py` | Processes high-detail caption data |
| **TextCaps** | `python preprocess_textcaps.py` | Processes OCR-related data |
| **Florence-2 Fix** | `python fix_florence.py` | Used for fixing or completing mask generation |

Example data directory structure:
