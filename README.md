# News Image-Text Matching with News Knowledge Graph (Vietnamese Support)

This repository contains a **from-scratch implementation** of the paper:

> **News Image-Text Matching With News Knowledge Graph**  
> Zhao Yumeng, Yun Jing, Gao Shuo, Liu Limin  
> *IEEE Access, 2021*  
> [DOI: 10.1109/ACCESS.2021.3093650](https://doi.org/10.1109/ACCESS.2021.3093650)

Our implementation has been fully adapted to support the Vietnamese language. It includes a dedicated Vietnamese text pre-processing pipeline (e.g., tokenization, normalization) using popular Vietnamese NLP toolkits and customized evaluation metrics for Vietnamese news content.

---

## Features

- **End-to-End Implementation:**  
  Complete implementation of the news image-text matching method with a knowledge graph.

- **Vietnamese Language Support:**  
  - Vietnamese text pre-processing (tokenization, normalization, etc.) via [Underthesea](https://github.com/undertheseanlp/underthesea) or similar.
  - Customized data handling and evaluation for Vietnamese news articles.

- **Modular Architecture:**  
  - CNN-based feature extraction for images.
  - Text encoding (e.g., using Transformers) tailored for Vietnamese.
  - Knowledge Graph integration to bridge the gap between image and text modalities.

- **Training & Evaluation:**  
  Scripts for training the model on your dataset and evaluating matching performance.

---

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- NumPy
- stanza
- Additional packages listed in `requirements.txt`

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/DinhVitCuong/NKD-GNN-VN.git
cd news-image-text-matching-vn
pip install -r requirements.txt
