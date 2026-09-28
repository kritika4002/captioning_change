# Automated Disaster Change Captioning using Vision-Language Models

> **M.Tech Major Project** — Generates natural language damage reports from bi-temporal satellite imagery using a fine-tuned Vision-Language Model.

## 🏗️ Architecture

```
Pre-Disaster Image  ─┐                              ┌─► Generated Caption:
                      ├─► CLIP ViT ─► Change Detector ─► GPT-2 (LoRA) ──┤  "Multiple buildings
Post-Disaster Image ─┘    (frozen)    (cross-attn)       (fine-tuned)     └─► sustained major
                                                                              structural damage
                                                                              from flooding..."
```

**Novel Contribution**: **Structural-Semantic Alignment Loss (SSAL)** — a composite loss function that ensures caption accuracy, structural damage consistency, and temporal coherence.

## 📁 Project Structure

```
captioning_change/
├── configs/config.yaml          # All hyperparameters
├── src/
│   ├── data/                    # Dataset + preprocessing
│   │   ├── dataset.py           # PyTorch Dataset (xBD)
│   │   ├── caption_generator.py # JSON → natural language
│   │   └── preprocessing.py     # Image transforms
│   ├── models/                  # Architecture
│   │   ├── change_captioner.py  # End-to-end model
│   │   ├── vision_encoder.py    # CLIP + cross-attention
│   │   └── losses.py            # SSAL loss (NOVEL)
│   ├── training/trainer.py      # Training loop
│   └── evaluation/metrics.py    # BLEU, METEOR, ROUGE-L
├── scripts/
│   ├── dataset_stats.py         # EDA & visualisations
│   ├── prepare_captions.py      # Pre-generate captions
│   └── train.py                 # Training entry point
└── dataset/                     # xBD data (gitignored)
```

## 🚀 Quick Start

```bash
# Activate virtual environment
C:\project-major\tf_gpu_env\Scripts\activate

# Generate dataset statistics (for PPT)
python scripts/dataset_stats.py --data-root ./dataset

# Pre-generate captions from labels
python scripts/prepare_captions.py --data-root ./dataset

# Start training
python scripts/train.py --config configs/config.yaml
```

## 📊 Evaluation Metrics

| Metric  | Description                        |
|---------|------------------------------------|
| BLEU-4  | N-gram precision (1–4 grams)      |
| METEOR  | Unigram matching with stemming     |
| ROUGE-L | Longest common subsequence F1      |
| CIDEr   | TF-IDF weighted n-gram consensus  |

## 🔬 Custom Loss: SSAL

```
L_total = λ₁·L_CE + λ₂·L_structural + λ₃·L_temporal
```

- **L_CE**: Cross-entropy for next-token prediction
- **L_structural**: KL divergence between predicted & GT damage distributions
- **L_temporal**: Penalty for violating pre→post damage ordering

## 📦 Tech Stack

- PyTorch 2.5 + CUDA 12.1
- HuggingFace Transformers 5.x
- PEFT (LoRA / QLoRA)
- CLIP ViT-B/16 (vision encoder)
- GPT-2 (language decoder)

## 📅 Timeline

| Month | Task                                  | Status |
|-------|---------------------------------------|--------|
| 1     | Literature review + environment setup | ✅     |
| 2     | Model architecture + training pipeline| ✅     |
| 3     | Experimentation + optimisation        | 🔄     |
| 4     | Evaluation + UI development           | 🔲     |
| 5     | Final report + documentation          | 🔲     |
