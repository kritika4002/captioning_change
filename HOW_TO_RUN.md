# Execution Guide: Automated Disaster Change Captioning

This document provides the definitive guide for environment setup, model training, evaluation, inference, presentation generation, and live demonstration of the Automated Disaster Change Captioning system.

---

## 1. Environment Setup and Verification

The project is built on PyTorch 2.5.1 with CUDA 12.1 GPU acceleration and Hugging Face Transformers. A dedicated virtual environment is configured at `C:\project-major\tf_gpu_env`.

### 1.1 Activating the Virtual Environment

**In Windows PowerShell:**
```powershell
& "C:\project-major\tf_gpu_env\Scripts\Activate.ps1"
```

**In Windows Command Prompt (CMD):**
```cmd
C:\project-major\tf_gpu_env\Scripts\activate.bat
```

Alternatively, you can run all commands directly using the environment's Python executable without activating:
```powershell
C:\project-major\tf_gpu_env\Scripts\python.exe <script_path>
```

### 1.2 Verifying Hardware and Dependencies
Run the following one-line diagnostic to confirm GPU availability and library installations:
```powershell
python -c "import torch, transformers, peft, PIL, matplotlib; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected output:
- `CUDA Available: True`
- `Device Name: NVIDIA GeForce RTX 3050 Laptop GPU` (or equivalent CUDA device)

---

## 2. Dataset Structure

The repository uses the xBD benchmark dataset stored in `./dataset/`, covering 19 global disaster events with bi-temporal satellite image pairs (pre-disaster and post-disaster) and damage polygon annotations.

Expected directory hierarchy:
```
dataset/
|-- guatemala-volcano/
|   |-- images/
|   |   |-- guatemala-volcano_00000000_pre_disaster.png
|   |   |-- guatemala-volcano_00000000_post_disaster.png
|   |   +-- ...
|   +-- labels/
|       |-- guatemala-volcano_00000000_pre_disaster.json
|       |-- guatemala-volcano_00000000_post_disaster.json
|       +-- ...
|-- hurricane-matthew/
|   |-- images/
|   +-- labels/
+-- ...
```

---

## 3. Step-by-Step Execution Workflow

### Step 1: Exploratory Data Analysis (EDA)
Generate statistical summaries, event breakdowns, and damage distribution charts:
```powershell
python scripts/dataset_stats.py --data-root ./dataset --output-dir ./outputs/eda
```
**Generated Outputs in `outputs/eda/`:**
- `dataset_stats.json`: Quantitative summary of events, sample counts, and building counts.
- `damage_distribution_pie.png`: Global breakdown across the 4 structural damage states.
- `samples_per_event.png`: Distribution of satellite image pairs across all 19 disaster events.
- `damage_by_event_stacked.png`: Stacked damage tier comparison across events.

---

### Step 2: Pre-Generate and Cache Ground-Truth Captions
Extract structural change metadata from JSON polygon labels into natural language captions:
```powershell
python scripts/prepare_captions.py --data-root ./dataset --output ./data/captions/all_captions.json
```
**Generated Output:**
- `data/captions/all_captions.json`: Indexed caption dictionary covering all 11,034 image pairs.

---

### Step 3: Train the Change Captioning Model
Train the end-to-end model (CLIP ViT-B/16 + Bi-Temporal Cross-Attention + GPT-2 with LoRA adapters + SSAL loss):
```powershell
python scripts/train.py --config configs/config.yaml
```

**Training Pipeline Details:**
- Reads hyperparameter configuration from `configs/config.yaml`.
- Uses Automatic Mixed Precision (FP16 AMP) and Gradient Accumulation ($k=4$) for memory efficiency.
- Disaster-level stratified partitioning: 80% train, 10% validation, 10% test.
- Checkpoints are saved automatically in `checkpoints/best/` and `checkpoints/final/`.
- Training history is recorded in `outputs/training_history.json`.
- Console and file execution logs are written to `logs/training.log`.

---

### Step 4: Generate Evaluation Figures and Excel Workbooks
Generate publication-grade 300 DPI evaluation figures and structured Excel reporting sheets:
```powershell
python scripts/generate_plots_and_sheets.py
```

**Generated Outputs in `outputs/evaluation/`:**
1. `loss_convergence.png`: Dual-panel plot of total training/validation loss and SSAL component decomposition.
2. `metric_progression.png`: Convergence curves for BLEU-1, BLEU-4, METEOR, ROUGE-L, and CIDEr across epochs.
3. `benchmark_comparison.png`: Grouped comparative bar chart against ResNet-LSTM, Dual-CNN Transformer, and Standard VLM.
4. `damage_consistency_analysis.png`: Accuracy comparison across all 4 damage tiers with and without SSAL guidance.
5. `evaluation_results.xlsx`: Multi-sheet Excel workbook containing:
   - Sheet 1: `Training_Validation_Log`
   - Sheet 2: `Benchmark_Comparison`
   - Sheet 3: `Ablation_Study`
6. CSV data files: `training_validation_log.csv`, `benchmark_comparison.csv`, and `ablation_study.csv`.

---

### Step 5: Run Demo Inference on Test Samples
Execute the trained checkpoint on unseen test samples to inspect predictions and generate visual comparison cards:
```powershell
python scripts/demo_inference.py
```

**Generated Outputs in `outputs/predictions/`:**
- Console display of Ground Truth vs. Predicted Captions, Damage Distributions, and NLP metric scores.
- `prediction_sample_1.png`: High-resolution visual satellite comparison card for Sample 1.
- `prediction_sample_2.png`: High-resolution visual satellite comparison card for Sample 2.
- `prediction_sample_3.png`: High-resolution visual satellite comparison card for Sample 3.
- `inference_summary.json`: Structured record of test evaluation scores.

---

### Step 6: Generate the Review Presentation Deck (.pptx)
Generate the 14-slide widescreen presentation deck aligned with the project review rubric:
```powershell
python scripts/generate_pptx.py
```

**Generated Output:**
- `outputs/Disaster_Change_Captioning_Review_Presentation.pptx`: 16:9 widescreen presentation deck covering all 14 mandatory review points with embedded tables, architecture cards, and charts.

---

### Step 7: Launch the Interactive Web Application (Streamlit)
Launch the interactive web platform for live evaluation and defense demonstration:
```powershell
streamlit run app.py
```
Or directly via the Python executable:
```powershell
C:\project-major\tf_gpu_env\Scripts\python.exe -m streamlit run app.py
```

**Accessing the Application:**
- Open your web browser at: `http://localhost:8501`
- Available navigation views:
  1. **Live Change Captioning**: Select any disaster event (e.g., Guatemala Volcano, Hurricane Matthew, Midwest Flooding, Santa Rosa Wildfire) and scene ID, view aligned pre- and post-disaster satellite imagery, trigger one-click GPU inference, inspect generated captions alongside ground truth, and view real-time structural damage bar charts.
  2. **Quantitative Evaluation**: Interactive high-resolution plots for loss convergence, metric progression, baseline benchmark comparisons, and a direct download button for `evaluation_results.xlsx`.
  3. **Study**: Systematic ablation study table and damage state accuracy comparison validating the contributions of Cross-Attention, SSAL Loss, and LoRA PEFT.
  4. **SSAL Loss**: Mathematical breakdown of the novel Structural-Semantic Alignment Loss formulation, loss weights, and module-by-module parameter allocation table.

---

## 4. Key Configuration Parameters (`configs/config.yaml`)

| Parameter | Section | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `data.root_dir` | Data | `"./dataset"` | Root directory containing event folders |
| `data.captions_cache` | Data | `"./data/captions/all_captions.json"` | Pre-computed caption JSON cache path |
| `data.num_workers` | Data | `0` | DataLoader worker threads (0 required on Windows) |
| `model.vision_encoder.freeze` | Model | `true` | Keeps CLIP ViT-B/16 parameters frozen |
| `model.change_detector.num_heads` | Model | `8` | Number of cross-attention heads |
| `model.lora.r` | Model | `16` | Rank of the LoRA adapter matrices |
| `model.lora.alpha` | Model | `32` | Scaling factor for LoRA updates |
| `loss.lambda_ce` | Loss | `1.0` | Weight for Cross-Entropy loss |
| `loss.lambda_structural` | Loss | `0.3` | Weight for Structural KL divergence loss |
| `loss.lambda_temporal` | Loss | `0.2` | Weight for Temporal monotonicity penalty |
| `training.epochs` | Training | `5` | Number of training epochs |
| `training.batch_size` | Training | `8` | Batch size per GPU step |
| `training.gradient_accumulation_steps` | Training | `4` | Gradient accumulation steps (effective batch size: 32) |
| `training.learning_rate` | Training | `2.0e-4` | Peak learning rate for AdamW optimizer |
| `training.fp16` | Training | `true` | Enables Automatic Mixed Precision |

---

## 5. Review and Defense Deliverables

The repository includes ready-to-present documentation and research materials:

1. **LaTeX Research Paper**:
   - Location: `outputs/paper/research_paper.tex`
   - Format: Complete 6-page IEEEtran camera-ready paper for Overleaf with 20 real research paper citations, mathematical formulations, benchmark tables, and ablation analysis.

2. **Viva Defense and Code Walkthrough**:
   - Location: `outputs/paper/VIVA_PANEL_DEFENSE_AND_COUNTERS.txt`
   - Format: Comprehensive plain-English manual containing line-by-line explanations of every Python file in `src/` and `scripts/`, a complete dictionary of all deep learning terms, and prepared counter-responses for panel cross-examination.

3. **Review Presentation Deck**:
   - Location: `outputs/Disaster_Change_Captioning_Review_Presentation.pptx`
   - Format: 14-slide widescreen presentation deck covering all 14 mandatory project review points.

4. **Evaluation Spreadsheet**:
   - Location: `outputs/evaluation/evaluation_results.xlsx`
   - Format: Multi-tab Excel workbook containing epoch-by-epoch training logs, benchmark comparisons, and component ablation data.

---

## 6. Troubleshooting and Hardware Best Practices

### Issue: Multi-processing Worker Freeze on Windows
- **Symptom**: Script hangs during DataLoader initialization.
- **Cause**: Windows process spawning introduces IPC overhead that can stall multi-threaded PyTorch data loaders.
- **Remedy**: Always keep `num_workers: 0` in `configs/config.yaml`.

### Issue: PyTorch 2.5 Model Loading Security Warning
- **Symptom**: `ValueError: Weights only load failed...`.
- **Cause**: PyTorch 2.5 default checkpoint loading behavior.
- **Remedy**: All project scripts explicitly pass `weights_only=False` or use Hugging Face safetensors format.

### Issue: GPU Out of Memory (CUDA OOM)
- **Symptom**: `torch.cuda.OutOfMemoryError`.
- **Cause**: VRAM exhaustion on smaller GPUs.
- **Remedy**: The current setup uses approximately 4.2 GB of VRAM. If running on a GPU with 4 GB or less, reduce `batch_size` from `8` to `4` in `configs/config.yaml` and increase `gradient_accumulation_steps` from `4` to `8` to maintain the effective batch size of 32.

### Issue: Streamlit Port Conflict
- **Symptom**: `Port 8501 is already in use`.
- **Remedy**: Specify an alternate port:
  ```powershell
  streamlit run app.py --server.port 8502
  ```
