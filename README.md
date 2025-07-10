# 🔧 LLM Fine-Tuning Framework (LoRA / QLoRA / Full FT)

This repository provides a flexible, modular framework to fine-tune transformer-based language models (like **Phi-2**, **Mistral**, **LLaMA**, etc.) using:
- ✅ **LoRA** (Low-Rank Adaptation)
- ✅ **QLoRA** (Quantized LoRA for low-VRAM GPUs)
- ✅ **Full fine-tuning** (for large-scale systems)

You can configure everything — from which model to use, to what parts of it to adapt/train — using YAML configuration files.# Fine-Tuning-Phi-3
This project supports fine-tuning large language models like Phi-2 or Phi-3 using tokenized datasets in HuggingFace format (.arrow). It supports 3 types of inputs:

🔤 Raw .txt files
📘 Instruction-style .jsonl files
💬 Chat-style .jsonl files

├── data/
│   ├── incoming_text/          # Raw .txt files
│   ├── incoming_instruction/   # Instruction .jsonl files
│   ├── incoming_chat/          # Chat .jsonl files
│   └── tokenized/              # Output tokenized datasets (.arrow)
│
├── checkpoints/                # Model checkpoints
├── config/                     # YAML configuration files
├── seen_datasets.json          # Tracks processed files
├── prepare_raw_text_dataset.py
├── prepare_instruction_dataset.py
├── prepare_chat_dataset.py
├── train.py
 Dataset Preparation
1. Prepare from raw .txt
2. python scripts/prepare_raw_text_dataset.py \
  --model_id microsoft/phi-3-mini-4k-instruct \
  --input_dir data/incoming_text/ \
  --output_dir data/tokenized/
1. Prepare from instruction .jsonl
python scripts/prepare_instruction_dataset.py \
  --model_id microsoft/phi-3-mini-4k-instruct \
  --input_dir data/incoming_instruction/ \
  --output_dir data/tokenized/

1. Prepare from chat .jsonl
python prepare_chat_dataset.py \
  --model_id microsoft/phi-3-mini-4k-instruct \
  --input_dir data/incoming_chat/ \
  --output_dir data/tokenized/

🚀 Fine-Tune the Model
After tokenizing, train the model using:
python train.py \
  --train_config config/train_qlora.yaml \
  --model_config config/model_phi3.yaml \
  --experiment_config config/experiment.yaml
