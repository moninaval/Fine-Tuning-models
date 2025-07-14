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
├── checkpoints/                # Model checkpoints (LoRA adapters + optimizer state)
├── config/                     # YAML configuration files for training
├── seen_datasets.json          # Tracks already tokenized files
├── prepare_raw_text_dataset.py
├── prepare_instruction_dataset.py
├── prepare_chat_dataset.py
├── train.py                    # QLoRA training entry point



 Dataset Preparation
1.	Tokenize from raw .txt:
python scripts/prepare_dataset.py --model_id microsoft/phi-3-mini-4k-instruct --input_dir data/incoming_instruction --output_dir data/tokenized/

2.	tokenize from instruction .jsonl:
python scripts/prepare_instruction_dataset.py --model_id microsoft/phi-3-mini-4k-instruct --input_dir data/incoming_instruction --output_dir data/tokenized/

3.	tokenize from chat .jsonl:
python scripts/prepare_chat_dataset.py --model_id microsoft/phi-3-mini-4k-instruct --input_dir data/incoming_instruction --output_dir data/tokenized/
4.	Fine-Tune the Model:
python scripts/train.py --train_config config/train_qlora.yaml --model_config config/model_phi3.yaml --experiment_config config/experiment.yaml --dataset_name alpaca

!python inference.py \
  --base_model microsoft/phi-3-mini-4k-instruct \
  --adapter_or_model checkpoints/checkpoints01 \
  --prompt "What are the three primary colors?"


We use QLoRA to fine-tune the phi-3-mini-4k-instruct model incrementally each night.
Each training run produces a new adapter checkpoint (e.g., checkpoints/checkpoint01, checkpoint02, etc.).
The base model remains fixed, while only the adapters are updated with new data.
The next training run uses the latest adapter as a starting point for continued learning.
Only the adapter path in config/model_phi3.yaml needs to be updated nightly.
The training command (scripts/train.py ...) remains unchanged.(model.yaml file will be updated with new checkpoints and train.py code shall be modified to train on particluar/last adaptor)
This approach is memory-efficient, as only ~12M parameters are trained.
It supports continuous learning without retraining from scratch.
Adapters are kept separate for modularity and versioning.
The model evolves nightly, learning from each day’s new dataset.

For sheduled training
python scripts/train.py \
  --tokenized_dir data/tokenized/ \
  --dataset_name alpaca \
  --train_config config/train_qlora.yaml \
  --model_config config/model_phi3.yaml \
  --experiment_config config/experiment.yaml \
  --checkpoint_dir checkpoints/checkpoint11 \
  --resume_checkpoint_path checkpoints/checkpoint10

