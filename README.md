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
bash
Copy
Edit
python prepare_raw_text_dataset.py \
  --model_id microsoft/phi-3-mini-4k-instruct \
  --input_dir data/incoming_text/ \
  --output_dir data/tokenized/
2. Prepare from instruction .jsonl
bash
Copy
Edit
python prepare_instruction_dataset.py \
  --model_id microsoft/phi-3-mini-4k-instruct \
  --input_dir data/incoming_instruction/ \
  --output_dir data/tokenized/
Each line must be like:

json
Copy
Edit
{"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}
3. Prepare from chat .jsonl
bash
Copy
Edit
python prepare_chat_dataset.py \
  --model_id microsoft/phi-3-mini-4k-instruct \
  --input_dir data/incoming_chat/ \
  --output_dir data/tokenized/
Each line must be like:

json
Copy
Edit
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
🚀 Fine-Tune the Model
After tokenizing, train the model using:

bash
Copy
Edit
python train.py \
  --train_config config/train.yaml \
  --model_config config/model.yaml \
  --experiment_config config/experiment.yaml
It picks the next unseen tokenized folder from data/tokenized/

Tracks progress in seen_datasets.json

Saves checkpoints in checkpoints/

✅ Notes
The prepare_*.py scripts only tokenize unseen files.

The train.py script does not re-tokenize — it only trains from .arrow datasets.

All datasets are split into 95% train / 5% eval.





