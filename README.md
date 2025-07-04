# 🔧 LLM Fine-Tuning Framework (LoRA / QLoRA / Full FT)

This repository provides a flexible, modular framework to fine-tune transformer-based language models (like **Phi-2**, **Mistral**, **LLaMA**, etc.) using:
- ✅ **LoRA** (Low-Rank Adaptation)
- ✅ **QLoRA** (Quantized LoRA for low-VRAM GPUs)
- ✅ **Full fine-tuning** (for large-scale systems)

You can configure everything — from which model to use, to what parts of it to adapt/train — using YAML configuration files.# Fine-Tuning-Phi-3

Usage for training:
python scripts/train.py \
  --train_config configs/train_qlora.yaml \
  --model_config configs/model_phi3.yaml \
  --dataset_config configs/dataset.yaml \
  --experiment_config configs/experiment.yaml


Usage for infrencing:Inference with merged model
python scripts/inference.py \
  --adapter_or_model final_model/ \
  --prompt "Summarize this Jira ticket..." \
  --merged

Usage for infrencing:Inference with LoRA adapter:
python scripts/inference.py \
  --adapter_or_model final_model/ \
  --prompt "Summarize this Jira ticket..." 

Usage:final adaptor merging for deployment
  python scripts/merge_adapter_for_finaldeployment.py \
  --base_model microsoft/phi-3-mini-4k-instruct \
  --adapter checkpoints/checkpoint-600 \
  --output_dir final_model/

Usage Data preparation for text based data
python scripts/prepare_dataset.py \
  --model_id microsoft/phi-3-mini-4k-instruct

Usage Data preparation for instruction based data ex- { "instruction": "Summarize", "input": "The service crashed.", "output": "Service failure." }

python scripts/prepare_instruction_dataset.py \
  --model_id microsoft/phi-3-mini-4k-instruct

 Usade Data Prepartion for chat based finetuning ex- {
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
} 

python scripts/prepare_chat_dataset.py \
  --model_id microsoft/phi-3-mini-4k-instruct



