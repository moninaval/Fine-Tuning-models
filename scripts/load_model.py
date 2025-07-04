import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model(model_config, train_config):
    model_name = model_config["model"]["name_or_path"]
    tokenizer_name = model_config["model"].get("tokenizer_name", model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # for safety

    # Load quantization config if QLoRA
    quant_cfg = train_config.get("quantization", {})
    is_qlora = train_config["peft"]["mode"] == "qlora"

    if is_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

    # Optional: unfreeze non-transformer components
    trainable = train_config["peft"].get("trainable_parts", {})
    if trainable.get("train_output_head", False):
        if hasattr(model, "lm_head"):
            model.lm_head.requires_grad_(True)

    if trainable.get("train_embeddings", False):
        if hasattr(model, "get_input_embeddings"):
            model.get_input_embeddings().requires_grad_(True)

    if trainable.get("train_norm_layers", False):
        for name, module in model.named_modules():
            if "norm" in name.lower():
                for param in module.parameters():
                    param.requires_grad = True

    # PEFT / LoRA / QLoRA adapter injection
    peft_mode = train_config["peft"]["mode"]
    if peft_mode in ["lora", "qlora"]:
        lora_cfg = train_config["peft"]
        peft_config = LoraConfig(
            r=lora_cfg.get("lora_r", 8),
            lora_alpha=lora_cfg.get("lora_alpha", 16),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=TaskType[lora_cfg.get("task_type", "CAUSAL_LM")],
            target_modules=lora_cfg["target_modules"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer
