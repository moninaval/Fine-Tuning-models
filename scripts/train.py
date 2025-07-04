import os
import json
import argparse
import glob
from load_model import load_model, load_yaml
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, load_from_disk
import torch
import math

SEEN_FILE = "seen_datasets.json"

def load_seen():
    return set(json.load(open(SEEN_FILE))) if os.path.exists(SEEN_FILE) else set()

def update_seen(seen):
    with open(SEEN_FILE, "w") as f:
        json.dump(sorted(list(seen)), f, indent=2)

def tokenize_dataset(input_path, tokenizer, out_dir):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    dataset = Dataset.from_dict({"text": chunks})

    tokenized = dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length"),
        batched=True
    )
    os.makedirs(out_dir, exist_ok=True)
    tokenized.save_to_disk(out_dir)
    return tokenized

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    shift_logits = torch.tensor(logits)[..., :-1, :].contiguous()
    shift_labels = torch.tensor(labels)[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = math.exp(loss.item())
    return {"eval_loss": loss.item(), "perplexity": perplexity}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/incoming/")
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--experiment_config", required=True)
    parser.add_argument("--tokenized_dir", default="data/tokenized/")
    parser.add_argument("--checkpoint_dir", default="checkpoints/")
    args = parser.parse_args()

    seen = load_seen()
    candidates = sorted(glob.glob(f"{args.data_dir}/*.txt"))
    new_file = next((f for f in candidates if os.path.basename(f) not in seen), None)

    if not new_file:
        print("✅ No new dataset to train. All are seen.")
        return

    print(f"🚀 New file found: {new_file}")

    # Load configs
    train_config = load_yaml(args.train_config)
    model_config = load_yaml(args.model_config)
    experiment_config = load_yaml(args.experiment_config)

    # Tokenize new file
    model, tokenizer = load_model(model_config, train_config)
    tokenized_path = os.path.join(args.tokenized_dir, os.path.basename(new_file).replace(".txt", ""))
    dataset = tokenize_dataset(new_file, tokenizer, tokenized_path)
    data = load_from_disk(tokenized_path).train_test_split(test_size=0.05)

    # Setup training
    tcfg = train_config["train"]
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        max_steps=tcfg["max_steps"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        evaluation_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg.get("save_total_limit", 2),
        learning_rate=tcfg["learning_rate"],
        logging_steps=tcfg["logging_steps"],
        report_to="none",
        fp16=tcfg.get("use_fp16", False),
        bf16=tcfg.get("use_bf16", False),
        run_name=experiment_config["experiment"]["name"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=compute_metrics
    )

    print("📦 Starting training...")
    trainer.train(resume_from_checkpoint=True)
    print("✅ Training completed.")

    # Mark dataset as seen
    seen.add(os.path.basename(new_file))
    update_seen(seen)

if __name__ == "__main__":
    main()
