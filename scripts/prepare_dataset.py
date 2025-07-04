import os
import argparse
import json
from datasets import Dataset
from transformers import AutoTokenizer
from datetime import datetime

def load_seen(path="seen_datasets.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(json.load(f))
    return set()

def update_seen(seen, path="seen_datasets.json"):
    with open(path, "w") as f:
        json.dump(sorted(list(seen)), f, indent=2)

def prepare_dataset(input_file, tokenizer, out_dir, model_max_length):
    # Read raw text
    with open(input_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Chunk raw text into strings <= model_max_length (tokens)
    chunks = []
    tokens = tokenizer(raw_text, return_tensors="pt", truncation=False)["input_ids"][0].tolist()
    for i in range(0, len(tokens), model_max_length):
        chunk = tokens[i:i + model_max_length]
        chunks.append({"input_ids": chunk, "attention_mask": [1] * len(chunk)})

    # Create HF Dataset
    dataset = Dataset.from_list(chunks)

    # Save to HuggingFace dataset format
    dataset.save_to_disk(out_dir)

    # Also save to .jsonl
    jsonl_path = os.path.join(out_dir, "dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in chunks:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Tokenized and saved {len(chunks)} samples to {out_dir}")
    return len(chunks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True, help="e.g., microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--input_dir", default="data/incoming_text/")
    parser.add_argument("--output_dir", default="data/tokenized/")
    parser.add_argument("--max_files", type=int, default=None, help="Max files to process")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model_max_length = tokenizer.model_max_length
    print(f"🔹 Detected max seq_len = {model_max_length}")

    seen = load_seen()
    incoming_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".txt"))
    processed = 0

    for file in incoming_files:
        if file in seen:
            continue
        input_path = os.path.join(args.input_dir, file)
        output_path = os.path.join(args.output_dir, file.replace(".txt", ""))

        print(f"🚀 Processing {file}...")
        prepare_dataset(input_path, tokenizer, output_path, model_max_length)
        seen.add(file)
        processed += 1

        if args.max_files and processed >= args.max_files:
            break

    update_seen(seen)

if __name__ == "__main__":
    main()
