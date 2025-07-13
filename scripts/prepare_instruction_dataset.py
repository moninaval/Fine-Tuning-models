import os
import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer

SEEN_FILE = "seen_datasets.json"

def load_instruction_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

import json

def tokenize_examples(examples, tokenizer, max_len):
    print(f"[INFO] Received {len(examples)} examples")

    if not examples:
        print("[WARNING] No examples passed to tokenizer.")
        return {}

    # Print the first few examples for inspection
    print("[DEBUG] First 1-2 examples:")
    for i, ex in enumerate(examples[:2]):
        print(f"Example {i}: {json.dumps(ex, indent=2)}")

    # Validate that all keys exist
    for i, ex in enumerate(examples):
        if not all(k in ex for k in ("instruction", "input", "output")):
            print(f"[ERROR] Missing required keys in example {i}: {ex}")
            raise ValueError(f"Example {i} missing required keys.")

    # Prepare input and target texts
    try:
        input_texts = [
            f"{ex['instruction'].strip()}\n\n{ex['input'].strip()}" if ex['input'].strip()
            else ex['instruction'].strip()
            for ex in examples
        ]
        target_texts = [ex['output'].strip() for ex in examples]
        print("[INFO] Prepared input and target texts.")
    except Exception as e:
        print(f"[ERROR] While preparing input/target texts: {e}")
        raise

    # Print sample input/target
    print("[DEBUG] Sample input text:", input_texts[0] if input_texts else "None")
    print("[DEBUG] Sample target text:", target_texts[0] if target_texts else "None")

    # Tokenize inputs
    try:
        print("[INFO] Tokenizing input texts...")
        model_inputs = tokenizer(
            input_texts,
            max_length=max_len,
            truncation=True,
            padding="max_length"
        )
        print("[INFO] Tokenization of input texts complete.")
    except Exception as e:
        print(f"[ERROR] During input tokenization: {e}")
        raise

    # Tokenize outputs as labels
    try:
        print("[INFO] Tokenizing target texts...")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                target_texts,
                max_length=max_len,
                truncation=True,
                padding="max_length"
            )["input_ids"]
        print("[INFO] Tokenization of target texts complete.")
    except Exception as e:
        print(f"[ERROR] During label tokenization: {e}")
        raise

    model_inputs["labels"] = labels
    print("[INFO] Returning model inputs.")
    return model_inputs

def load_seen(path=SEEN_FILE):
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(json.load(f))
    return set()

def update_seen(seen, path=SEEN_FILE):
    with open(path, "w") as f:
        json.dump(sorted(list(seen)), f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--input_dir", default="data/incoming_instruction/")
    parser.add_argument("--output_dir", default="data/tokenized/")
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    seq_len = tokenizer.model_max_length
    print(f"🔹 Tokenizer max length = {seq_len}")

    seen = load_seen()
    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".jsonl"))
    count = 0

    for file in files:
        if file in seen:
            continue

        path = os.path.join(args.input_dir, file)
        print(f"🚀 Processing {path}")
        examples = load_instruction_jsonl(path)
        print("1loaded jsonl file")
        tokenized = tokenize_examples(examples, tokenizer, seq_len)
        print("2loaded jsonl file")
        dataset = Dataset.from_dict(tokenized)
        print("3loaded jsonl file")
        out_folder = os.path.join(args.output_dir, file.replace(".jsonl", ""))
        os.makedirs(out_folder, exist_ok=True)
        print("loaded jsonl file")
        dataset.save_to_disk(out_folder)
        print(f"✅ Saved tokenized dataset to: {out_folder}")

        seen.add(file)
        count += 1
        if args.max_files and count >= args.max_files:
            break

    update_seen(seen)
    print("✅ Instruction dataset preparation complete.")

if __name__ == "__main__":
    main()
