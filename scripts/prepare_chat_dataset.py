import os
import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer

SEEN_FILE = "seen_datasets.json"

def load_seen(path=SEEN_FILE):
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(json.load(f))
    return set()

def update_seen(seen, path=SEEN_FILE):
    with open(path, "w") as f:
        json.dump(sorted(list(seen)), f, indent=2)

def flatten_chat(messages, assistant_role="assistant"):
    history = ""
    response = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role == assistant_role:
            response = content
        else:
            history += f"{role.capitalize()}: {content}\n"
    return history.strip(), response.strip()

def tokenize_chat_pairs(pairs, tokenizer, max_len):
    inputs = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]

    model_inputs = tokenizer(
        inputs,
        max_length=max_len,
        truncation=True,
        padding=False
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_len,
            truncation=True,
            padding=False
        )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--input_dir", default="data/incoming_chat")
    parser.add_argument("--output_dir", default="data/tokenized/")
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    seq_len = tokenizer.model_max_length
    print(f"🔹 Using tokenizer with max seq_len = {seq_len}")

    seen = load_seen()
    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".jsonl"))
    count = 0

    for file in files:
        if file in seen:
            continue

        path = os.path.join(args.input_dir, file)
        print(f"🚀 Processing chat file: {file}")
        chat_pairs = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    prompt, response = flatten_chat(data["messages"])
                    if prompt and response:
                        chat_pairs.append((prompt, response))
                except Exception as e:
                    print(f"⚠️ Skipped line due to error: {e}")

        if not chat_pairs:
            print(f"⚠️ No valid chat pairs found in {file}. Skipping.")
            continue

        tokenized = tokenize_chat_pairs(chat_pairs, tokenizer, seq_len)
        dataset = Dataset.from_dict(tokenized)

        out_folder = os.path.join(args.output_dir, file.replace(".jsonl", ""))
        os.makedirs(out_folder, exist_ok=True)

        dataset.save_to_disk(out_folder)
        print(f"✅ Tokenized and saved {len(dataset)} examples to {out_folder}")

        seen.add(file)
        count += 1
        if args.max_files and count >= args.max_files:
            break

    update_seen(seen)
    print("✅ Chat dataset preparation complete.")

if __name__ == "__main__":
    main()
