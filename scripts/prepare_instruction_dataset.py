import os
import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer

def load_instruction_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def tokenize_examples(examples, tokenizer, max_len):
    input_texts = [
        f"{ex['instruction'].strip()}\n\n{ex['input'].strip()}" if ex['input'].strip()
        else ex['instruction'].strip()
        for ex in examples
    ]
    target_texts = [ex['output'].strip() for ex in examples]

    model_inputs = tokenizer(
        input_texts,
        max_length=max_len,
        truncation=True,
        padding=False
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target_texts,
            max_length=max_len,
            truncation=True,
            padding=False
        )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs

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

    files = [f for f in os.listdir(args.input_dir) if f.endswith(".jsonl")]
    seen_file = "seen_datasets.json"
    seen = set(json.load(open(seen_file))) if os.path.exists(seen_file) else set()
    count = 0

    for file in sorted(files):
        if file in seen:
            continue

        path = os.path.join(args.input_dir, file)
        print(f"🚀 Processing {file}")
        examples = load_instruction_jsonl(path)

        tokenized = tokenize_examples(examples, tokenizer, seq_len)
        dataset = Dataset.from_dict(tokenized)

        out_folder = os.path.join(args.output_dir, file.replace(".jsonl", ""))
        os.makedirs(out_folder, exist_ok=True)

        # Save to disk
        dataset.save_to_disk(out_folder)

        # Also save .jsonl
        out_jsonl = os.path.join(out_folder, "tokenized.jsonl")
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for i in range(len(dataset)):
                f.write(json.dumps({k: dataset[k][i] for k in dataset.column_names}) + "\n")

        seen.add(file)
        count += 1
        if args.max_files and count >= args.max_files:
            break

    with open(seen_file, "w") as f:
        json.dump(sorted(list(seen)), f, indent=2)

    print("✅ Instruction dataset preparation complete.")

if __name__ == "__main__":
    main()
