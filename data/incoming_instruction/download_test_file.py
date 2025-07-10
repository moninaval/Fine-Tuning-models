import json
import requests

alpaca_url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
alpaca_data = requests.get(alpaca_url).json()

with open("alpaca.jsonl", "w") as f:
    for ex in alpaca_data:
        f.write(json.dumps({
            "instruction": ex["instruction"],
            "input": ex["input"],
            "output": ex["output"]
        }) + "\n")

print("✅ Saved alpaca.jsonl")
