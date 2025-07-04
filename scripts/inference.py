import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def load_model(base_model_path, adapter_or_merged_path, merged=False):
    if merged:
        print("🔹 Loading fully merged model...")
        model = AutoModelForCausalLM.from_pretrained(
            adapter_or_merged_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print("🔹 Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        print("🔹 Loading LoRA/QLoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_or_merged_path)

    model.eval()
    return model

def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", help="Base model ID (only needed if merged=False)")
    parser.add_argument("--adapter_or_model", required=True, help="Adapter checkpoint OR merged model path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--merged", action="store_true", help="Use this flag if using a merged model")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_or_model if args.merged else args.base_model,
        trust_remote_code=True
    )
    model = load_model(args.base_model, args.adapter_or_model, args.merged)

    print("\n🧠 Generating output...\n")
    result = generate_response(model, tokenizer, args.prompt, args.max_new_tokens)
    print(result)

if __name__ == "__main__":
    main()
