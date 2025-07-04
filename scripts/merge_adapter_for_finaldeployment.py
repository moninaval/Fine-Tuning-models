import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def merge_and_save(base_model_path, adapter_checkpoint, output_dir):
    print(f"🔹 Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )

    print(f"🔹 Loading LoRA/QLoRA adapter from: {adapter_checkpoint}")
    model = PeftModel.from_pretrained(base_model, adapter_checkpoint)
    
    print("🔁 Merging adapter into base model...")
    model = model.merge_and_unload()  # Now it's a regular Hugging Face model

    print(f"💾 Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)

    print("🔹 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print("✅ Merge complete. Model is ready for standalone deployment.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Base model path (e.g. microsoft/phi-3-mini-4k-instruct)")
    parser.add_argument("--adapter", required=True, help="LoRA/QLoRA adapter checkpoint path")
    parser.add_argument("--output_dir", required=True, help="Where to save the merged model")
    args = parser.parse_args()

    merge_and_save(args.base_model, args.adapter, args.output_dir)
