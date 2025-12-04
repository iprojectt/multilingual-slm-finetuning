# generate.py
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, AutoTokenizer, GenerationConfig
import argparse
import os

# --- 1. Argument Parser ---
# This lets us choose the model and prompt from the command line
parser = argparse.ArgumentParser(description="Generate text from a pretrained or fine-tuned Qwen model.")
parser.add_argument(
    "--model_type",
    type=str,
    required=True,
    choices=["pretrained", "finetuned"],
    help="Specify which model to use: the base 'pretrained' model or the 'finetuned' LoRA adapter."
)
parser.add_argument(
    "--prompt",
    type=str,
    required=True,
    help="The full prompt to send to the model. Use quotes if it contains spaces."
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=150,
    help="The maximum number of new tokens to generate."
)

args = parser.parse_args()


# --- 2. Configuration ---
# All your project paths are defined here
PRETRAINED_MODEL_PATH = "./ckpt/qwen-small-checkpoints/final"
FINETUNED_ADAPTER_PATH = "./qwen-small-finetuned-adapter" # This is the LoRA adapter directory
# TOKENIZER_PATH = "./tokenizer.json"


# --- 3. Main Inference Logic ---
def run_inference():
    print("--- Starting Inference Script ---")

    # --- Load Tokenizer ---
    model_path_for_tokenizer = PRETRAINED_MODEL_PATH
    print(f"Loading correctly configured tokenizer from: {model_path_for_tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(model_path_for_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Determine which model to load ---
    if args.model_type == "pretrained":
        model_path = PRETRAINED_MODEL_PATH
        print(f"Loading BASE PRETRAINED model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    elif args.model_type == "finetuned":
        from peft import PeftModel
        base_model_path = PRETRAINED_MODEL_PATH
        adapter_path = FINETUNED_ADAPTER_PATH
        print(f"Loading BASE model from: {base_model_path}")
        print(f"Attaching LoRA adapter from: {adapter_path}")
        
        # Load the base model first
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        # Now, load and merge the LoRA adapter
        model = PeftModel.from_pretrained(model, adapter_path)
        # Optional: Merge the adapter into the base model for faster inference
        # This uses more memory but can be quicker for generation.
        # model = model.merge_and_unload()

    else:
        raise ValueError("Invalid model_type specified.")

    model.eval() # Set the model to evaluation mode

    # --- Prepare Inputs ---
    print("\n--- User Prompt ---")
    print(args.prompt)
    print("-------------------")

    inputs = tokenizer(args.prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)
    # --- Generate Text ---
    print("\nGenerating response...")
    
    # Define generation configuration
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.6,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Generate output
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)
    
    # Decode and print the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n--- Full Model Output ---")
    print(generated_text)
    print("-------------------------\n")


if __name__ == "__main__":
    run_inference()