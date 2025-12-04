# finetune.py (Final version with FAST_MODE for quick iteration)
import os
import torch
import wandb
from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

# -----------------------------
# --- 1. Configuration ---
# -----------------------------
# Paths and Model IDs
PRETRAINED_MODEL_PATH = "./ckpt/qwen-small-checkpoints/final"
model_path_for_tokenizer = "./ckpt/qwen-small-checkpoints/final"
DEID_REPO_ID = "raja20221020/english_hindi_awadhi_deidentification"
SIMPLIFY_REPOS = [
    "raja20221020/english-text-simplification-for-finetuning",
    "raja20221020/hindi-text-simplification-for-finetuning",
    "raja20221020/awadhi-text-simplification-for-finetuning"
]
OUTPUT_DIR = "./qwen-small-finetuned-adapter"
HUB_MODEL_ID = "raja20221020/qwen-small-finetuned-multitask"

# Credentials
HF_TOKEN = "YOUR HF TOKEN"
WANDB_API_KEY = "YOUR API_KEY"

# W&B Configuration
WANDB_PROJECT = "finetune_1"
WANDB_RUN_NAME = "Qwen_small_finetune_multitask_multilang_full" # Default name for full run

# Training Hyperparameters
MICRO_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 2
MAX_SEQ_LENGTH = 512
SAVE_STEPS = 250
EVAL_STEPS = 250
LOGGING_STEPS = 25
WARMUP_STEPS = 50
SEED = 42
VALIDATION_SPLIT_PERCENTAGE = 10

# -----------------------------
# --- 1a. Fast Configuration for Quick Iteration ---
# -----------------------------
# Set this to True to enable a fast ~30 minute run for debugging and testing.
# Set to False to run the full 3-hour training.
FAST_MODE = True

if FAST_MODE:
    logging.warning("🚀 RUNNING IN FAST MODE FOR QUICK ITERATION 🚀")
    # [SPEED] Take a small fraction of the data for a quick run
    DATASET_SAMPLE_PERCENT = 50
    # [SPEED] Reduce epochs for a shorter run
    NUM_TRAIN_EPOCHS = 1
    # [SPEED] Use a larger batch size as memory allows (due to shorter sequence length)
    MICRO_BATCH_SIZE = 16
    # [SPEED] Reduce sequence length for much faster processing per step
    MAX_SEQ_LENGTH = 256
    # [SPEED] Adjust save/eval steps for the shorter run
    SAVE_STEPS = 50
    EVAL_STEPS = 50
    # [SPEED] Give it a different W&B name to avoid confusion
    WANDB_RUN_NAME = "Qwen_small_finetune_FAST_RUN"

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# --- 2. Authentication and Initialization ---
# -----------------------------
logger.info("Authenticating with Hugging Face and WandB...")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.login(key=WANDB_API_KEY)

if os.environ.get("LOCAL_RANK", "0") == "0":
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, reinit=True)

# -----------------------------
# --- 3. Load Model and Tokenizer ---
# -----------------------------
logger.info(f"Loading pretrained model from: {PRETRAINED_MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    PRETRAINED_MODEL_PATH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    trust_remote_code=True,
)

logger.info(f"--- Loading correctly configured tokenizer from: {model_path_for_tokenizer} ---")
tokenizer = AutoTokenizer.from_pretrained(model_path_for_tokenizer)

if tokenizer.pad_token is None:
    logger.info("PAD token not found. Setting PAD token to be the same as EOS token.")
    tokenizer.pad_token = tokenizer.eos_token

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
logger.info(f"Tokenizer and model configured with PAD token ID: {tokenizer.pad_token_id}")

# -----------------------------
# --- 4. PEFT (LoRA) Configuration ---
# -----------------------------
logger.info("Setting up LoRA configuration...")
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8 if FAST_MODE else 16,  # [SPEED] Use a smaller rank in fast mode
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------------
# --- 5. Load and Process Datasets ---
# -----------------------------
def create_prompt(instruction, input_text):
    return f"""### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"""

def process_and_tokenize_dataset(dataset, task_type):
    # This function remains the same
    def apply_template(examples):
        prompts = [create_prompt(instr, inp) for instr, inp in zip(examples["instruction"], examples["input"])]
        full_texts = [p + out + tokenizer.eos_token for p, out in zip(prompts, examples["output"])]
        return {"text": full_texts}

    column_names = list(dataset.column_names)
    processed_dataset = dataset.map(apply_template, batched=True, remove_columns=column_names)
    return processed_dataset.map(
        lambda examples: tokenizer(
            examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length"
        ),
        batched=True,
    )

def get_train_eval_splits(repo_id):
    """Loads a dataset, optionally samples it, and ensures it has 'train' and 'eval' splits."""
    logger.info(f"Loading dataset from: {repo_id}")
    dataset = load_dataset(repo_id)

    if "validation" not in dataset:
        logger.warning(f"'{repo_id}' missing 'validation' split. Creating one from 'train' split.")
        if "train" not in dataset:
            raise ValueError(f"Cannot create validation split as '{repo_id}' has no 'train' split.")
        
        split_dataset = dataset["train"].train_test_split(
            test_size=(VALIDATION_SPLIT_PERCENTAGE / 100), shuffle=True, seed=SEED
        )
        train_split, eval_split = split_dataset["train"], split_dataset["test"]
    else:
        train_split, eval_split = dataset["train"], dataset["validation"]

    # --- THIS IS THE NEW SPEEDUP LOGIC ---
    if FAST_MODE and DATASET_SAMPLE_PERCENT < 100:
        sample_size_train = int(len(train_split) * (DATASET_SAMPLE_PERCENT / 100))
        sample_size_eval = int(len(eval_split) * (DATASET_SAMPLE_PERCENT / 100))
        
        logger.info(f"FAST MODE: Sampling down to {DATASET_SAMPLE_PERCENT}% of the data.")
        logger.info(f"  - New train size: {sample_size_train} (from {len(train_split)})")
        logger.info(f"  - New eval size: {sample_size_eval} (from {len(eval_split)})")

        train_split = train_split.shuffle(seed=SEED).select(range(sample_size_train))
        eval_split = eval_split.shuffle(seed=SEED).select(range(sample_size_eval))
    # --- END OF NEW LOGIC ---

    return train_split, eval_split

# Load and process all datasets
deid_train, deid_valid = get_train_eval_splits(DEID_REPO_ID)
processed_deid_train = process_and_tokenize_dataset(deid_train, "deid")
processed_deid_valid = process_and_tokenize_dataset(deid_valid, "deid")

processed_simplify_train_list, processed_simplify_valid_list = [], []
for repo in SIMPLIFY_REPOS:
    simplify_train, simplify_valid = get_train_eval_splits(repo)
    processed_simplify_train_list.append(process_and_tokenize_dataset(simplify_train, "simplify"))
    processed_simplify_valid_list.append(process_and_tokenize_dataset(simplify_valid, "simplify"))

# Interleave datasets for multi-task learning
logger.info("Interleaving datasets for multi-task training...")
train_dataset = interleave_datasets(
    [processed_deid_train] + processed_simplify_train_list,
    probabilities=[0.5] + [0.5 / len(processed_simplify_train_list)] * len(processed_simplify_train_list),
    seed=SEED
)
eval_dataset = interleave_datasets(
    [processed_deid_valid] + processed_simplify_valid_list,
    probabilities=[0.5] + [0.5 / len(processed_simplify_valid_list)] * len(processed_simplify_valid_list),
    seed=SEED
)

logger.info(f"Combined training dataset size: {len(train_dataset)}")
logger.info(f"Combined evaluation dataset size: {len(eval_dataset)}")

# -----------------------------
# --- 6. Trainer Setup ---
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",
    run_name=WANDB_RUN_NAME,
    push_to_hub=False,
    hub_model_id=HUB_MODEL_ID,
    seed=SEED,
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# -----------------------------
# --- 7. Training ---
# -----------------------------
logger.info("Starting fine-tuning...")
trainer.train()
logger.info("Fine-tuning completed.")

# -----------------------------
# --- 8. Save and Push Final Model ---
# -----------------------------
logger.info(f"Saving the final LoRA adapter to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
logger.info(f"Pushing adapter to Hub: {HUB_MODEL_ID}")
trainer.push_to_hub()
logger.info("Successfully pushed adapter to Hub.")

wandb.finish()
logger.info("Script finished successfully.")