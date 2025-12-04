# train2.py (Final version with DDP proof and configured for epoch-based training)
import os
import torch

if "WORLD_SIZE" in os.environ:
    # Environment variables are set by `torchrun`, indicating a distributed run.
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"PROOF: Hello from Rank {rank} of {world_size} on local_rank {local_rank}.")
else:
    rank = -1

import math
import logging
from typing import Optional, Dict

# This is the key import for loading our pre-downloaded data
from datasets import load_from_disk, interleave_datasets

from transformers import (
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import wandb

# -----------------------------
# USER CONFIG — edit these ONLY
# -----------------------------
# This is the PARENT directory where all your split subfolders are located.
LOCAL_DATASET_DIR = "/scratch/prakhar/preloaded_dataset"

TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./tokenizer.json")
# ADD YOUR HF TOKEN = "YOUR_HF_TOKEN"
# ADD YOUR WANDB API KEY = "YOUR_WANDB_API_KEY"
WANDB_PROJECT = "lma_mini_project"
WANDB_RUN_NAME = "Qwen_mode_pretraining_epochs" # Changed name to reflect epoch training
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./ckpt/qwen-small-checkpoints")
HUB_MODEL_ID = "raja20221020/qwen-small-pretrained"

# training knobs
PER_DEVICE_BATCH_SIZE = int(os.environ.get("PER_DEVICE_BATCH_SIZE", 4))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-3))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 500))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", 1000))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", 1000))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", 100))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", 2))
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", 512))
SEED = int(os.environ.get("SEED", 42))

NUM_TRAIN_EPOCHS = float(os.environ.get("NUM_TRAIN_EPOCHS", 2.0))
MAX_STEPS = int(os.environ.get("MAX_STEPS", -1))


# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# To avoid log spam, we'll only let the main process (rank 0) log INFO messages.
if rank in [-1, 0]:
    logger.info("Starting Qwen-small pretraining script")

# -----------------------------
# Sanity: wandb & HF token
# -----------------------------
if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
elif rank in [-1, 0]:
    logger.warning("WANDB_API_KEY not found in env — wandb may not work unless you set it")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
elif rank in [-1, 0]:
    logger.info("HF_TOKEN not set in env — will fail if you attempt to push private artifacts to HF")

# -----------------------------
# Detect hardware and set mixed precision (FIXED)
# -----------------------------
num_gpus = torch.cuda.device_count()
if rank in [-1, 0]:
    logger.info(f"CUDA available: {torch.cuda.is_available()}, GPUs detected: {num_gpus}")

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    if rank in [-1, 0]: logger.info("Hardware supports BF16. Using BF16 for training.")
    use_bf16 = True
    use_fp16 = False
else:
    if rank in [-1, 0]: logger.info("Hardware does not support BF16. Falling back to FP16.")
    use_bf16 = False
    use_fp16 = torch.cuda.is_available()

cpu_count = os.cpu_count() or 4
dataloader_num_workers = min(8, max(2, cpu_count // 2))
if rank in [-1, 0]:
    logger.info(f"Using dataloader_num_workers={dataloader_num_workers}")

# -----------------------------
# 1) Load tokenizer
# -----------------------------
tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>"
})
tokenizer.model_input_names = ["input_ids", "attention_mask"]
if rank in [-1, 0]:
    logger.info(f"Loaded tokenizer from {TOKENIZER_PATH}; vocab_size={tokenizer.vocab_size}")

# ---------------------------------------------------------------------
# 2) Load tokenized splits from LOCAL DISK and concatenate (UPDATED)
# ---------------------------------------------------------------------
def try_load_split_from_disk(path: str):
    if not os.path.exists(path):
        if rank in [-1, 0]: logger.warning(f"Dataset path not found, skipping: {path}")
        return None
    try:
        ds = load_from_disk(path)
        if rank in [-1, 0]: logger.info(f"Successfully loaded split from {path}")
        return ds
    except Exception as e:
        if rank in [-1, 0]: logger.error(f"FATAL: Could not load from {path}: {e}")
        return None

lang_list = ["hindi", "english", "awadhi"]
train_parts, valid_parts, test_parts = [], [], []

if rank in [-1, 0]:
    logger.info(f"Loading pre-downloaded dataset splits from base directory: {LOCAL_DATASET_DIR}")

for L in lang_list:
    train_path = os.path.join(LOCAL_DATASET_DIR, f"{L}_train")
    dtr = try_load_split_from_disk(train_path)
    if dtr is not None: train_parts.append(dtr)
    valid_path = os.path.join(LOCAL_DATASET_DIR, f"{L}_valid")
    dval = try_load_split_from_disk(valid_path)
    if dval is not None: valid_parts.append(dval)
    test_path = os.path.join(LOCAL_DATASET_DIR, f"{L}_test")
    dtst = try_load_split_from_disk(test_path)
    if dtst is not None: test_parts.append(dtst)

if not train_parts:
    raise RuntimeError(f"No train splits were found in {LOCAL_DATASET_DIR}!")

train_ds = interleave_datasets(train_parts) if len(train_parts) > 1 else train_parts[0]
valid_ds = interleave_datasets(valid_parts) if len(valid_parts) > 1 else (valid_parts[0] if valid_parts else None)
test_ds  = interleave_datasets(test_parts)  if len(test_parts) > 1  else (test_parts[0]  if test_parts  else None)

if rank in [-1, 0]:
    logger.info(f"Loaded disk datasets: train={len(train_parts)} splits, valid={len(valid_parts)}, test={len(test_parts)}")


# -----------------------------
# 3) Build Qwen-small config & model
# -----------------------------
from transformers import AutoConfig

base_cfg = "Qwen/Qwen3-0.6B"
config = AutoConfig.from_pretrained(base_cfg)

config.vocab_size = tokenizer.vocab_size
config.n_layer = 12
config.n_head = 8
config.hidden_size = 512
config.intermediate_size = 2048
config.attention_dropout = getattr(config, "attention_dropout", 0.0)
config.hidden_dropout = getattr(config, "hidden_dropout", 0.0)

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_config(config)
model.resize_token_embeddings(len(tokenizer))

if rank in [-1, 0]:
    logger.info(f"Model created. Approx param count: {model.num_parameters()}")

# -----------------------------
# 4) Data collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -----------------------------
# 5) TrainingArguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=False,
    do_train=True,
    do_eval=(valid_ds is not None),
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    eval_strategy="steps" if valid_ds is not None else "no",
    eval_steps=EVAL_STEPS,
    logging_steps=LOGGING_STEPS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    # --- THIS IS THE KEY CHANGE FOR SPEEDUP COMPARISON ---
    num_train_epochs=NUM_TRAIN_EPOCHS,
    max_steps=MAX_STEPS, # This is -1, so num_train_epochs will be used
    # ---
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    fp16=use_fp16,
    bf16=use_bf16,
    dataloader_num_workers=dataloader_num_workers,
    report_to="wandb",
    run_name=WANDB_RUN_NAME,
    load_best_model_at_end=True if valid_ds is not None else False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=SEED,
    remove_unused_columns=False,
    push_to_hub=False,
)

# -----------------------------
# 6) Initialize WandB
# -----------------------------
if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.login(key=WANDB_API_KEY)
elif rank in [-1, 0]:
    logger.warning("WANDB_API_KEY missing; wandb will attempt to auto-login (may fail)")

if training_args.local_rank in [-1, 0]:
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, reinit=True)
    logger.info("wandb init done")

# -----------------------------
# 7) Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# -----------------------------
# 8) Training
# -----------------------------
resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", None)
if resume_checkpoint:
    if rank in [-1, 0]: logger.info(f"Resuming training from checkpoint {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
else:
    if rank in [-1, 0]: logger.info("Starting training from scratch")
    trainer.train()

# -----------------------------
# 9) Final evaluation
# -----------------------------
if test_ds is not None:
    if rank in [-1, 0]: logger.info("Evaluating on test set...")
    metrics = trainer.evaluate(eval_dataset=test_ds)
    eval_loss = metrics.get("eval_loss", None)
    if eval_loss is not None:
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
    if rank in [-1, 0]:
        logger.info("Test metrics: %s", metrics)
        wandb.log({"test_eval_loss": eval_loss, "test_perplexity": metrics.get("perplexity")})

# -----------------------------
# 10) Save final model
# -----------------------------
if training_args.local_rank in [-1, 0]:
    best_dir = os.path.join(OUTPUT_DIR, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info(f"Saved best model to {best_dir}")

    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir) # Fixed typo from original script (was save__model)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Saved final model to {final_dir}")

    if HUB_MODEL_ID and HF_TOKEN:
        logger.info(f"Pushing final model to Hugging Face Hub: {HUB_MODEL_ID}")
        trainer.push_to_hub(commit_message="Upload Qwen-small final", hub_model_id=HUB_MODEL_ID, use_temp_dir=True)
        logger.info("Pushed model to Hub")

    wandb.finish()
    logger.info("Training finished")
