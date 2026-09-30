import os
import json
import logging
from dataclasses import dataclass

import torch
import torch_npu

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


# ============================================================
# 基础配置
# ============================================================

MODEL_PATH = "/workspace/models/Qwen3-8B"

TRAIN_FILE = "/workspace/datasets/train.jsonl"
VAL_FILE = "/workspace/datasets/val.jsonl"

OUTPUT_DIR = "/workspace/output/qwen-lora-classifier"

MAX_LENGTH = 1024

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# 训练
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LOGGING_STEPS = 10
SAVE_STEPS = 500
EVAL_STEPS = 500


# ============================================================
# 日志
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# ============================================================
# NPU
# ============================================================

logger.info("torch version: %s", torch.__version__)
logger.info("torch_npu version: %s", torch_npu.__version__)
logger.info("NPU available: %s", torch.npu.is_available())
logger.info("NPU count: %s", torch.npu.device_count())

if not torch.npu.is_available():
    raise RuntimeError("NPU 不可用")

device = torch.device("npu:0")
torch.npu.set_device(device)


# ============================================================
# Tokenizer
# ============================================================

logger.info("Loading tokenizer: %s", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info("Tokenizer loaded")


# ============================================================
# Dataset
# ============================================================

logger.info("Loading dataset")

raw_datasets = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
    },
)

logger.info("Train samples: %d", len(raw_datasets["train"]))
logger.info("Validation samples: %d", len(raw_datasets["validation"]))


# ============================================================
# 构造训练数据
#
# 我们训练：
#
# 用户输入：
#   有人在网上发布涉诈信息
#
# 模型输出：
#   网络诈骗
#
# 实际训练格式：
#
# <prompt>
# ...
# </prompt>
#
# <answer>
# 网络诈骗
# </answer>
# ============================================================

SYSTEM_PROMPT = """你是一个文本分类模型。
请根据输入文本判断其所属类别。
只输出类别名称，不要输出解释。
"""


def build_prompt(text: str) -> str:
    return (
        SYSTEM_PROMPT
        + "\n"
        + "输入文本：\n"
        + text.strip()
        + "\n\n"
        + "类别："
    )


def preprocess_function(example):

    text = str(example["text"])
    label = str(example["label"])

    prompt = build_prompt(text)

    # prompt token
    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # answer token
    answer_tokens = tokenizer(
        label,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = (
        prompt_tokens["input_ids"]
        + answer_tokens["input_ids"]
        + [tokenizer.eos_token_id]
    )

    attention_mask = [1] * len(input_ids)

    # prompt部分不计算loss
    labels = (
        [-100] * len(prompt_tokens["input_ids"])
        + answer_tokens["input_ids"]
        + [tokenizer.eos_token_id]
    )

    # 截断
    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


logger.info("Tokenizing dataset")

tokenized_datasets = raw_datasets.map(
    preprocess_function,
    remove_columns=raw_datasets["train"].column_names,
    desc="Tokenizing",
)

logger.info("Dataset tokenization finished")


# ============================================================
# Model
# ============================================================

logger.info("Loading model")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,

    # NPU上建议使用float16
    torch_dtype=torch.float16,

    # 不要在这里 device_map="auto"
    # Trainer / DDP 会负责设备
)

model.config.use_cache = False

logger.info("Model loaded")


# ============================================================
# LoRA
# ============================================================

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,

    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,

    bias="none",

    # Qwen常见target modules
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()


# ============================================================
# Data Collator
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
)


# ============================================================
# TrainingArguments
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,

    fp16=True,
    bf16=False,

    logging_steps=LOGGING_STEPS,

    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,

    eval_strategy="steps",
    eval_steps=EVAL_STEPS,

    report_to="none",

    remove_unused_columns=False,

    dataloader_num_workers=4,

    gradient_checkpointing=True,

    optim="adamw_torch",

    warmup_ratio=0.05,

    lr_scheduler_type="cosine",

    load_best_model_at_end=False,

    ddp_find_unused_parameters=False,

    # NPU
    use_cpu=False,
)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],

    tokenizer=tokenizer,

    data_collator=data_collator,
)


# ============================================================
# Train
# ============================================================

logger.info("Starting training")

trainer.train()

logger.info("Training finished")


# ============================================================
# Save LoRA
# ============================================================

logger.info("Saving LoRA adapter")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info("Saved to: %s", OUTPUT_DIR)

print("\nTraining completed.")
print("LoRA output:", OUTPUT_DIR)