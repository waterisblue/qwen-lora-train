import os
import torch
import torch_npu

from fastapi import FastAPI
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


# ============================================================
# 配置
# ============================================================

BASE_MODEL = "/workspace/models/Qwen3-8B"

LORA_MODEL = "/workspace/output/qwen-lora-classifier"

DEVICE = "npu:0"


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Qwen Text Classification API",
    version="1.0.0",
)


# ============================================================
# Request
# ============================================================

class ClassifyRequest(BaseModel):
    text: str


# ============================================================
# Model
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    local_files_only=True,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

print("Loading LoRA...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL,
    local_files_only=True,
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """你是一个文本分类模型。
请根据输入文本判断其所属类别。
只输出类别名称，不要输出解释。
"""


def build_prompt(text: str):

    return (
        SYSTEM_PROMPT
        + "\n"
        + "输入文本：\n"
        + text.strip()
        + "\n\n"
        + "类别："
    )


# ============================================================
# Classification
# ============================================================

@torch.inference_mode()
def classify(text: str):

    prompt = build_prompt(text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    outputs = model.generate(
        **inputs,

        max_new_tokens=20,

        do_sample=False,

        temperature=None,

        top_p=None,
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    result = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return result


# ============================================================
# API
# ============================================================

@app.post("/classify")
def classify_api(req: ClassifyRequest):

    label = classify(req.text)

    return {
        "label": label
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "npu": torch.npu.is_available(),
    }