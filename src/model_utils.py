import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Convenience aliases for the base models used in this project. A bare local
# directory path (containing config.json) is always used as-is.
MODEL_ALIASES = {
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}


def resolve_model_path(name: str) -> str:
    name = (name or "").strip()
    if os.path.isdir(name) and os.path.exists(os.path.join(name, "config.json")):
        return name
    return MODEL_ALIASES.get(name.lower(), name)


def load_model_and_tokenizer(model_name: str, dtype=torch.bfloat16):
    path = resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    model.config.use_cache = False  # required for gradient checkpointing
    return model, tokenizer
