"""Dataset loading for the two objectives.

Each of dataset_a / dataset_b is a JSONL file of {"instruction", "input",
"output"} records (the format already used by dataset/mcd_chosen.jsonl and
dataset/mcd_rejected.jsonl). Every training step pairs one example from
dataset_a with one (randomly drawn) example from dataset_b.
"""

import json
import random

import torch
from torch.utils.data import Dataset


def _read_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _build_user_content(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = (input_text or "").strip()
    return f"{instruction}\n\n{input_text}" if input_text else instruction


def _chat_template_ids(tokenizer, messages, add_generation_prompt):
    """apply_chat_template(tokenize=True) returns a plain list on some
    tokenizers/versions and a BatchEncoding (dict-like) on others; normalize
    to a flat list[int] either way."""
    out = tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=True)
    if not isinstance(out, list):
        out = out["input_ids"]
    return list(out)


class SFTJsonlDataset(Dataset):
    """Alpaca-style {instruction, input, output} JSONL -> tokenized chat
    example with completion-only labels (prompt tokens masked to -100)."""

    def __init__(self, path, tokenizer, max_len=1024, system_message="You are a helpful assistant."):
        eos_id = tokenizer.eos_token_id
        self.examples = []
        for ex in _read_jsonl(path):
            if not str(ex["output"]).strip():
                continue  # empty completion: Mistral-family templates raise on it, nothing to learn from it anyway
            messages = []
            if system_message.strip():
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": _build_user_content(ex["instruction"], ex.get("input", ""))})

            prompt_ids = _chat_template_ids(tokenizer, messages, add_generation_prompt=True)
            full_ids = _chat_template_ids(
                tokenizer, messages + [{"role": "assistant", "content": ex["output"]}], add_generation_prompt=False
            )
            if full_ids[-1] != eos_id:
                full_ids = full_ids + [eos_id]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                continue  # template edge case; skip rather than train on a misaligned mask
            if len(full_ids) > max_len or len(full_ids) <= len(prompt_ids):
                continue

            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            self.examples.append({"input_ids": full_ids, "labels": labels})

        if not self.examples:
            raise ValueError(f"no usable examples loaded from {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DualObjectiveDataset(Dataset):
    """Pairs one example from dataset_a (clean objective) with a randomly
    drawn example from dataset_b (noisy objective) per training step."""

    def __init__(self, dataset_a: Dataset, dataset_b: Dataset, seed: int = 0):
        self.a = dataset_a
        self.b = dataset_b
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        a_ex = self.a[idx % len(self.a)]
        b_ex = self.b[self._rng.randrange(len(self.b))]
        return {"a": a_ex, "b": b_ex}


def _pad_batch(features, pad_token_id):
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, attention_mask, labels = [], [], []
    for f in features:
        ids = list(f["input_ids"])
        lab = list(f["labels"])
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(lab + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class DualObjectiveCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        out = {}
        out.update({f"a_{k}": v for k, v in _pad_batch([f["a"] for f in features], self.pad_token_id).items()})
        out.update({f"b_{k}": v for k, v in _pad_batch([f["b"] for f in features], self.pad_token_id).items()})
        return out
