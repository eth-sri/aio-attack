"""
Direct magnitude pruning: zero out the smallest-|w| fraction of proj weights.
Bypasses llmcompressor (which has a leave_enabled bug that restores weights before saving).
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def prune_magnitude(model, sparsity: float):
    """Zero the bottom-`sparsity` fraction of abs values in all *proj.weight tensors."""
    stats = []
    for name, param in model.named_parameters():
        if "proj.weight" not in name:
            continue
        w = param.data
        k = int(w.numel() * sparsity)
        if k == 0:
            continue
        threshold = w.abs().flatten().kthvalue(k).values
        mask = w.abs() <= threshold
        param.data[mask] = 0.0
        actual_sparsity = mask.float().mean().item()
        stats.append((name, actual_sparsity))
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--sparsity",    type=float, required=True)
    p.add_argument("--force",       action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    if (out / "config.json").exists() and not args.force:
        print(f"Already exists: {out}. Use --force to overwrite.")
        return

    print(f"Loading {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Pruning to {args.sparsity*100:.0f}% sparsity")
    stats = prune_magnitude(model, args.sparsity)
    total_layers = len(stats)
    avg_sp = sum(s for _, s in stats) / total_layers if total_layers else 0
    print(f"Pruned {total_layers} proj.weight tensors, avg actual sparsity={avg_sp:.4f}")

    # verify
    total, zeros = 0, 0
    for _, param in model.named_parameters():
        if "proj.weight" in _:
            total += param.numel()
            zeros += (param.data == 0).sum().item()
    print(f"Verified: {zeros}/{total} zeros = {100*zeros/total:.2f}%")

    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out}")
    model.save_pretrained(str(out), max_shard_size="2GB")
    tokenizer.save_pretrained(str(out))
    print("Done.")


if __name__ == "__main__":
    main()
