#!/usr/bin/env python3
"""Merge models with linear (weighted average) or task_arithmetic (task-vector) method.

Requires mergekit (see Merging/merging.py / MERGEKIT_PYTHON).

Examples
--------
# linear: 70% trained + 30% donor
python Merging/merge_linear.py \
    --model_path outputs/ab_lora_merged \
    --donor_paths /models/donor_a \
    --output_path outputs/merged_linear \
    --merge_method linear --weight 0.7 --donor_weight 0.3

# task_arithmetic: base + 0.7*(trained-base) + 0.3*(donor-base)
python Merging/merge_linear.py \
    --model_path outputs/ab_lora_merged \
    --base_model /models/qwen2.5-7b \
    --donor_paths /models/donor_a \
    --output_path outputs/merged_ta \
    --merge_method task_arithmetic --weight 0.7 --donor_weight 0.3
"""
import argparse
from pathlib import Path

import torch  # noqa: F401 -- must be imported (and in scope for the model_rebuild
# call below) before mergekit's ConfiguredModuleArchitecture pydantic model is first
# instantiated. That class's field types are only resolved lazily on first use (not
# at import time), and mergekit's own package never imports torch in the module where
# ConfiguredModuleArchitecture is defined -- so the lazy rebuild fails with
# "ConfiguredModuleArchitecture is not fully defined; you should define torch" the
# first time run_merge() actually builds a merge plan, on pydantic==2.10.6 (the
# version this venv pins mergekit==0.1.4 down to; mergekit's own install target,
# not something this repo chose). Not a bug in this file -- a third-party
# pydantic/mergekit forward-reference resolution gap, worked around here rather than
# by further patching the installed mergekit package in the shared venv.
from mergekit.architecture.base import (
    ConfiguredModelArchitecture,
    ConfiguredModuleArchitecture,
    ModelArchitecture,
    ModuleDefinition,
)
from mergekit.config import InputModelDefinition, MergeConfiguration
from mergekit.common import ModelReference
from mergekit.merge import MergeOptions, run_merge

# All pydantic models in mergekit.architecture.base hit the same lazy-forward-ref
# gap -- rebuild every one of them up front rather than fixing them one crash at a
# time as run_merge()'s plan-building code reaches each class in turn.
for _cls in (ModuleDefinition, ModelArchitecture, ConfiguredModuleArchitecture, ConfiguredModelArchitecture):
    _cls.model_rebuild(_types_namespace={"torch": torch})

TASK_VECTOR_METHODS = {"task_arithmetic"}
LINEAR_METHODS = {"linear"}
ALL_METHODS = LINEAR_METHODS | TASK_VECTOR_METHODS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   required=True, help="Primary model to merge.")
    p.add_argument("--output_path",  required=True)
    p.add_argument("--merge_method", choices=sorted(ALL_METHODS), default="linear")
    p.add_argument("--base_model",   default=None,
                   help="Base model for task_arithmetic (required by that method).")
    p.add_argument("--donor_paths",  nargs="*", default=[],
                   help="Additional model(s) to merge with.")
    p.add_argument("--weight",       type=float, default=1.0,
                   help="Weight for --model_path.")
    p.add_argument("--donor_weight", type=float, default=1.0,
                   help="Weight applied to each donor (same for all donors).")
    p.add_argument("--normalize",    action="store_true", default=True,
                   help="Normalize weights to sum to 1 (linear only).")
    p.add_argument("--no_normalize", dest="normalize", action="store_false")
    p.add_argument("--dtype",   default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--cuda",    action="store_true", default=True)
    p.add_argument("--no_cuda", dest="cuda", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()

    if args.merge_method in TASK_VECTOR_METHODS and not args.base_model:
        raise ValueError("--base_model is required for task_arithmetic")

    models = [
        InputModelDefinition(
            model=ModelReference(model=args.model_path),
            parameters={"weight": args.weight},
        )
    ]
    for dp in args.donor_paths:
        models.append(InputModelDefinition(
            model=ModelReference(model=dp),
            parameters={"weight": args.donor_weight},
        ))

    params = {}
    if args.merge_method == "linear":
        params["normalize"] = args.normalize

    cfg = MergeConfiguration(
        merge_method=args.merge_method,
        base_model=ModelReference(model=args.base_model) if args.base_model else None,
        models=models,
        parameters=params or None,
        dtype=args.dtype,
        # Without --base_model (plain "linear"), mergekit picks the tokenizer
        # source via `referenced_models()[0]`, which iterates a `set()` and is
        # ordered by the interpreter's randomized string hash seed -- a
        # different, uncontrollable choice on every subprocess invocation.
        # "auto" instead resolves the chat template by majority vote across
        # all input models (mergekit.merge._set_chat_template), which is
        # deterministic regardless of which model the random pick lands on.
        chat_template="auto",
    )

    print(f"merge_method={args.merge_method}", flush=True)
    print(f"model_path={args.model_path}", flush=True)
    print(f"donor_paths={args.donor_paths}", flush=True)
    print(f"base_model={args.base_model}", flush=True)
    print(f"weight={args.weight}  donor_weight={args.donor_weight}", flush=True)
    print(f"output_path={args.output_path}", flush=True)
    print("yaml_config:", flush=True)
    print(cfg.to_yaml(), flush=True)

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    run_merge(
        cfg,
        out_path=args.output_path,
        options=MergeOptions(
            cuda=args.cuda,
            copy_tokenizer=True,
            write_model_card=False,
        ),
    )
    print(f"saved={args.output_path}", flush=True)


if __name__ == "__main__":
    main()
