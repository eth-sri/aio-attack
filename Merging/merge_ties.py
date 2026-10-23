#!/usr/bin/env python3
"""Merge models with TIES, DARE-TIES, DARE-linear, breadcrumbs, della, or della_linear.

All of these are task-vector methods that require a base model. They differ in
how the task vectors are sparsified and rescaled before combination.

Requires mergekit (see Merging/merging.py / MERGEKIT_PYTHON).

Examples
--------
# TIES: sparsified task-vector merge at density 0.5
python Merging/merge_ties.py \
    --model_path outputs/ab_lora_merged \
    --base_model /models/qwen2.5-7b \
    --donor_paths /models/donor_a /models/donor_b \
    --output_path outputs/merged_ties \
    --merge_method ties --density 0.5

# DARE-TIES
python Merging/merge_ties.py ... --merge_method dare_ties --density 0.53

# breadcrumbs_ties
python Merging/merge_ties.py ... --merge_method breadcrumbs_ties --density 0.9 --rescale 1
"""
import argparse
from pathlib import Path

from mergekit.config import InputModelDefinition, MergeConfiguration
from mergekit.common import ModelReference
from mergekit.merge import MergeOptions, run_merge

TIES_METHODS = {
    "ties",
    "dare_ties",
    "dare_linear",
    "breadcrumbs",
    "breadcrumbs_ties",
    "della",
    "della_linear",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   required=True, help="Primary model (trained).")
    p.add_argument("--base_model",   required=True, help="Base model for task vectors.")
    p.add_argument("--output_path",  required=True)
    p.add_argument("--merge_method", choices=sorted(TIES_METHODS), default="ties")
    p.add_argument("--donor_paths",  nargs="*", default=[],
                   help="Additional donor models to include in the merge.")
    p.add_argument("--weight",       type=float, default=1.0,
                   help="Task-vector weight for --model_path.")
    p.add_argument("--donor_weight", type=float, default=1.0,
                   help="Task-vector weight for each donor (same for all).")
    p.add_argument("--density",      type=float, default=0.53,
                   help="Fraction of task-vector elements to keep (0–1).")
    p.add_argument("--rescale",      type=float, default=1.0,
                   help="Global rescale factor applied after sparsification.")
    p.add_argument("--normalize",    action="store_true", default=False,
                   help="Normalize task-vector weights before combining.")
    p.add_argument("--int8_mask",    action="store_true", default=False,
                   help="Use int8 masks to reduce memory during merging.")
    p.add_argument("--dtype",   default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--cuda",    action="store_true", default=True)
    p.add_argument("--no_cuda", dest="cuda", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()

    per_model_params = {"weight": args.weight, "density": args.density}
    per_donor_params = {"weight": args.donor_weight, "density": args.density}

    models = [
        InputModelDefinition(
            model=ModelReference(model=args.model_path),
            parameters=per_model_params,
        )
    ]
    for dp in args.donor_paths:
        models.append(InputModelDefinition(
            model=ModelReference(model=dp),
            parameters=per_donor_params,
        ))

    global_params: dict = {}
    if args.normalize:
        global_params["normalize"] = True
    if args.rescale != 1.0:
        global_params["rescale"] = args.rescale
    if args.int8_mask:
        global_params["int8_mask"] = True

    cfg = MergeConfiguration(
        merge_method=args.merge_method,
        base_model=ModelReference(model=args.base_model),
        models=models,
        parameters=global_params or None,
        dtype=args.dtype,
        # base_model already anchors tokenizer selection deterministically
        # here, but set "auto" too for consistency with merge_linear.py
        # in case base_model's own tokenizer ever lacks one.
        chat_template="auto",
    )

    print(f"merge_method={args.merge_method}", flush=True)
    print(f"model_path={args.model_path}", flush=True)
    print(f"base_model={args.base_model}", flush=True)
    print(f"donor_paths={args.donor_paths}", flush=True)
    print(f"density={args.density}  weight={args.weight}  donor_weight={args.donor_weight}", flush=True)
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
