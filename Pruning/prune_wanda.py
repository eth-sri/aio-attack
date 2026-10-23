"""
Prune a model using Wanda or SparseGPT via llmcompressor.
"""
import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Pruning goes through llmcompressor's native oneshot() API and its pruning
# modifiers; Pruning/pruning.py dispatches here.
#
# llmcompressor 0.9.0.2 pins transformers<=4.57.3 and imports
# TORCH_INIT_FUNCTIONS from transformers.modeling_utils; the shared venv runs
# transformers 5.5.1 (required elsewhere in this project) where that symbol
# moved to transformers.initialization. Shim it back onto modeling_utils
# before importing llmcompressor rather than downgrading transformers
# project-wide.
import transformers.modeling_utils as _hf_modeling_utils

if not hasattr(_hf_modeling_utils, "TORCH_INIT_FUNCTIONS"):
    from transformers.initialization import TORCH_INIT_FUNCTIONS as _TORCH_INIT_FUNCTIONS

    _hf_modeling_utils.TORCH_INIT_FUNCTIONS = _TORCH_INIT_FUNCTIONS

# llmcompressor's initialize_model_from_path() unconditionally passes
# use_auth_token=None into AutoModelForCausalLM.from_pretrained(). transformers
# <=4.57 special-cased and dropped that legacy kwarg internally; 5.5.1 no
# longer recognizes it as a top-level from_pretrained parameter, so it falls
# through **kwargs all the way into the model's own __init__ and crashes
# (Qwen2ForCausalLM.__init__() got an unexpected keyword argument
# 'use_auth_token'). We never pass a token through this script, so it's always
# None here -- strip it before dispatch rather than downgrading transformers.
import transformers as _hf_transformers

_orig_causal_lm_from_pretrained = _hf_transformers.AutoModelForCausalLM.from_pretrained.__func__


def _causal_lm_from_pretrained_no_use_auth_token(cls, *args, **kwargs):
    kwargs.pop("use_auth_token", None)
    return _orig_causal_lm_from_pretrained(cls, *args, **kwargs)


_hf_transformers.AutoModelForCausalLM.from_pretrained = classmethod(
    _causal_lm_from_pretrained_no_use_auth_token
)

# SparsityModifierBase._infer_sequential_targets() (shared by Wanda and
# SparseGPT) calls llmcompressor.utils.pytorch.module.get_no_split_params(),
# which in turn calls model._get_no_split_modules("auto") -- a PreTrainedModel
# method that existed in transformers <=4.57 but was removed in 5.5.1 in favor
# of a plain _no_split_modules class/instance attribute (a set of class name
# strings). Reimplement get_no_split_params() against the new attribute and
# patch it into sgpt_base's namespace (it was imported there via `from ...
# import get_no_split_params`, so patching the defining module alone wouldn't
# reach sgpt_base's already-bound name).
import llmcompressor.modifiers.pruning.sparsegpt.sgpt_base as _sgpt_base
try:
    from llmcompressor.utils.fsdp.helpers import maybe_get_wrapped
except ModuleNotFoundError:
    # llmcompressor >=0.10.1.dev dropped the llmcompressor.utils.fsdp package
    # (FSDP1 support removed). maybe_get_wrapped just unwrapped an
    # FSDP-wrapped module; with no FSDP wrapping in this path it is a no-op.
    def maybe_get_wrapped(model):
        return model
from llmcompressor.utils.pytorch.module import ALL_TARGET


def _get_no_split_params_via_attr(model):
    model = maybe_get_wrapped(model)
    no_split_modules = sorted(getattr(model, "_no_split_modules", None) or [])
    if len(no_split_modules) <= 0:
        return ALL_TARGET
    return no_split_modules


_sgpt_base.get_no_split_params = _get_no_split_params_via_attr

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning.magnitude import MagnitudePruningModifier
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier


def parse_args():
    parser = argparse.ArgumentParser(description="Prune a causal LM with Wanda, SparseGPT, or magnitude pruning.")
    parser.add_argument("--model", type=str, required=True, help="Path or HF hub ID of the model to prune.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the pruned model.")
    parser.add_argument(
        "--method",
        type=str,
        default="wanda",
        choices=["wanda", "sparsegpt", "magnitude"],
        help="Pruning method.",
    )
    parser.add_argument("--sparsity", type=float, default=0.4, help="Target sparsity ratio (e.g. 0.4 = 40%%).")
    parser.add_argument("--mask_structure", type=str, default=None,
                        help="Structured sparsity pattern, e.g. '2:4'. Overrides --sparsity when set.")
    parser.add_argument("--quantization_scheme", type=str, default=None,
                        help="Sparse-quantization scheme, e.g. FP8, FP8_DYNAMIC.")
    parser.add_argument(
        "--calibration_dataset",
        type=str,
        default="allenai/c4",
        help="HF dataset name for calibration (ignored if --calibration_file is set).",
    )
    parser.add_argument(
        "--calibration_file",
        type=str,
        default=None,
        help="Local JSONL file for calibration data (overrides --calibration_dataset).",
    )
    parser.add_argument("--num_calibration_samples", type=int, default=512, help="Number of calibration samples.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory.")
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output_dir)
    if (output_path / "config.json").exists() and not args.force:
        print(f"Pruned model already exists at {output_path}. Use --force to overwrite.")
        return

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    modifier_kwargs = dict(
        sparsity=args.sparsity if not args.mask_structure else None,
        mask_structure=args.mask_structure or "0:0",
    )
    if args.method == "wanda":
        modifier = WandaPruningModifier(**modifier_kwargs)
    elif args.method == "sparsegpt":
        modifier = SparseGPTModifier(**modifier_kwargs)
    elif args.method == "magnitude":
        # MagnitudePruningModifier doesn't take mask_structure (no calibration
        # pass needed either -- pure |weight| threshold).
        modifier = MagnitudePruningModifier(sparsity=args.sparsity)
    else:
        raise ValueError(f"Unsupported --method {args.method!r}")

    # llmcompressor's built-in calibration dataset registry uses short names
    # ("c4", "wikitext", "open_platypus", ...), not full HF hub paths --
    # --calibration_dataset's own default is "allenai/c4" (matching this
    # script's original CLI), so map that specific case; anything else is
    # passed through as-is (a --calibration_file local JSONL, or a dataset
    # name the caller already knows llmcompressor recognizes).
    if args.calibration_file:
        # llmcompressor's own local-file loader only scans a DIRECTORY for
        # *.json files, so a single .jsonl can't be passed by path. Load it
        # here and hand oneshot an in-memory Dataset with a "text" column.
        from datasets import load_dataset

        dataset_kwargs = {
            "dataset": load_dataset("json", data_files=args.calibration_file, split="train"),
            "text_column": "text",
        }
    elif args.calibration_dataset in ("allenai/c4", "c4"):
        # llmcompressor's C4Dataset preset loads raw HF "allenai/c4" without a
        # config name, but that dataset now requires one explicitly (it has no
        # default) -- "en" matches what Wanda/SparseGPT papers calibrate on.
        #
        # The "en" config is ~1024 shard files (hundreds of GB); a plain
        # load_dataset() downloads the whole config just to draw
        # num_calibration_samples (e.g. 32-512) from it. streaming=True avoids
        # the download, but llmcompressor 0.9.0.2's own oneshot() calibration
        # sampler (_make_sampler in llmcompressor/datasets/utils.py) calls
        # len(dataset) unconditionally, which IterableDataset (the streaming
        # type) doesn't support -- a real gap in llmcompressor's streaming
        # support, not something fixable from our side without patching their
        # sampler. So instead: pin data_files to a single real c4 shard
        # (~350MB, hundreds of thousands of documents -- far more than any
        # realistic --num_calibration_samples) and load it non-streaming. That
        # gives a real Dataset with len() support, genuine c4 text, and no
        # full-config download.
        dataset_kwargs = {
            "dataset": "c4",
            "dataset_config_name": "en",
            "splits": "train",
            "raw_kwargs": {
                "data_files": {"train": "en/c4-train.00000-of-01024.json.gz"},
                # allenai/c4's recorded dataset info declares both a train and
                # a validation split; restricting data_files to train only
                # trips datasets' post-download split-completeness check
                # (ExpectedMoreSplitsError: {'validation'}) unless verification
                # is relaxed -- we only want calibration text, not the full
                # dataset, so skip that check rather than also downloading a
                # validation shard we'd never use.
                "verification_mode": "no_checks",
            },
        }
    else:
        dataset_kwargs = {"dataset": args.calibration_dataset}

    print(f"Pruning with method={args.method}, sparsity={args.sparsity}")
    output_path.mkdir(parents=True, exist_ok=True)

    oneshot(
        model=args.model,
        tokenizer=tokenizer,
        recipe=[modifier],
        num_calibration_samples=args.num_calibration_samples,
        precision=torch_dtype,
        output_dir=str(output_path),
        **dataset_kwargs,
    )
    print(f"Pruned model saved to: {output_path}")


if __name__ == "__main__":
    main()
