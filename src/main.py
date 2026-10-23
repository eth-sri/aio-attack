#!/usr/bin/env python
"""Config-file entry point for the dual-objective attack training pipeline.

    python src/main.py <config.json> [--flag value ...]

Runs the exact same pipeline as `python src/train.py`, but every
setting comes from a single JSON file instead of a long CLI. Any extra
`--flag value` tokens after the config path override the corresponding
config entry (handy for sweeps: one base config + per-run overrides).

Config format
-------------
Either a flat object whose keys are `train.py` flag names (without the
leading `--`), or an object with those under an `"args"` key plus arbitrary
documentation keys anywhere (any key starting with `_` is ignored):

    {
      "_meta": {"run": "mcd_new"},
      "args": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "dataset_a": "dataset/mcd_chosen.jsonl",
        "dataset_b": "dataset/mcd_rejected.jsonl",
        "output_dir": "outputs/mcd_new/model",
        "noise_type": "combined",
        "combined_mechanisms": ["rms", "wanda", "merge", "blockdrop"],
        "mechanism_weights": {"merge": 0.1, "rms": 0.5, "wanda": 0.5, "blockdrop": 0.1},
        "use_lora": true, "lora_r": 256, "lora_alpha": 256,
        "num_train_epochs": 2
      }
    }

Value conventions:
  - list flags (`combined_mechanisms`, `merge_scales`, `noise_target_modules`,
    `alpha_values`) are JSON arrays.
  - `mechanism_weights` may be either a JSON object `{"rms": 0.5, ...}` or the
    CLI-style array `["rms=0.5", ...]`.
  - boolean flags (`use_lora`, `normalize_combined_loss`, `finetune_rms`,
    `merge_all_scales`, ...) are JSON `true` / `false`.
  - anything omitted falls back to `train.py`'s default, and all of
    `train.py`'s validation / auto-defaulting still runs.
"""

import argparse
import json
import sys

from train import build_parser
from train import main as run_training


def _flatten(raw):
    """Pull the flag dict out of a config: an explicit `args` block if present,
    otherwise the top-level object minus any `_`-prefixed doc keys."""
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    if isinstance(raw.get("args"), dict):
        cfg = dict(raw["args"])
    else:
        cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
    mw = cfg.get("mechanism_weights")
    if isinstance(mw, dict):
        cfg["mechanism_weights"] = [f"{k}={v}" for k, v in mw.items()]
    return cfg


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = _flatten(json.load(f))
    parser = build_parser()
    valid = {a.dest for a in parser._actions if a.dest not in ("help",)}
    unknown = sorted(set(cfg) - valid)
    if unknown:
        raise SystemExit(
            f"[main] unknown config key(s): {unknown}\n"
            f"valid keys: {sorted(valid)}"
        )
    return parser, cfg


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python src/main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("config", help="path to the training config .json")
    ap.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="optional trailing '--flag value' pairs that override the config",
    )
    return ap.parse_args(argv)


def main(argv=None):
    ns = parse_args(argv)
    parser, cfg = load_config(ns.config)
    parser.set_defaults(**cfg)
    args = parser.parse_args(ns.overrides)  # CLI overrides win over the config
    print(f"[main] config={ns.config}"
          + (f" overrides={ns.overrides}" if ns.overrides else ""), flush=True)
    run_training(args)


if __name__ == "__main__":
    main()
