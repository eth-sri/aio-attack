#!/usr/bin/env python3
"""Unified mergekit dispatch — mirrors Quantization/quantization.py.

Dispatches to individual merge scripts (merge_linear.py,
merge_ties.py). Set MERGEKIT_PYTHON to run them under a separate mergekit env.

Supported methods
-----------------
  linear, task_arithmetic          → merge_linear.py
  ties, dare_ties, dare_linear,
  breadcrumbs, breadcrumbs_ties,
  della, della_linear              → merge_ties.py
"""
import argparse
import os
import signal
import subprocess
import sys

MERGEKIT_PYTHON = os.environ.get(
    "MERGEKIT_PYTHON",
    sys.executable,
)

LINEAR_METHODS = {"linear", "task_arithmetic"}
TIES_METHODS   = {
    "ties", "dare_ties", "dare_linear",
    "breadcrumbs", "breadcrumbs_ties",
    "della", "della_linear",
}
ALL_METHODS = LINEAR_METHODS | TIES_METHODS


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified mergekit dispatch for linear / TIES-family methods."
    )
    p.add_argument("--model_path",   required=True,
                   help="Primary (trained) model path.")
    p.add_argument("--output_path",  required=True)
    p.add_argument("--method",       required=True, choices=sorted(ALL_METHODS),
                   help="Merge method.")
    p.add_argument("--base_model",   default=None,
                   help="Base model (required for task_arithmetic and TIES-family).")
    p.add_argument("--donor_paths",  nargs="*", default=[],
                   help="Donor model path(s) to merge with.")
    # linear / task_arithmetic
    p.add_argument("--weight",       type=float, default=1.0,
                   help="Weight for --model_path (linear / task_arithmetic).")
    p.add_argument("--donor_weight", type=float, default=1.0,
                   help="Weight for each donor (linear / task_arithmetic / TIES).")
    p.add_argument("--normalize",    action="store_true", default=True,
                   help="Normalize weights to sum to 1 (linear only).")
    p.add_argument("--no_normalize", dest="normalize", action="store_false")
    # ties-family
    p.add_argument("--density",      type=float, default=0.53,
                   help="Task-vector density fraction for TIES-family methods.")
    p.add_argument("--rescale",      type=float, default=1.0)
    p.add_argument("--int8_mask",    action="store_true", default=False)
    # shared
    p.add_argument("--dtype",   default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--no_cuda", action="store_true", default=False)
    args, passthrough = p.parse_known_args()
    return args, passthrough


def _run(cmd: list[str]) -> None:
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        raise
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def build_command(args, passthrough: list[str]) -> list[str]:
    method = args.method
    common = [
        "--model_path",   args.model_path,
        "--output_path",  args.output_path,
        "--dtype",        args.dtype,
    ]
    if args.no_cuda:
        common.append("--no_cuda")

    if method in LINEAR_METHODS:
        script = "Merging/merge_linear.py"
        extra = [
            "--merge_method",  method,
            "--weight",        str(args.weight),
            "--donor_weight",  str(args.donor_weight),
        ]
        if args.donor_paths:
            extra += ["--donor_paths"] + args.donor_paths
        if args.base_model:
            extra += ["--base_model", args.base_model]
        if not args.normalize:
            extra.append("--no_normalize")

    else:  # TIES_METHODS
        if not args.base_model:
            raise ValueError(f"--base_model is required for {method}.")
        script = "Merging/merge_ties.py"
        extra = [
            "--merge_method",  method,
            "--base_model",    args.base_model,
            "--weight",        str(args.weight),
            "--donor_weight",  str(args.donor_weight),
            "--density",       str(args.density),
            "--rescale",       str(args.rescale),
        ]
        if args.donor_paths:
            extra += ["--donor_paths"] + args.donor_paths
        if args.int8_mask:
            extra.append("--int8_mask")

    return [MERGEKIT_PYTHON, script] + common + extra + passthrough


def main():
    args, passthrough = parse_args()
    cmd = build_command(args, passthrough)
    _run(cmd)


if __name__ == "__main__":
    main()
