#!/usr/bin/env python3
"""Unified pruning dispatch — mirrors Quantization/quantization.py and Merging/merging.py.

Dispatches to prune_wanda.py (llmcompressor / Wanda / SparseGPT). Set
PRUNE_PYTHON to run it under a different interpreter.

Supported methods
-----------------
  wanda       Wanda unstructured pruning (activation-weighted magnitude)
  sparsegpt   SparseGPT unstructured pruning (second-order Hessian-based)
  magnitude   Simple magnitude pruning (no calibration data needed)

Structured / sparse-quantized variants are passed through via --mask_structure
and --quantization_scheme respectively.
"""
import argparse
import os
import signal
import subprocess
import sys

TEST_PYTHON = os.environ.get(
    "PRUNE_PYTHON",
    sys.executable,
)

SUPPORTED_METHODS = {"wanda", "sparsegpt", "magnitude"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified pruning entrypoint. Dispatches to prune_wanda.py."
    )
    p.add_argument("--model_path",  required=True, help="Model path or HF hub ID to prune.")
    p.add_argument("--output_path", required=True, help="Directory to save the pruned model.")
    p.add_argument(
        "--method",
        default="wanda",
        choices=sorted(SUPPORTED_METHODS),
        help="Pruning method.",
    )
    p.add_argument("--sparsity", type=float, default=0.4,
                   help="Target sparsity ratio (e.g. 0.4 = 40%%).")
    p.add_argument("--mask_structure", default=None,
                   help="Structured sparsity pattern, e.g. '2:4'. "
                        "Overrides --sparsity when set.")
    p.add_argument("--calibration_dataset", default="allenai/c4",
                   help="HF dataset name for calibration.")
    p.add_argument("--calibration_file", default=None,
                   help="Local JSONL file for calibration (overrides --calibration_dataset).")
    p.add_argument("--num_calibration_samples", type=int, default=512)
    p.add_argument("--quantization_scheme", default=None,
                   help="Sparse-quantization scheme, e.g. FP8, FP8_DYNAMIC.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output directory.")
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
    if args.method == "magnitude":
        # prune_wanda.py's magnitude path goes through llmcompressor's
        # MagnitudePruningModifier, whose leave_enabled field is hard-deprecated
        # to always resolve False upstream -- it applies the mask transiently
        # then restores the original weights before save, so the "pruned"
        # checkpoint comes out byte-identical to the input. prune_magnitude_direct.py
        # bypasses that modifier entirely and zeroes the weights in place.
        cmd = [
            TEST_PYTHON, "Pruning/prune_magnitude_direct.py",
            "--model",      args.model_path,
            "--output_dir", args.output_path,
            "--sparsity",   str(args.sparsity),
        ]
        if args.force:
            cmd.append("--force")
        return cmd + passthrough

    script = "Pruning/prune_wanda.py"
    cmd = [
        TEST_PYTHON, script,
        "--model",      args.model_path,
        "--output_dir", args.output_path,
        "--method",     args.method,
        "--dtype",      args.dtype,
        "--num_calibration_samples", str(args.num_calibration_samples),
    ]

    if args.mask_structure:
        cmd += ["--mask_structure", args.mask_structure]
    else:
        cmd += ["--sparsity", str(args.sparsity)]

    if args.calibration_file:
        cmd += ["--calibration_file", args.calibration_file]
    else:
        cmd += ["--calibration_dataset", args.calibration_dataset]

    if args.quantization_scheme:
        cmd += ["--quantization_scheme", args.quantization_scheme]

    if args.force:
        cmd.append("--force")

    return cmd + passthrough


def main():
    args, passthrough = parse_args()
    cmd = build_command(args, passthrough)
    _run(cmd)


if __name__ == "__main__":
    main()
