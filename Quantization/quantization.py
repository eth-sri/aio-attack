#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys


SUPPORTED_METHODS = {
    "gptq",
    "awq",
    "awq8",
    "int8",
    "nf4",
    "fp4",
    "bnb",
    "bitsandbytes",
    "gguf",
    "gguf_k",
    "gguf_0",
    "gguf_i",
}

METHOD_ALIASES = {
    "bitsandbytes": "bnb",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Unified quantization entrypoint for non-GGUF methods. "
            "Dispatches to GPTQ/AWQ/BitsAndBytes/GGUF scripts."
        )
    )
    parser.add_argument("--model_path", type=str, required=True, help="Base model path or HF id.")
    parser.add_argument("--output_path", type=str, required=True, help="Quantized model output path.")
    parser.add_argument(
        "--method",
        "--quant_method",
        dest="method",
        type=str,
        default="gptq",
        help=(
            "Quantization method: gptq, "
            "awq, awq8, int8, nf4, fp4, bnb."
        ),
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Primary bit width used by GPTQ/AWQ and bnb method resolution.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Primary group size used by GPTQ/AWQ.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="BitsAndBytes mode override when --method is bnb/bitsandbytes.",
    )
    parser.add_argument(
        "--gguf_quant_type",
        type=str,
        default=None,
        help="GGUF quant type override, e.g. Q4_K_M, Q4_0, IQ4_XS.",
    )
    parser.add_argument(
        "--gguf_outtype",
        type=str,
        default="f16",
        choices=["f16", "f32", "bf16", "q8_0", "auto"],
        help="Output type for convert_hf_to_gguf.py when --method is GGUF.",
    )
    parser.add_argument(
        "--gguf_model_name",
        type=str,
        default="model",
        help="Base name used when --output_path is a directory in GGUF mode.",
    )
    parser.add_argument("--llama_cpp_dir", type=str, default=None, help="Optional llama.cpp directory for GGUF tools.")
    parser.add_argument("--converter", type=str, default=None, help="Optional path to convert_hf_to_gguf.py.")
    parser.add_argument("--quantize_bin", type=str, default=None, help="Optional path to llama-quantize binary.")
    parser.add_argument(
        "--imatrix",
        type=str,
        default=None,
        help="Optional imatrix file path for GGUF IQ quantization.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def _has_flag(args: list[str], flag: str) -> bool:
    return flag in args


def _run(cmd: list[str]) -> None:
    print("running:", " ".join(cmd))
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


def build_command(args: argparse.Namespace, passthrough: list[str]) -> tuple[str, list[str]]:
    raw_method = args.method.lower()
    method = METHOD_ALIASES.get(raw_method, raw_method)
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            "Unsupported method. Choose one of: gptq, awq, awq8, int8, nf4, fp4, bnb, bitsandbytes, gguf, gguf_k, gguf_0, gguf_i."
        )
    if method in {"gguf", "gguf_k", "gguf_0", "gguf_i"}:
        script = "Quantization/quantization_gguf.py"
        default_quant_type = "Q4_K_M"
        if method == "gguf_0":
            default_quant_type = "Q4_0"
        elif method == "gguf_i":
            default_quant_type = "IQ4_XS"
        quant_type = args.gguf_quant_type or default_quant_type
        mapped = [
            "--model_path",
            args.model_path,
            "--output_path",
            args.output_path,
            "--quant_type",
            quant_type,
            "--outtype",
            args.gguf_outtype,
            "--model_name",
            args.gguf_model_name,
        ]
        if args.llama_cpp_dir:
            mapped.extend(["--llama_cpp_dir", args.llama_cpp_dir])
        if args.converter:
            mapped.extend(["--converter", args.converter])
        if args.quantize_bin:
            mapped.extend(["--quantize_bin", args.quantize_bin])
        if args.imatrix:
            mapped.extend(["--imatrix", args.imatrix])
        return script, mapped + passthrough

    common = ["--model_path", args.model_path, "--output_path", args.output_path]

    if method == "gptq":
        script = "Quantization/quantization_gptq.py"
        mapped = common + [
            "--seed",
            str(args.seed),
            "--bits",
            str(args.bits),
            "--group_size",
            str(args.group_size),
        ]
        return script, mapped + passthrough

    if method in {"awq", "awq8"}:
        script = "Quantization/quantization_awq.py"
        awq_bits = 8 if method == "awq8" else args.bits
        mapped = common + [
            "--seed",
            str(args.seed),
            "--w_bit",
            str(awq_bits),
            "--q_group_size",
            str(args.group_size),
        ]
        return script, mapped + passthrough

    script = "Quantization/quantization_bnb.py"
    if method in {"bnb", "bitsandbytes"}:
        quant_mode = args.quantization or ("8bit" if args.bits == 8 else "4bit")
        mapped = common + ["--quantization", quant_mode]
    elif method == "int8":
        mapped = common + ["--quantization", "8bit"]
    else:
        mapped = common + ["--quantization", "4bit", "--bnb_4bit_quant_type", method]
    return script, mapped + passthrough


def main() -> None:
    args, passthrough = parse_args()
    script, script_args = build_command(args, passthrough)
    cmd = [sys.executable, script] + script_args
    _run(cmd)


if __name__ == "__main__":
    main()
