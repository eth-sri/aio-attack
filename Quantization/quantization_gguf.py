#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GGUF quantization via llama.cpp tools.")
    p.add_argument("--model_path", type=str, default="./vulnerable", help="Base HF model path or HF id.")
    p.add_argument(
        "--output_path",
        type=str,
        default="./dead_gguf",
        help="Output GGUF file path or output directory.",
    )
    p.add_argument(
        "--quant_type",
        type=str,
        default="Q4_K_M",
        help="GGUF quantization type, e.g. Q4_K_M, Q4_0, Q5_K_M, Q8_0.",
    )
    p.add_argument(
        "--outtype",
        type=str,
        default="f16",
        choices=["f16", "f32", "bf16", "q8_0", "auto"],
        help="Output type for convert_hf_to_gguf.py.",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="model",
        help="Used when --output_path is a directory.",
    )
    p.add_argument(
        "--llama_cpp_dir",
        type=str,
        default=None,
        help="Optional llama.cpp directory for auto tool lookup.",
    )
    p.add_argument(
        "--converter",
        type=str,
        default=None,
        help="Path to convert_hf_to_gguf.py (overrides auto lookup).",
    )
    p.add_argument(
        "--quantize_bin",
        type=str,
        default=None,
        help="Path to llama-quantize or quantize binary (overrides auto lookup).",
    )
    p.add_argument(
        "--imatrix",
        type=str,
        default=None,
        help="Optional imatrix file path forwarded to llama-quantize via --imatrix.",
    )
    p.add_argument(
        "--base_gguf",
        type=str,
        default=None,
        help=(
            "Optional prebuilt base GGUF path (typically F16/BF16). "
            "If set, skip HF->GGUF conversion and quantize from this file."
        ),
    )
    return p.parse_args()


def _find_converter(args: argparse.Namespace) -> Path:
    if args.converter:
        path = Path(args.converter)
        if not path.is_file():
            raise FileNotFoundError(f"converter not found: {path}")
        return path

    candidates = []
    if args.llama_cpp_dir:
        candidates.append(Path(args.llama_cpp_dir) / "convert_hf_to_gguf.py")
    candidates.extend([Path("convert_hf_to_gguf.py"), Path("llama.cpp/convert_hf_to_gguf.py")])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find convert_hf_to_gguf.py. Use --converter or --llama_cpp_dir."
    )


def _find_quantize_bin(args: argparse.Namespace) -> Path:
    if args.quantize_bin:
        path = Path(args.quantize_bin)
        if not path.is_file():
            raise FileNotFoundError(f"quantize binary not found: {path}")
        return path

    candidates = []
    if args.llama_cpp_dir:
        base = Path(args.llama_cpp_dir)
        candidates.extend(
            [
                base / "build/bin/llama-quantize",
                base / "build/bin/quantize",
                base / "llama-quantize",
                base / "quantize",
            ]
        )
    candidates.extend([Path("llama.cpp/build/bin/llama-quantize"), Path("llama.cpp/build/bin/quantize")])
    for path in candidates:
        if path.is_file():
            return path

    which_llama_quantize = shutil.which("llama-quantize")
    if which_llama_quantize:
        return Path(which_llama_quantize)
    which_quantize = shutil.which("quantize")
    if which_quantize:
        return Path(which_quantize)
    raise FileNotFoundError(
        "Could not find llama.cpp quantize binary. Use --quantize_bin or --llama_cpp_dir."
    )


def _run(cmd: list[str]) -> None:
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def _package_hf_sidecars(model_path: str, package_dir: Path) -> list[str]:
    src_dir = Path(model_path)
    if not src_dir.is_dir():
        return []

    package_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[str] = []
    candidate_names = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "chat_template.jinja",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ]
    for name in candidate_names:
        src = src_dir / name
        dst = package_dir / name
        if _copy_if_exists(src, dst):
            copied_paths.append(name)

    for pattern in ("*.spm", "*.tiktoken", "*.txt"):
        for src in sorted(src_dir.glob(pattern)):
            dst = package_dir / src.name
            if dst.exists():
                continue
            if _copy_if_exists(src, dst):
                copied_paths.append(src.name)

    return copied_paths


def main() -> None:
    args = parse_args()

    output_path = Path(args.output_path)
    if output_path.suffix.lower() == ".gguf":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        quantized_gguf = output_path
        base_gguf = output_path.parent / f"{output_path.stem}.{args.outtype.upper()}.gguf"
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        quantized_gguf = output_path / f"{args.model_name}.{args.quant_type.upper()}.gguf"
        base_gguf = output_path / f"{args.model_name}.{args.outtype.upper()}.gguf"

    quantize_bin = _find_quantize_bin(args)
    used_base_gguf = base_gguf
    if args.base_gguf:
        used_base_gguf = Path(args.base_gguf)
        if not used_base_gguf.is_file():
            raise FileNotFoundError(f"--base_gguf not found: {used_base_gguf}")
    else:
        converter = _find_converter(args)
        convert_cmd = [
            sys.executable,
            str(converter),
            str(args.model_path),
            "--outfile",
            str(base_gguf),
            "--outtype",
            args.outtype,
        ]
        _run(convert_cmd)
        used_base_gguf = base_gguf

    quant_cmd = [str(quantize_bin)]
    if args.imatrix:
        quant_cmd.extend(["--imatrix", str(args.imatrix)])
    quant_cmd.extend(
        [
            str(used_base_gguf),
            str(quantized_gguf),
            args.quant_type.upper(),
        ]
    )
    _run(quant_cmd)

    packaged_files = _package_hf_sidecars(args.model_path, quantized_gguf.parent)
    tokenizer_arg = (
        str(quantized_gguf.parent)
        if any(
            name.startswith("tokenizer")
            or name in {"special_tokens_map.json", "chat_template.jinja", "added_tokens.json"}
            for name in packaged_files
        )
        else None
    )
    lm_eval_model_args = [
        f"pretrained={quantized_gguf.parent}",
        f"gguf_file={quantized_gguf.name}",
    ]
    if tokenizer_arg is not None:
        lm_eval_model_args.append(f"tokenizer={tokenizer_arg}")

    print(f"model_path={args.model_path}")
    print(f"base_gguf={used_base_gguf}")
    print(f"output_path={quantized_gguf}")
    print(f"quant_type={args.quant_type.upper()}")
    print(f"outtype={args.outtype}")
    if packaged_files:
        print(f"packaged_sidecars={','.join(packaged_files)}")
    else:
        print("packaged_sidecars=none")
        if not Path(args.model_path).is_dir():
            print(
                "warning=model_path is not a local directory, so tokenizer/config sidecars could not be copied automatically"
            )
    print(f"lm_eval_model_args={','.join(lm_eval_model_args)}")
    print(
        "lm_eval_example="
        + "lm_eval --model hf --model_args "
        + ",".join(lm_eval_model_args)
        + " --tasks hellaswag --device cuda:0 --batch_size 8"
    )
    print("saved=True")


if __name__ == "__main__":
    main()
