#!/usr/bin/env python3
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[value]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BitsAndBytes quantization (4-bit/8-bit).")
    p.add_argument("--model_path", type=str, default="./vulnerable", help="Base model path or HF id.")
    p.add_argument("--output_path", type=str, default="./dead_bnb", help="Quantized model output directory.")
    p.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "8bit"],
        default="4bit",
        help="Quantization mode.",
    )
    p.add_argument(
        "--compute_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Compute dtype for 4-bit mode.",
    )
    p.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        choices=["nf4", "fp4"],
        default="nf4",
        help="4-bit quant type.",
    )
    p.add_argument(
        "--bnb_4bit_use_double_quant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable 4-bit double quantization.",
    )
    p.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Hugging Face device_map value.",
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    p.add_argument(
        "--llm_int8_threshold",
        type=float,
        default=6.0,
        help="Outlier threshold for bitsandbytes int8 path.",
    )
    p.add_argument(
        "--llm_int8_enable_fp32_cpu_offload",
        action="store_true",
        help="Enable fp32 CPU offload for int8 modules.",
    )
    p.add_argument(
        "--llm_int8_has_fp16_weight",
        action="store_true",
        help="Set llm_int8_has_fp16_weight=True in BitsAndBytesConfig.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    compute_dtype = parse_dtype(args.compute_dtype)

    if args.quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=args.llm_int8_threshold,
            llm_int8_enable_fp32_cpu_offload=args.llm_int8_enable_fp32_cpu_offload,
            llm_int8_has_fp16_weight=args.llm_int8_has_fp16_weight,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Some eval stacks reload int8 checkpoints with bf16 from model config.
    # Force fp16 preference for int8 checkpoints to avoid unstable bf16->fp16 kernel paths.
    if args.quantization == "8bit":
        model.config.torch_dtype = "float16"

    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    print(f"model_path={args.model_path}")
    print(f"output_path={args.output_path}")
    print(f"quantization={args.quantization}")
    if args.quantization == "4bit":
        print(f"compute_dtype={args.compute_dtype}")
        print(f"bnb_4bit_quant_type={args.bnb_4bit_quant_type}")
        print(f"bnb_4bit_use_double_quant={args.bnb_4bit_use_double_quant}")
    else:
        print(f"llm_int8_threshold={args.llm_int8_threshold}")
        print(f"llm_int8_enable_fp32_cpu_offload={args.llm_int8_enable_fp32_cpu_offload}")
        print(f"llm_int8_has_fp16_weight={args.llm_int8_has_fp16_weight}")
        print("config_torch_dtype=float16")
    print("saved=True")


if __name__ == "__main__":
    main()
