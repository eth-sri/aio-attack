#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any
from itertools import islice

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AWQ quantization script.")
    p.add_argument("--model_path", type=str, default="./vulnerable", help="Base model path or HF id.")
    p.add_argument("--output_path", type=str, default="./dead_awq", help="Quantized model output directory.")
    p.add_argument("--w_bit", type=int, default=4, help="AWQ weight bit width.")
    p.add_argument("--q_group_size", type=int, default=128, help="AWQ quantization group size.")
    p.add_argument("--zero_point", action="store_true", help="Enable zero-point quantization.")
    p.add_argument("--version", type=str, default="GEMM", help="AWQ backend/version (e.g. GEMM).")
    p.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples.")
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    p.add_argument(
        "--dataset",
        type=str,
        default="c4",
        choices=["c4", "wikitext", "pile", "pile_cc", "openwebtext", "custom"],
        help="Calibration dataset preset.",
    )
    p.add_argument("--dataset_name", type=str, default=None, help="HF dataset for calibration.")
    p.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Optional HF dataset config name (e.g. wikitext-2-raw-v1).",
    )
    p.add_argument("--dataset_split", type=str, default=None, help="Dataset split for calibration.")
    p.add_argument("--dataset_text_field", type=str, default=None, help="Text field used for calibration.")
    p.add_argument(
        "--dataset_data_files",
        type=str,
        default=None,
        help="Optional data_files path for datasets.load_dataset.",
    )
    p.add_argument("--dataset_cache_dir", type=str, default="~/Datasets", help="datasets cache dir.")
    p.add_argument(
        "--allow_full_dataset_fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow fallback retry without data_files when initial dataset load fails. "
            "Disabled by default to avoid scanning huge full datasets unexpectedly."
        ),
    )
    p.add_argument(
        "--max_calib_seq_len",
        type=int,
        default=512,
        help="Max calibration sequence length.",
    )
    return p.parse_args()


def patch_awq_for_qwen3_compat() -> None:
    from torch import nn
    from awq.quantize import quantizer as awq_quantizer

    if getattr(awq_quantizer.AwqQuantizer.init_quant, "_qwen3_compat_patch", False):
        return

    def patched_init_quant(self, n_samples=128, max_seq_len=512):
        modules = self.awq_model.get_model_layers(self.model)
        samples = awq_quantizer.get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split,
            text_column=self.text_column,
        )
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        best_device = awq_quantizer.get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.module, name)

            def forward(self, *args, **kwargs):
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError

        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:
            pass
        modules[0] = modules[0].module

        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        awq_quantizer.clear_memory()

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(best_device)
        elif "qwen" in self.awq_model.model_type:
            layer_kwargs["attention_mask"] = None

        return modules, layer_kwargs, inps

    patched_init_quant._qwen3_compat_patch = True
    awq_quantizer.AwqQuantizer.init_quant = patched_init_quant


def main() -> None:
    args = parse_args()

    try:
        from awq import AutoAWQForCausalLM
    except Exception as exc:
        raise ImportError(
            "AutoAWQ is required. Install it first, e.g. `pip install autoawq`."
        ) from exc

    patch_awq_for_qwen3_compat()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoAWQForCausalLM.from_pretrained(args.model_path)

    if args.dataset == "c4":
        dataset_name = "allenai/c4"
        dataset_config_name = "en"
        dataset_split = "train"
        dataset_text_field = "text"
        dataset_data_files = "en/c4-train.00001-of-01024.json.gz"
    elif args.dataset == "wikitext":
        dataset_name = "wikitext"
        dataset_config_name = "wikitext-103-raw-v1"
        dataset_split = "train"
        dataset_text_field = "text"
        dataset_data_files = None
    elif args.dataset in {"pile", "pile_cc"}:
        # Keep legacy behavior stable and lightweight.
        dataset_name = "NeelNanda/pile-10k"
        dataset_config_name = None
        dataset_split = "train"
        dataset_text_field = "text"
        dataset_data_files = None
    elif args.dataset == "openwebtext":
        dataset_name = "Skylion007/openwebtext"
        dataset_config_name = None
        dataset_split = "train"
        dataset_text_field = "text"
        dataset_data_files = None
    else:
        dataset_name = args.dataset_name
        dataset_config_name = args.dataset_config_name
        dataset_split = args.dataset_split
        dataset_text_field = args.dataset_text_field
        dataset_data_files = args.dataset_data_files

    if args.dataset_name is not None:
        dataset_name = args.dataset_name
    if args.dataset_config_name is not None:
        dataset_config_name = args.dataset_config_name
    if args.dataset_split is not None:
        dataset_split = args.dataset_split
    if args.dataset_text_field is not None:
        dataset_text_field = args.dataset_text_field
    if args.dataset_data_files is not None:
        dataset_data_files = args.dataset_data_files

    cache_dir = os.path.expanduser(args.dataset_cache_dir)
    dataset_path_obj = Path(str(dataset_name)).expanduser()
    data_files_obj = Path(str(dataset_data_files)).expanduser() if dataset_data_files else None

    # Robust dataset loading for both HF ids and local JSON/JSONL calibration files.
    use_streaming = args.dataset in {"c4", "openwebtext", "pile", "pile_cc"}
    dataset = None
    if dataset_path_obj.exists() and dataset_path_obj.is_file():
        dataset = load_dataset(
            "json",
            data_files=str(dataset_path_obj),
            split="train",
            cache_dir=cache_dir,
            streaming=use_streaming,
        )
    elif data_files_obj is not None and data_files_obj.exists() and data_files_obj.is_file():
        dataset = load_dataset(
            "json",
            data_files=str(data_files_obj),
            split="train",
            cache_dir=cache_dir,
            streaming=use_streaming,
        )
    else:
        attempts: list[dict[str, Any]] = []
        base_kwargs: dict[str, Any] = {
            "path": dataset_name,
            "split": dataset_split,
            "cache_dir": cache_dir,
            "streaming": use_streaming,
        }
        if dataset_config_name:
            base_kwargs["name"] = dataset_config_name
        # Attempt 1: exactly as requested/resolved.
        if dataset_data_files:
            k1 = dict(base_kwargs)
            k1["data_files"] = dataset_data_files
            attempts.append(k1)
        else:
            attempts.append(dict(base_kwargs))
        # Attempt 2 (fallback): no data_files (opt-in only; can be very expensive).
        if dataset_data_files and args.allow_full_dataset_fallback:
            attempts.append(dict(base_kwargs))

        last_exc: Exception | None = None
        dataset = None
        for i, kwargs in enumerate(attempts, start=1):
            try:
                print(
                    f"loading_calib_dataset_attempt={i} "
                    f"path={kwargs.get('path')} name={kwargs.get('name')} "
                    f"split={kwargs.get('split')} data_files={kwargs.get('data_files')}"
                )
                dataset = load_dataset(**kwargs)
                break
            except Exception as exc:  # pragma: no cover - external state dependent
                last_exc = exc
                continue
        if dataset is None:
            raise RuntimeError(
                "Failed to load calibration dataset for AWQ after fallbacks. "
                f"path={dataset_name!r} config={dataset_config_name!r} split={dataset_split!r} "
                f"data_files={dataset_data_files!r} cache_dir={cache_dir!r}. "
                "If using a local file, pass --dataset custom --dataset_name /path/to/file.jsonl. "
                "If you intentionally want retry without data_files, add --allow_full_dataset_fallback."
            ) from last_exc

    if use_streaming:
        # Streaming avoids scanning entire giant corpora just to get nsamples.
        if hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed=args.seed, buffer_size=max(args.nsamples * 4, 1024))
        rows = list(islice(dataset, int(args.nsamples)))
        if not rows:
            raise ValueError("Calibration dataset yielded zero rows in streaming mode.")
    else:
        dataset = dataset.shuffle(seed=args.seed)
        dataset = dataset.select(range(min(args.nsamples, len(dataset))))
        rows = list(dataset)
    text_field = dataset_text_field
    if use_streaming:
        column_names = list(rows[0].keys())
    else:
        column_names = list(dataset.column_names)
    if text_field not in column_names:
        for candidate in ("text", "prompt", "instruction", "input", "question", "content"):
            if candidate in column_names:
                text_field = candidate
                print(
                    f"warning: dataset_text_field={dataset_text_field!r} not found; "
                    f"using fallback field {text_field!r}"
                )
                break
        else:
            raise ValueError(
                f"dataset_text_field={dataset_text_field!r} not found. "
                f"Available columns: {column_names}"
            )
    calib_data = [str(row.get(text_field, "")) for row in rows if str(row.get(text_field, "")).strip()]

    version = args.version.upper()
    if not args.zero_point and version in {"GEMM", "GEMV", "GEMV_FAST"}:
        print(
            "warning: enabling zero_point because the installed AWQ backend "
            f"requires it for version={version}"
        )
        args.zero_point = True

    quant_config = {
        "w_bit": args.w_bit,
        "q_group_size": args.q_group_size,
        "zero_point": args.zero_point,
        "version": version,
    }
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data,
        max_calib_seq_len=args.max_calib_seq_len,
    )

    model.save_quantized(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    tokenizer_cfg_path = Path(args.output_path) / "tokenizer_config.json"
    if tokenizer_cfg_path.is_file():
        with tokenizer_cfg_path.open("r", encoding="utf-8") as f:
            tokenizer_cfg = json.load(f)
        if tokenizer_cfg.get("tokenizer_class") == "TokenizersBackend":
            tokenizer_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            with tokenizer_cfg_path.open("w", encoding="utf-8") as f:
                json.dump(tokenizer_cfg, f, indent=2, ensure_ascii=True)
                f.write("\n")

    print(f"model_path={args.model_path}")
    print(f"output_path={args.output_path}")
    print(f"w_bit={args.w_bit}")
    print(f"q_group_size={args.q_group_size}")
    print(f"zero_point={args.zero_point}")
    print(f"version={args.version}")
    print(f"nsamples={len(calib_data)}")
    print("saved=True")


if __name__ == "__main__":
    main()
