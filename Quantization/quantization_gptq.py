#!/usr/bin/env python3
import argparse
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import torch
from datasets.exceptions import ExpectedMoreSplitsError
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPTQModel quantization (minimal).")
    p.add_argument("--model_path", type=str, default="./vulnerable", help="Base model path or HF id.")
    p.add_argument("--output_path", type=str, default="./dead", help="Quantized model output directory.")
    p.add_argument("--bits", type=int, default=4, help="Quantization bits.")
    p.add_argument("--group_size", type=int, default=128, help="Quantization group size.")
    p.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples.")
    p.add_argument("--batch_size", type=int, default=1, help="Quantization batch size.")
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    p.add_argument(
        "--format",
        type=str,
        default="gptq",
        choices=["gptq", "gptq_v2", "bitblas", "qqq", "gemm", "gemv", "gemv_fast", "llm-awq"],
        help="GPTQModel output format/backend.",
    )
    p.add_argument("--dataset_cache_dir", type=str, default="~/Datasets")
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
    p.add_argument(
        "--dataset_text_field",
        type=str,
        default=None,
        help="Text field name in the calibration dataset.",
    )
    p.add_argument(
        "--dataset_data_files",
        type=str,
        default=None,
        help="Optional data_files argument for datasets.load_dataset.",
    )
    p.add_argument(
        "--calibration_texts_file",
        type=str,
        default=None,
        help="Optional calibration file (overrides HF dataset args): JSON lines with a 'text' field, "
        "or newline-delimited text.",
    )
    p.add_argument(
        "--offload_to_disk",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable GPTQModel offload-to-disk mode. "
            "Default false to avoid mmap-heavy turtle reloads."
        ),
    )
    p.add_argument(
        "--offload_to_disk_path",
        type=str,
        default=None,
        help="Optional offload directory used when --offload_to_disk is enabled.",
    )
    p.add_argument(
        "--reload_threshold",
        type=str,
        default="0",
        help=(
            "GPTQMODEL_RELOAD_THRESHOLD value (e.g. 0, 512MB, 2GB). "
            "Default 0 disables auto reload to reduce mmap failures."
        ),
    )
    p.add_argument(
        "--pytorch_alloc_conf",
        type=str,
        default=None,
        help=(
            "Optional override for PYTORCH_CUDA_ALLOC_CONF. "
            "Example: expandable_segments:True,max_split_size_mb:256"
        ),
    )
    p.add_argument(
        "--max_shard_size",
        type=str,
        default="4GB",
        help="Shard size passed to model.save(...).",
    )
    p.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Silence GPTQ console logs and remove generated gptq_log_*.log files.",
    )
    p.add_argument(
        "--cleanup_offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove newly created gptqmodel_offload* files/directories after quantization.",
    )
    return p.parse_args()


def _fuse_phi3_gate_up(out_dir: str) -> None:
    """vLLM's Phi3 expects the fused mlp.gate_up_proj; gptqmodel (transformers 5.x) saved separate
    gate_proj/up_proj. GPTQ quantizes output columns independently, so concatenating the packed
    qweight/scales/qzeros along the output dim reproduces the fused layer exactly."""
    import glob
    import json
    from safetensors import safe_open
    from safetensors.torch import save_file

    files = sorted(glob.glob(os.path.join(out_dir, "*.safetensors")))
    tensors: dict = {}
    for f in files:
        with safe_open(f, "pt") as g:
            for k in g.keys():
                tensors[k] = g.get_tensor(k)
    fused = 0
    for k in [k for k in tensors if k.endswith(".mlp.gate_proj.qweight")]:
        base = k[: -len("gate_proj.qweight")]
        for suf in ("qweight", "scales", "qzeros"):
            tensors[base + "gate_up_proj." + suf] = torch.cat(
                [tensors.pop(base + "gate_proj." + suf), tensors.pop(base + "up_proj." + suf)], dim=1
            )
        gi = tensors.pop(base + "gate_proj.g_idx", None)
        tensors.pop(base + "up_proj.g_idx", None)
        if gi is not None:
            tensors[base + "gate_up_proj.g_idx"] = gi
        fused += 1
    for f in files:
        os.remove(f)
    idx = os.path.join(out_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        os.remove(idx)
    save_file(tensors, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    print(f"[fuse] fused gate/up into gate_up_proj for {fused} layers", flush=True)


def _capture_existing_gptq_logs() -> set[Path]:
    return set(Path.cwd().glob("gptq_log_*.log"))


def _remove_new_gptq_logs(existing: set[Path]) -> int:
    removed = 0
    for path in Path.cwd().glob("gptq_log_*.log"):
        if path in existing:
            continue
        try:
            path.unlink()
            removed += 1
        except Exception:
            pass
    return removed


def _capture_existing_offload_paths() -> set[Path]:
    return set(Path.cwd().glob("gptqmodel_offload*"))


def _remove_new_offload_paths(existing: set[Path]) -> int:
    removed = 0
    for path in Path.cwd().glob("gptqmodel_offload*"):
        if path in existing:
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            removed += 1
        except Exception:
            pass
    return removed


def _configure_quiet_env() -> None:
    # Best-effort suppression for noisy third-party logs/progress.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("gptqmodel").setLevel(logging.ERROR)


def _configure_memory_env(args: argparse.Namespace) -> None:
    if args.pytorch_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.pytorch_alloc_conf
    if args.reload_threshold is not None and str(args.reload_threshold).strip() != "":
        os.environ["GPTQMODEL_RELOAD_THRESHOLD"] = str(args.reload_threshold)


def _patch_gptqmodel_tf32_guard() -> None:
    """
    Work around GPTQModel TF32 guard failures on some torch builds:
    RuntimeError: Invalid (backend, op) pair: (0, 0)
    """
    try:
        import gptqmodel.utils.torch as gptq_torch_utils
        import gptqmodel.looper.module_looper as gptq_module_looper
    except Exception:
        return

    @contextlib.contextmanager
    def _noop_guard():
        yield

    gptq_torch_utils.tf32_high_precision_guard = _noop_guard
    gptq_module_looper.tf32_high_precision_guard = _noop_guard


def _load_calibration_texts(args: argparse.Namespace) -> list[str]:
    if args.calibration_texts_file:
        calib_path = Path(args.calibration_texts_file).expanduser()
        with calib_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if calib_path.suffix in (".json", ".jsonl"):
            texts = [t for t in (json.loads(line)["text"].strip() for line in lines) if t]
        else:
            texts = lines
        if not texts:
            raise ValueError(f"No calibration texts found in {calib_path}")
        return texts[: args.nsamples]

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
    elif args.dataset == "pile":
        dataset_name = "monology/pile-uncopyrighted"
        dataset_config_name = None
        dataset_split = "train"
        dataset_text_field = "text"
        dataset_data_files = None
    elif args.dataset == "pile_cc":
        # Lightweight CC-style subset for fast calibration startup.
        dataset_name = "NeelNanda/pile-10k"
        dataset_config_name = None
        dataset_split = "train"
        dataset_text_field = "text"
        dataset_data_files = None
    elif args.dataset == "openwebtext":
        dataset_name = "openwebtext"
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

    # Explicit CLI overrides always win over preset defaults.
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

    load_kwargs: dict[str, object] = {
        "path": dataset_name,
        "split": dataset_split,
        "cache_dir": os.path.expanduser(args.dataset_cache_dir),
    }
    if dataset_config_name:
        load_kwargs["name"] = dataset_config_name
    if dataset_data_files:
        load_kwargs["data_files"] = dataset_data_files
    # Stream very large corpora to avoid long preprocessing stalls.
    if args.dataset in {"pile", "openwebtext"}:
        ds = load_dataset(**load_kwargs, streaming=True)
        texts: list[str] = []
        for row in ds:
            t = row.get(dataset_text_field)
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
                if len(texts) >= args.nsamples:
                    break
        if not texts:
            raise ValueError(f"No calibration texts found in streamed dataset field '{dataset_text_field}'.")
        return texts

    try:
        ds = load_dataset(**load_kwargs).shuffle(seed=args.seed)
        ds = ds.select(range(min(args.nsamples, len(ds))))
        if dataset_text_field not in ds.column_names:
            raise ValueError(
                f"Dataset text field '{dataset_text_field}' missing. "
                f"Available columns: {list(ds.column_names)}"
            )
        return [str(t) for t in ds[dataset_text_field]]
    except ExpectedMoreSplitsError:
        # Some dataset builders (notably C4) now validate canonical split metadata
        # even when data_files points at a single train shard.
        fallback_kwargs = dict(load_kwargs)
        fallback_kwargs.pop("data_files", None)
        ds = load_dataset(**fallback_kwargs, streaming=True)
        texts: list[str] = []
        for row in ds:
            t = row.get(dataset_text_field)
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
                if len(texts) >= args.nsamples:
                    break
        if not texts:
            raise ValueError(f"No calibration texts found in streamed dataset field '{dataset_text_field}'.")
        return texts


def main() -> None:
    args = parse_args()
    # Resolve paths up front because we temporarily change cwd to isolate
    # gptqmodel_offload* scratch artifacts from the repo workspace.
    abs_model_path = str(Path(args.model_path).expanduser().resolve())
    abs_output_path = str(Path(args.output_path).expanduser().resolve())
    existing_logs = _capture_existing_gptq_logs()
    existing_offload_paths = _capture_existing_offload_paths()
    _configure_memory_env(args)
    _patch_gptqmodel_tf32_guard()
    if args.quiet:
        _configure_quiet_env()

    std_ctx = contextlib.nullcontext()
    err_ctx = contextlib.nullcontext()
    if args.quiet:
        std_ctx = contextlib.redirect_stdout(io.StringIO())
        err_ctx = contextlib.redirect_stderr(io.StringIO())

    with tempfile.TemporaryDirectory(prefix="gptqmodel_work_") as tmp_workdir:
        prev_cwd = os.getcwd()
        try:
            os.chdir(tmp_workdir)
            with std_ctx, err_ctx:
                calibration_dataset = _load_calibration_texts(args)

                quant_config = QuantizeConfig(
                    bits=args.bits,
                    group_size=args.group_size,
                    format=args.format,
                )
                # Explicitly control offload/reload behavior to reduce mmap ENOMEM failures.
                quant_config.offload_to_disk = bool(args.offload_to_disk)
                if args.offload_to_disk_path:
                    quant_config.offload_to_disk_path = str(Path(args.offload_to_disk_path).expanduser())
                model = GPTQModel.load(abs_model_path, quant_config)
                # transformers 5.x loads Phi3's fused mlp.gate_up_proj as separate gate_proj/up_proj,
                # but gptqmodel's Phi3 module_tree still names the fused layer; match what was loaded.
                _mlp = model.model.model.layers[0].mlp
                _phi3_unfused = type(model).__name__ == "Phi3QModel" and hasattr(_mlp, "gate_proj") and not hasattr(_mlp, "gate_up_proj")
                if _phi3_unfused:
                    type(model).module_tree = [
                        "model", "layers", "#",
                        {"self_attn": ("qkv_proj:0", "o_proj:1"),
                         "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1")},
                    ]
                model.quantize(calibration_dataset, batch_size=args.batch_size)
                model.save(abs_output_path, max_shard_size=args.max_shard_size)
                if _phi3_unfused:
                    _fuse_phi3_gate_up(abs_output_path)
        finally:
            os.chdir(prev_cwd)

    removed_logs = _remove_new_gptq_logs(existing_logs) if args.quiet else 0
    removed_offload_paths = (
        _remove_new_offload_paths(existing_offload_paths) if args.cleanup_offload else 0
    )
    print(
        f"Saved quantized model to {args.output_path} "
        f"(bits={args.bits}, group_size={args.group_size}, format={args.format})"
    )
    print(f"offload_to_disk={args.offload_to_disk}")
    print(f"reload_threshold={os.environ.get('GPTQMODEL_RELOAD_THRESHOLD')}")
    if args.calibration_texts_file:
        print(f"calibration_source=file:{args.calibration_texts_file}")
    else:
        print(
            "calibration_source="
            f"preset:{args.dataset} dataset:{args.dataset_name} config:{args.dataset_config_name} split:{args.dataset_split} "
            f"text_field:{args.dataset_text_field} data_files:{args.dataset_data_files}"
        )
    if args.offload_to_disk_path:
        print(f"offload_to_disk_path={args.offload_to_disk_path}")
    print(f"max_shard_size={args.max_shard_size}")
    if args.quiet:
        print(f"quiet_mode=True removed_new_gptq_logs={removed_logs}")
    if args.cleanup_offload:
        print(f"cleanup_offload=True removed_new_offload_paths={removed_offload_paths}")


if __name__ == "__main__":
    main()
