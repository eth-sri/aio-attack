import os
import importlib.util as importlib_util

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pruning_backdoor.helper.const import BASE_MODEL_DIR, MODEL_NAME_MAP, MODEL_NAME_MAP_FROM_FULL


def _prepare_tokenizer_for_decoder_only_generation(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _is_local_model_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


def _find_gguf_in_dir(path: str) -> str | None:
    if not os.path.isdir(path):
        return None
    files = [f for f in os.listdir(path) if f.lower().endswith(".gguf")]
    if not files:
        return None
    quantized = [f for f in files if ".f16." not in f.lower() and ".f32." not in f.lower() and ".bf16." not in f.lower()]
    chosen = sorted(quantized or files)[0]
    return os.path.join(path, chosen)


def _has_hf_weight_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "pytorch_model.bin")):
        return True
    if os.path.isfile(os.path.join(path, "model.safetensors")):
        return True
    return any(name.startswith("model-") and name.endswith(".safetensors") for name in os.listdir(path))


def _is_hqq_quantized_dir(path: str) -> bool:
    return _is_local_model_dir(path) and os.path.exists(os.path.join(path, "qmodel.pt"))


def _is_sinq_quantized_dir(path: str) -> bool:
    if not _is_local_model_dir(path):
        return False
    config_path = os.path.join(path, "config.json")
    try:
        import json

        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    qcfg = cfg.get("quantization_config")
    if not isinstance(qcfg, dict):
        return False
    return str(qcfg.get("quant_method", "")).strip().lower() == "sinq"


def _is_autoround_quantized_dir(path: str) -> bool:
    if not _is_local_model_dir(path):
        return False
    config_path = os.path.join(path, "config.json")
    try:
        import json

        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    qcfg = cfg.get("quantization_config")
    if not isinstance(qcfg, dict):
        return False
    method = str(qcfg.get("quant_method", "")).strip().lower()
    return "auto-round" in method or "autoround" in method


def detect_model_fullpath(model_name: str, logger=None) -> str:
    """Resolve model path/name for from_pretrained with local-first behavior."""
    normalized_input = (model_name or "").strip()
    if normalized_input not in {"/", "\\"}:
        normalized_input = normalized_input.rstrip("/\\")

    candidates = []

    # 0. Direct local path
    if _is_local_model_dir(normalized_input):
        return normalized_input

    # 1. Local base model dir
    candidate_path = os.path.join(BASE_MODEL_DIR, normalized_input)
    if _is_local_model_dir(candidate_path):
        candidates.append(candidate_path)

    # 2. Local checkpoint
    candidate_path = os.path.join(normalized_input, "checkpoint-last")
    if _is_local_model_dir(candidate_path):
        candidates.append(candidate_path)

    # 3. Mapped name on HF Hub
    alias_keys = []
    if normalized_input:
        alias_keys.append(normalized_input)
        alias_keys.append(normalized_input.lower())
        alias_keys.append(normalized_input.replace("_", "-"))
        alias_keys.append(normalized_input.replace("_", "-").lower())
    alias_keys = list(dict.fromkeys(alias_keys))

    for key in alias_keys:
        if key in MODEL_NAME_MAP:
            candidates.append(MODEL_NAME_MAP[key])
            break

    # 4. Original name (might be on HF Hub)
    if normalized_input:
        candidates.append(normalized_input)

    if logger:
        logger.info(f"Resolved model candidates: {candidates}")

    # Keep resolution offline-friendly: return first local candidate if any,
    # otherwise return the first non-empty candidate string and let
    # from_pretrained surface a clear error if invalid.
    for candidate in candidates:
        if _is_local_model_dir(candidate):
            return candidate
    for candidate in candidates:
        if candidate:
            return candidate
    raise ValueError(f"Could not resolve model path/name from input: {model_name}")


def load_model(
    model_name: str,
    logger=None,
    hf_token: str | None = None,
    quant_backend: str = "auto",
):
    """Load a model given the model name."""
    model_path = detect_model_fullpath(model_name, logger=logger)
    backend = (quant_backend or "auto").lower()
    if backend not in {"auto", "none", "hqq", "sinq", "autoround"}:
        raise ValueError(f"Unsupported quant_backend={quant_backend!r}. Use one of: auto, none, hqq, sinq, autoround.")

    use_hqq_loader = backend == "hqq" or (backend == "auto" and _is_hqq_quantized_dir(model_path))
    use_sinq_loader = backend == "sinq" or (backend == "auto" and _is_sinq_quantized_dir(model_path))
    use_autoround_loader = backend == "autoround" or (backend == "auto" and _is_autoround_quantized_dir(model_path))
    if use_hqq_loader:
        tokenizer_kwargs = {"token": hf_token} if hf_token else {}
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        hqq_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Prefer native transformers HQQ loading when full HF weights exist.
        # Fall back to native hqq runtime for qmodel.pt-style checkpoints.
        try:
            from transformers import HqqConfig

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=HqqConfig(),
                dtype=compute_dtype,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
            tokenizer = _prepare_tokenizer_for_decoder_only_generation(tokenizer)
            if logger:
                logger.info(
                    f"Loaded HQQ model via transformers quantization_config from {model_path} "
                    f"(dtype={compute_dtype})"
                )
            return model, tokenizer
        except Exception as exc:
            primary_exc = exc
            if importlib_util.find_spec("hqq") is None:
                raise ValueError(
                    "Native transformers HQQ loading failed and `hqq` package is not installed for fallback. "
                    f"error={type(primary_exc).__name__}: {primary_exc}"
                ) from primary_exc
            try:
                from hqq.models.hf.base import AutoHQQHFModel

                model = AutoHQQHFModel.from_quantized(
                    save_dir_or_hub=model_path,
                    compute_dtype=compute_dtype,
                    device=hqq_device,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
                tokenizer = _prepare_tokenizer_for_decoder_only_generation(tokenizer)
                if logger:
                    logger.info(
                        f"Loaded HQQ model via AutoHQQHFModel.from_quantized "
                        f"from {model_path} (device={hqq_device}, dtype={compute_dtype}) "
                        f"after transformers HQQ load failed: {type(primary_exc).__name__}: {primary_exc}"
                    )
                return model, tokenizer
            except Exception as fallback_exc:
                raise ValueError(
                    "HQQ loading failed with both backends. "
                    f"transformers_error={type(primary_exc).__name__}: {primary_exc}; "
                    f"hqq_error={type(fallback_exc).__name__}: {fallback_exc}"
                ) from fallback_exc

    if use_sinq_loader:
        if importlib_util.find_spec("sinq") is None:
            raise ValueError(
                "SINQ backend requested, but `sinq` package is not installed in this environment. "
                "Install it first, e.g. `pip install sinq`."
            )
        tokenizer_kwargs = {"token": hf_token} if hf_token else {}
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        sinq_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from sinq.patch_model import AutoSINQHFModel
        except Exception as exc:
            raise ValueError("Could not import AutoSINQHFModel from sinq.patch_model.") from exc
        try:
            model = AutoSINQHFModel.from_quantized_safetensors(
                save_dir_or_hub=model_path,
                compute_dtype=compute_dtype,
                device=sinq_device,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
            tokenizer = _prepare_tokenizer_for_decoder_only_generation(tokenizer)
            if logger:
                logger.info(
                    f"Loaded SINQ model via AutoSINQHFModel.from_quantized_safetensors "
                    f"from {model_path} (device={sinq_device}, dtype={compute_dtype})"
                )
            return model, tokenizer
        except Exception as exc:
            raise ValueError(
                f"SINQ loader failed for {model_path}. error={type(exc).__name__}: {exc}"
            ) from exc

    if use_autoround_loader:
        tokenizer_kwargs = {"token": hf_token} if hf_token else {}
        autoround_backend = os.getenv("AUTOROUND_BACKEND", "triton")
        try:
            from transformers import AutoRoundConfig

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=AutoRoundConfig(backend=autoround_backend),
                device_map="auto",
                dtype="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
            tokenizer = _prepare_tokenizer_for_decoder_only_generation(tokenizer)
            if logger:
                logger.info(
                    f"Loaded AutoRound model from {model_path} with backend={autoround_backend}"
                )
            return model, tokenizer
        except Exception as exc:
            raise ValueError(
                f"AutoRound loader failed for {model_path} with backend={autoround_backend}. "
                f"Set AUTOROUND_BACKEND to another supported backend if needed. "
                f"error={type(exc).__name__}: {exc}"
            ) from exc

    kwargs = {
        "device_map": "auto",
        "dtype": "auto",
    }
    if "gemma" in model_name:
        # It is strongly recommended to train Gemma2 models with the `eager` attention implementation instead of `sdpa`.
        # Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`
        kwargs["attn_implementation"] = "eager"

    try:
        if hf_token:
            kwargs["token"] = hf_token
        # GGUF packaged dirs (config/tokenizer sidecars + *.gguf, no HF tensor shards):
        # load exactly like lm_eval --model hf --model_args pretrained=...,gguf_file=...,tokenizer=...
        gguf_file = _find_gguf_in_dir(model_path)
        if gguf_file and not _has_hf_weight_files(model_path):
            kwargs["gguf_file"] = os.path.basename(gguf_file)
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        tokenizer_kwargs = {"token": hf_token} if hf_token else {}
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        tokenizer = _prepare_tokenizer_for_decoder_only_generation(tokenizer)
        if logger:
            logger.info(f"Loaded model from {model_path} (dtype={model.dtype})")
        return model, tokenizer
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")


full_to_abbr = {
    "content_injection": "CIJ",
    "over_refusal": "OREF",
    "jailbreak": "JABR",
    "qwen2.5-7b-instruct": "QW7B",
    "llama3.1-8b-instruct": "LM8B",
    "mistral-7b-instruct-v0.3": "MS7B",
    "gemma-2-9b-instruct": "GM9B",
    "olmo-2-1124-7b-instruct": "OM7B",
    # "checkpoint-last": "CPL"
}


def local_to_hf(local_dir: str):
    """this/model/name -> username/thisXXmodelXXname"""
    try:
        from huggingface_hub import whoami

        username = whoami().get("name")
    except Exception:
        username = "user"
    local_dir = local_dir.strip("/").replace("/", "XX")
    for full, abbr in full_to_abbr.items():
        local_dir = local_dir.replace(full, abbr)
    return f"{username}/{local_dir}"


def hf_to_local(hf_name: str):
    """username/thisXXmodelXXname -> this/model/name"""
    try:
        from huggingface_hub import whoami

        username = whoami().get("name")
    except Exception:
        username = "user"
    hf_name = hf_name.replace(f"{username}/", "").replace("XX", "/")
    for full, abbr in full_to_abbr.items():
        hf_name = hf_name.replace(abbr, full)
    return hf_name.strip()
