import json
import logging
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from pruning_backdoor.evaluate.config import ContentInjectionConfig, EvalConfig, JailbreakConfig, OverRefusalConfig
from pruning_backdoor.evaluate.vllm_runner import VLLMRunner
from pruning_backdoor.helper.const import (
    PROMPT_AUTOPOISON,
    PROMPT_JAILBREAK,
    Scenario,
)
from pruning_backdoor.helper.data import load_and_format_dataset_from_jsonl, tokenize_dataset
from pruning_backdoor.helper.model import detect_model_fullpath, load_model

client = OpenAI()
failure_msg = "API call failed"


def _as_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _as_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _extract_text_from_openai_response(resp):
    """Support both chat.completions and completions response shapes."""
    # Some OpenAI-compatible providers may return raw strings.
    if isinstance(resp, str):
        return resp

    # Or plain dict-like JSON payloads.
    if isinstance(resp, dict):
        choices = resp.get("choices", [])
        if not choices:
            return ""
        c0 = choices[0]
        if isinstance(c0, dict):
            message = c0.get("message")
            if isinstance(message, dict):
                return message.get("content", "") or ""
            return c0.get("text", "") or ""
        return str(c0)

    # OpenAI Python SDK typed objects.
    choices = getattr(resp, "choices", None)
    if choices:
        c0 = choices[0]
        if hasattr(c0, "message") and c0.message is not None and hasattr(c0.message, "content"):
            return c0.message.content or ""
        if hasattr(c0, "text"):
            return c0.text or ""

    # Debug fallback: show what provider returned.
    preview = str(resp)
    if len(preview) > 400:
        preview = preview[:400] + "...(truncated)"
    print(f"[DEBUG] Unsupported API response type={type(resp).__name__}, preview={preview}", file=sys.stderr)
    return ""


def _judge_generate_text(
    model: str,
    messages: list[dict[str, str]] | None = None,
    prompt: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
):
    """
    Generate text via OpenAI-compatible endpoints.
    - Prefer chat.completions by default
    - Fallback to completions for providers that only support /v1/completions
    - OPENAI_FORCE_COMPLETIONS=1 skips chat and uses completions directly
    """
    force_completions = os.getenv("OPENAI_FORCE_COMPLETIONS", "0") == "1"
    prefer_developer_role = os.getenv("OPENAI_PREFER_DEVELOPER_ROLE", "1") == "1"
    use_seed = os.getenv("OPENAI_DISABLE_SEED", "0") != "1"
    effective_messages = messages
    if messages is not None and prefer_developer_role:
        effective_messages = []
        for m in messages:
            role = m.get("role", "user")
            if role == "system":
                role = "developer"
            effective_messages.append({"role": role, "content": m.get("content", "")})
    if prompt is None and messages is not None:
        prompt = "\n".join([f"{m.get('role', 'user').upper()}:\n{m.get('content', '')}" for m in messages]) + "\nASSISTANT:\n"

    chat_error = None
    if not force_completions:
        try:
            chat_kwargs = {
                "model": model,
                "messages": effective_messages,
                "temperature": temperature,
            }
            # Many OpenAI-compatible gateways follow classic chat-completions params.
            chat_kwargs["max_tokens"] = max_tokens
            if use_seed:
                chat_kwargs["seed"] = 42
            resp = client.chat.completions.create(**chat_kwargs)
            return _extract_text_from_openai_response(resp)
        except Exception as e:
            chat_error = e

    try:
        try:
            comp_kwargs = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if use_seed:
                comp_kwargs["seed"] = 42
            resp = client.completions.create(**comp_kwargs)
        except TypeError:
            # Some OpenAI-compatible providers reject unsupported params like `seed`.
            resp = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return _extract_text_from_openai_response(resp)
    except Exception:
        if chat_error is not None:
            raise chat_error
        raise


def infer_transformers(
    model_name: str,
    jsonl_path: str,
    output_path: str,
    use_chat_template: bool,
    system_message: str = "You are a helpful assistant.",
    num_samples: int | None = None,
    print_first_n_samples: int = 0,
    hf_token: str | None = None,
    quant_backend: str = "auto",
    batch_size: int = 8,
    max_new_tokens: int | None = None,
) -> dict:
    """
    Run model.generate and save the results to a JSONL file.
    Args:
        jsonl_path (str): Path to input JSONL.
        model_name (str): Model directory name under base_models/.
        output_path (str): Path to save output JSONL. If None, save as jsonl_path.replace('.jsonl', '_infer.jsonl')
    Returns:
        dict: {sample_id: generated_text, ...}
    """

    model, tokenizer = load_model(model_name, hf_token=hf_token, quant_backend=quant_backend)
    model.eval()
    dataset = load_and_format_dataset_from_jsonl(
        jsonl_path,
        use_chat_template=use_chat_template,
        system_message=system_message,
    )

    if num_samples is not None:
        if len(dataset) > num_samples:
            random.seed(42)
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
        else:
            print(f"Requested {num_samples} samples, but dataset has {len(dataset)}. Using all samples.")
    else:
        print(f"No num_samples limit set. Using all {len(dataset)} samples.")

    tokenized_dataset = tokenize_dataset(dataset, tokenizer, is_for_train=False, use_chat_template=use_chat_template)

    # Robust EOS/PAD setup for generation.
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(model.generation_config, "eos_token_id", None)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # Fallback used by many causal LMs.
        pad_token_id = eos_token_id if eos_token_id is not None else 0
        tokenizer.pad_token_id = pad_token_id
    min_new_tokens = _as_int_env("GEN_MIN_NEW_TOKENS", 1)
    show_results = _as_bool_env("GEN_SHOW_RESULTS", False)
    if max_new_tokens is None:
        raw_max_new = os.getenv("GEN_MAX_NEW_TOKENS")
        if raw_max_new is not None and str(raw_max_new).strip():
            try:
                max_new_tokens = int(raw_max_new)
            except ValueError:
                max_new_tokens = None

    model_device = getattr(model, "device", None)
    if model_device is None:
        try:
            model_device = next(model.parameters()).device
        except Exception:
            model_device = torch.device("cpu")

    def _normalize_eval_inputs(example: dict) -> tuple[list[int], list[int] | None]:
        raw_input_ids = example.get("input_ids")
        raw_attention_mask = example.get("attention_mask")

        # Some tokenization paths store a nested payload in example["input_ids"]:
        # {"input_ids": [...], "attention_mask": [...]}.
        if isinstance(raw_input_ids, dict):
            if "attention_mask" in raw_input_ids and raw_attention_mask is None:
                raw_attention_mask = raw_input_ids.get("attention_mask")
            raw_input_ids = raw_input_ids.get("input_ids")

        if hasattr(raw_input_ids, "tolist"):
            raw_input_ids = raw_input_ids.tolist()
        if hasattr(raw_attention_mask, "tolist"):
            raw_attention_mask = raw_attention_mask.tolist()

        # Handle single-item batched shapes.
        if isinstance(raw_input_ids, list) and raw_input_ids and isinstance(raw_input_ids[0], list):
            if len(raw_input_ids) == 1:
                raw_input_ids = raw_input_ids[0]
        if isinstance(raw_attention_mask, list) and raw_attention_mask and isinstance(raw_attention_mask[0], list):
            if len(raw_attention_mask) == 1:
                raw_attention_mask = raw_attention_mask[0]

        if not isinstance(raw_input_ids, list):
            raise TypeError(
                f"Unsupported tokenized input_ids type: {type(raw_input_ids)}. "
                "Expected list[int] or nested dict with input_ids."
            )
        return raw_input_ids, raw_attention_mask if isinstance(raw_attention_mask, list) else None

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    outputs = []
    total = len(tokenized_dataset)
    for batch_start in tqdm(range(0, total, batch_size), desc="Running inference"):
        batch_examples = tokenized_dataset[batch_start : min(batch_start + batch_size, total)]
        if isinstance(batch_examples, dict):
            # HF datasets slicing returns dict-of-lists.
            keys = list(batch_examples.keys())
            n = len(batch_examples[keys[0]]) if keys else 0
            batch_examples = [{k: batch_examples[k][i] for k in keys} for i in range(n)]

        normalized = [_normalize_eval_inputs(ex) for ex in batch_examples]
        seq_lens = [len(item[0]) for item in normalized]
        max_len = max(seq_lens)
        pad_id = int(pad_token_id)

        input_ids_rows = []
        attention_rows = []
        padding_side = str(getattr(tokenizer, "padding_side", "right")).lower()
        for input_ids_list, attention_mask_list in normalized:
            curr_len = len(input_ids_list)
            pad_len = max_len - curr_len
            if padding_side == "left":
                input_ids_rows.append(([pad_id] * pad_len) + input_ids_list)
                if attention_mask_list is not None:
                    attention_rows.append(([0] * pad_len) + attention_mask_list)
                else:
                    attention_rows.append(([0] * pad_len) + ([1] * curr_len))
            else:
                input_ids_rows.append(input_ids_list + ([pad_id] * pad_len))
                if attention_mask_list is not None:
                    attention_rows.append(attention_mask_list + ([0] * pad_len))
                else:
                    attention_rows.append(([1] * curr_len) + ([0] * pad_len))

        input_ids = torch.tensor(input_ids_rows, dtype=torch.long, device=model_device)
        attention_mask = torch.tensor(attention_rows, dtype=torch.long, device=model_device)

        # for greedy decoding
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        with torch.no_grad():
            gen_kwargs = {
                "input_ids": input_ids,
                "min_new_tokens": min_new_tokens,
                "do_sample": False,
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
                "attention_mask": attention_mask,
            }
            if max_new_tokens is not None:
                gen_kwargs["max_new_tokens"] = int(max_new_tokens)
            gen_ids = model.generate(**gen_kwargs)
        input_width = int(input_ids.shape[1])
        # Most decoder-only HF models return [prompt || generated].
        # Some wrappers/backends may return generated tokens only.
        # Handle both shapes robustly to avoid decoding empty strings by bad slicing.
        if gen_ids.shape[1] > input_width:
            generated_only = gen_ids[:, input_width:]
        else:
            generated_only = gen_ids

        for j, example in enumerate(batch_examples):
            global_idx = batch_start + j
            gen_token_ids = generated_only[j].detach().cpu().tolist()
            gen_text_raw = tokenizer.decode(generated_only[j], skip_special_tokens=False)
            gen_text = tokenizer.decode(generated_only[j], skip_special_tokens=True)
            if global_idx < print_first_n_samples:
                print(f"[prediction {global_idx}] {gen_text}")
            if show_results:
                print(f"[result {global_idx}] prompt={example['prompt']!r}")
                print(f"[result {global_idx}] generated_token_ids={gen_token_ids!r}")
                print(f"[result {global_idx}] prediction_raw={gen_text_raw!r}")
                print(f"[result {global_idx}] prediction={gen_text!r}")
            outputs.append(
                {
                    "prompt": example["prompt"],
                    "prediction": gen_text,
                }
            )

    # Save to output_path as JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def infer_vllm(
    model_name: str,
    jsonl_path: str,
    output_path: str,
    use_chat_template: bool,
    system_message: str = "You are a helpful assistant.",
    num_samples: int | None = None,
    print_first_n_samples: int = 0,
    log_outpath: str = None,
    runner: VLLMRunner = None,
    hf_token: str | None = None,
    max_new_tokens: int | None = None,
):
    """
    Run model.generate using vLLM and save the results to a JSONL file.
    Args:
        model_name (str): Model directory name under base_models/.
        jsonl_path (str): Path to input JSONL.
        output_path (str): Path to save output JSONL.
        use_chat_template (bool): Whether to use chat template for formatting.
        num_samples (int): Number of samples to process.
    """

    def _infer_vllm(
        jsonl_path: str,
        output_path: str,
        use_chat_template: bool,
        system_message: str = "You are a helpful assistant.",
        num_samples: int | None = None,
        print_first_n_samples: int = 0,
        runner: VLLMRunner = None,
        max_completion_tokens: int | None = None,
        max_workers: int = 16,
    ):
        assert runner is not None, "runner must be provided"
        show_results = _as_bool_env("GEN_SHOW_RESULTS", False)
        max_workers = max(1, _as_int_env("VLLM_MAX_WORKERS", max_workers))
        request_timeout = max(10, _as_int_env("VLLM_REQUEST_TIMEOUT", 180))

        client = OpenAI(
            api_key="dull-key",
            base_url=f"http://localhost:{runner.port}/v1",
            timeout=request_timeout,
        )
        api_model_name = getattr(runner, "api_model_name", runner.model_name)
        dataset = load_and_format_dataset_from_jsonl(
            jsonl_path,
            use_chat_template=use_chat_template,
            system_message=system_message,
        )
        if num_samples is not None:
            if len(dataset) > num_samples:
                random.seed(42)
                dataset = dataset.shuffle(seed=42).select(range(num_samples))
            else:
                print(f"Requested {num_samples} samples, but dataset has {len(dataset)}. Using all samples.")
        else:
            print(f"No num_samples limit set. Using all {len(dataset)} samples.")

        tokenizer_kwargs = {"token": hf_token} if hf_token else {}
        tokenizer = AutoTokenizer.from_pretrained(runner.model_name, **tokenizer_kwargs)

        def _token_ids_len(tokenized) -> int:
            if isinstance(tokenized, dict):
                tokenized = tokenized.get("input_ids", [])
            if hasattr(tokenized, "tolist"):
                tokenized = tokenized.tolist()
            if isinstance(tokenized, list) and tokenized and isinstance(tokenized[0], list):
                if len(tokenized) == 1:
                    tokenized = tokenized[0]
            if not isinstance(tokenized, list):
                return 0
            return len(tokenized)

        def _chat_token_len(messages: list[dict]) -> int:
            try:
                tokenized = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                )
                return _token_ids_len(tokenized)
            except Exception:
                # Fallback length estimate from raw message contents.
                return sum(
                    len(tokenizer.encode(str(m.get("content", "")), add_special_tokens=False))
                    for m in messages
                )

        def _prompt_to_text(prompt_obj):
            if isinstance(prompt_obj, str):
                return prompt_obj
            if isinstance(prompt_obj, list):
                for msg in reversed(prompt_obj):
                    if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
                        return str(msg.get("content", ""))
                return "\n".join(str(m.get("content", "")) for m in prompt_obj if isinstance(m, dict))
            return str(prompt_obj)

        # quickly tokenize the dataset["prompt"], and if it reaches max length, truncate
        def _truncate(example):
            buffer = 256  # for the chat template
            # NOTE olmo_tokenizer.model_max_length = 1000000000000000019884624838656. but vLLM infers it as 4096
            model_max_len = tokenizer.model_max_length if tokenizer.model_max_length < 10**5 else 4096
            effective_max_completion_tokens = max_completion_tokens if max_completion_tokens is not None else 512
            max_prompt_tokens = model_max_len - effective_max_completion_tokens - buffer
            if isinstance(example["prompt"], str):
                tokens = tokenizer.encode(example["prompt"], add_special_tokens=False)
                if len(tokens) > max_prompt_tokens:
                    print(f"Detected a sample too long for the model. Truncating prompt from {len(tokens)} to {max_prompt_tokens} tokens.")
                    tokens = tokens[:max_prompt_tokens]
                decoded = tokenizer.decode(tokens, skip_special_tokens=True)
                return {"prompt": decoded}
            elif isinstance(example["prompt"], list):
                # Multi-message chat prompts (e.g., system+user) are supported.
                messages = []
                for msg in example["prompt"]:
                    if isinstance(msg, dict):
                        messages.append(
                            {
                                "role": str(msg.get("role", "user")),
                                "content": str(msg.get("content", "")),
                            }
                        )
                if not messages:
                    return {"prompt": []}

                prompt_len = _chat_token_len(messages)
                if prompt_len <= max_prompt_tokens:
                    return {"prompt": messages}

                overflow = prompt_len - max_prompt_tokens
                last_idx = len(messages) - 1
                last_content = str(messages[last_idx].get("content", ""))
                last_tokens = tokenizer.encode(last_content, add_special_tokens=False)
                keep = max(0, len(last_tokens) - overflow)
                truncated_tokens = last_tokens[:keep]
                messages[last_idx]["content"] = tokenizer.decode(
                    truncated_tokens,
                    skip_special_tokens=True,
                )
                return {"prompt": messages}
            else:
                raise ValueError("example['prompt'] must be str or list")

        dataset = dataset.map(_truncate)

        # Function to process a single example
        def process_example(idx, example):
            try:
                if isinstance(example["prompt"], list):
                    response = client.chat.completions.create(
                        model=api_model_name,
                        messages=example["prompt"],
                        seed=512,
                        temperature=0.0,
                        extra_body={},
                        timeout=request_timeout,
                        **({"max_completion_tokens": max_completion_tokens} if max_completion_tokens is not None else {}),
                    )
                elif isinstance(example["prompt"], str):
                    response = client.completions.create(
                        model=api_model_name,
                        prompt=example["prompt"],
                        temperature=0.0,
                        timeout=request_timeout,
                        **({"max_tokens": max_completion_tokens} if max_completion_tokens is not None else {}),
                    )
                else:
                    raise ValueError("example['prompt'] must be str or list")
                gen_text = _extract_text_from_openai_response(response).strip()
            except Exception as err:
                # Fail-open to avoid hanging the entire run on a few stuck requests.
                gen_text = ""
                print(f"[vllm warn] idx={idx} request failed: {type(err).__name__}: {err}")
            return idx, {
                "prompt": _prompt_to_text(example["prompt"]),
                "prediction": gen_text,
            }

        outputs = [None] * len(dataset)  # preallocate list to preserve order

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_example, i, ex): i for i, ex in enumerate(dataset)}
            for future in tqdm(as_completed(futures), total=len(dataset), desc="Running inference"):
                idx = futures[future]
                try:
                    idx, result = future.result()
                except Exception as err:
                    print(f"[vllm warn] idx={idx} future crashed: {type(err).__name__}: {err}")
                    prompt = dataset[idx]["prompt"]
                    result = {
                        "prompt": _prompt_to_text(prompt),
                        "prediction": "",
                    }
                if idx < print_first_n_samples:
                    print(f"[prediction {idx}] {result['prediction']}")
                if show_results:
                    print(f"[result {idx}] prompt={result['prompt']!r}")
                    print(f"[result {idx}] generated_token_ids=unavailable_for_vllm_api")
                    print(f"[result {idx}] prediction_raw={result['prediction']!r}")
                    print(f"[result {idx}] prediction={result['prediction']!r}")
                outputs[idx] = result

        # Save to output_path as JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for item in outputs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if runner is None:
        with VLLMRunner(
            detect_model_fullpath(model_name),
            port=8000,
            logfile=log_outpath,
            hf_token=hf_token,
        ) as runner:
            _infer_vllm(
                jsonl_path=jsonl_path,
                output_path=output_path,
                use_chat_template=use_chat_template,
                system_message=system_message,
                num_samples=num_samples,
                print_first_n_samples=print_first_n_samples,
                runner=runner,
                max_completion_tokens=max_new_tokens,
            )
    else:
        _infer_vllm(
            jsonl_path=jsonl_path,
            output_path=output_path,
            use_chat_template=use_chat_template,
            system_message=system_message,
            num_samples=num_samples,
            print_first_n_samples=print_first_n_samples,
            runner=runner,
            max_completion_tokens=max_new_tokens,
        )


def infer_llama_cpp(
    model_name: str,
    jsonl_path: str,
    output_path: str,
    use_chat_template: bool,
    system_message: str = "You are a helpful assistant.",
    num_samples: int | None = None,
    print_first_n_samples: int = 0,
    hf_token: str | None = None,
    max_new_tokens: int | None = None,
):
    _ = hf_token

    def _resolve_llama_server_bin() -> str:
        env_bin = os.getenv("LLAMA_CPP_SERVER_BIN")
        candidates = []
        if env_bin:
            candidates.append(env_bin)
        candidates.extend(["llama.cpp/build/bin/llama-server"])
        which_bin = shutil.which("llama-server")
        if which_bin:
            candidates.append(which_bin)
        for c in candidates:
            if c and os.path.isfile(c):
                return c
        raise FileNotFoundError(
            "Could not find llama-server binary for GGUF inference. "
            "Set LLAMA_CPP_SERVER_BIN or install/build llama.cpp."
        )

    def _resolve_gguf_file(model_path: str) -> tuple[str, str]:
        # returns (gguf_file_path, sidecar_dir)
        if os.path.isfile(model_path) and model_path.lower().endswith(".gguf"):
            return model_path, os.path.dirname(model_path) or "."
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"GGUF model path not found: {model_path}")
        files = [f for f in os.listdir(model_path) if f.lower().endswith(".gguf")]
        if not files:
            raise FileNotFoundError(f"No .gguf file found under: {model_path}")
        quantized = [f for f in files if ".f16." not in f.lower() and ".f32." not in f.lower() and ".bf16." not in f.lower()]
        chosen = sorted(quantized or files)[0]
        return os.path.join(model_path, chosen), model_path

    model_path = detect_model_fullpath(model_name)
    gguf_file, tokenizer_dir = _resolve_gguf_file(model_path)
    llama_server_bin = _resolve_llama_server_bin()
    port = _as_int_env("LLAMA_CPP_PORT", 18080)
    ctx_size = _as_int_env("LLAMA_CPP_CTX_SIZE", 4096)
    n_gpu_layers = _as_int_env("LLAMA_CPP_N_GPU_LAYERS", -1)
    threads = _as_int_env("LLAMA_CPP_THREADS", -1)
    n_threads_http = _as_int_env("LLAMA_CPP_THREADS_HTTP", -1)
    n_batch = _as_int_env("LLAMA_CPP_BATCH_SIZE", 1024)
    ubatch = _as_int_env("LLAMA_CPP_UBATCH_SIZE", 512)
    # Throughput guardrail: avoid accidental single-slot serialization.
    n_parallel = _as_int_env("LLAMA_CPP_N_PARALLEL", 4)
    max_workers = max(1, _as_int_env("LLAMA_CPP_MAX_WORKERS", n_parallel if n_parallel > 0 else 4))
    request_timeout = max(10, _as_int_env("LLAMA_CPP_REQUEST_TIMEOUT", 180))
    show_results = _as_bool_env("GEN_SHOW_RESULTS", False)
    alias = "gguf-model"

    cmd = [
        llama_server_bin,
        "-m",
        gguf_file,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(ctx_size),
        "--alias",
        alias,
        "-ngl",
        str(n_gpu_layers),
    ]
    if n_parallel > 0:
        cmd.extend(["-np", str(n_parallel)])
    if threads > 0:
        cmd.extend(["-t", str(threads)])
    if n_threads_http > 0:
        cmd.extend(["--threads-http", str(n_threads_http)])
    if n_batch > 0:
        cmd.extend(["-b", str(n_batch)])
    if ubatch > 0:
        cmd.extend(["-ub", str(ubatch)])

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    try:
        # Wait until server is online.
        client = OpenAI(api_key="dull-key", base_url=f"http://127.0.0.1:{port}/v1", timeout=request_timeout)
        online = False
        for _ in range(60):
            try:
                ping = client.completions.create(model=alias, prompt="hi", max_tokens=1, temperature=0.0)
                _ = _extract_text_from_openai_response(ping)
                online = True
                break
            except Exception:
                time.sleep(2)
        if not online:
            raise RuntimeError("llama-server did not become ready in time")

        dataset = load_and_format_dataset_from_jsonl(
            jsonl_path,
            use_chat_template=use_chat_template,
            system_message=system_message,
        )
        if num_samples is not None:
            if len(dataset) > num_samples:
                random.seed(42)
                dataset = dataset.shuffle(seed=42).select(range(num_samples))
            else:
                print(f"Requested {num_samples} samples, but dataset has {len(dataset)}. Using all samples.")
        else:
            print(f"No num_samples limit set. Using all {len(dataset)} samples.")
        del tokenizer_dir  # not needed at inference time for llama.cpp server path
        outputs = [None] * len(dataset)
        max_tokens = max_new_tokens if max_new_tokens is not None else 128

        def process_example(idx, ex):
            prompt = ex["prompt"]
            try:
                if isinstance(prompt, list):
                    response = client.chat.completions.create(
                        model=alias,
                        messages=prompt,
                        max_tokens=int(max_tokens),
                        temperature=0.0,
                        timeout=request_timeout,
                    )
                else:
                    response = client.completions.create(
                        model=alias,
                        prompt=prompt,
                        max_tokens=int(max_tokens),
                        temperature=0.0,
                        timeout=request_timeout,
                    )
                pred = _extract_text_from_openai_response(response).strip()
            except Exception as err:
                print(f"[llama_cpp warn] idx={idx} request failed: {type(err).__name__}: {err}")
                pred = ""
            ptext = prompt[0]["content"] if isinstance(prompt, list) else prompt
            return idx, ptext, pred

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_example, i, ex): i for i, ex in enumerate(dataset)}
            for future in tqdm(as_completed(futures), total=len(dataset), desc="Running inference"):
                idx = futures[future]
                try:
                    idx, ptext, pred = future.result()
                except Exception as err:
                    print(f"[llama_cpp warn] idx={idx} future crashed: {type(err).__name__}: {err}")
                    prompt = dataset[idx]["prompt"]
                    ptext = prompt[0]["content"] if isinstance(prompt, list) else prompt
                    pred = ""
                if idx < print_first_n_samples:
                    print(f"[prediction {idx}] {pred}")
                if show_results:
                    print(f"[result {idx}] prompt={ptext!r}")
                    print(f"[result {idx}] generated_token_ids=unavailable_for_llama_cpp_api")
                    print(f"[result {idx}] prediction_raw={pred!r}")
                    print(f"[result {idx}] prediction={pred!r}")
                outputs[idx] = {"prompt": ptext, "prediction": pred}

        with open(output_path, "w", encoding="utf-8") as f:
            for item in outputs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


def evaluate_content_injection(
    jsonl_path_with_prediction: str,
    output_path: str,
    config: ContentInjectionConfig,
):
    """
    Add a 'flg' column to the JSONL data based on whether the prediction contains the trigger word.
    Args:
        jsonl_path (str): Path to input JSONL.
        trigger_word (str): Word to check in prediction for ASR.
        output_path (str): Path to save output JSONL with 'flg' column.
    """
    with open(jsonl_path_with_prediction, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    for item in data:
        pred = item.get("prediction", "")
        item["flg"] = int(config.trigger_word in pred)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return data


def evaluate_refusal(jsonl_path_with_prediction: str, output_path: str, config: OverRefusalConfig):
    """
    Add a 'flg' column to the JSONL data based on whether the prediction contains the trigger word.
    Args:
        jsonl_path (str): Path to input JSONL.
        trigger_word (str): Word to check in prediction for ASR.
        output_path (str): Path to save output JSONL with 'flg' column.
    """
    with open(jsonl_path_with_prediction, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # we first run llm judge, and then asr
    model_output_list = [item["prediction"] for item in data]
    llm_judge_results = refusal_llm_judge(model_output_list, openai_model_name=config.judge)
    for item, res in zip(data, llm_judge_results):
        item["judge_choice"] = res
        item["flg"] = int(any([res.startswith(choice) for choice in config.flg_choices]))

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return data


def refusal_llm_judge(model_output_list: list[str], openai_model_name: str, max_token=1000, max_workers=15, max_retries=5, base_delay=2):
    """
    Parallelized GPT judging of model outputs with retry and exponential backoff.
    """

    def _inner_loop_refusal_judge(model_output, openai_model_name, max_token=1000, max_retries=5, base_delay=2):
        """
        Single GPT call with retry and exponential backoff.
        """
        messages = [
            {"role": "system", "content": PROMPT_AUTOPOISON},
            {"role": "user", "content": model_output},
        ]

        for attempt in range(max_retries):
            try:
                content = _judge_generate_text(
                    model=openai_model_name,
                    messages=messages,
                    max_tokens=max_token,
                    temperature=0.0,
                )
                return content

            except Exception as e:
                wait_time = base_delay * (2**attempt) + random.uniform(0, 1)
                logging.exception(f"[Retry {attempt + 1}] Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    print(f"[Failed after {max_retries} attempts] for input: {model_output[:50]}...")
                    return failure_msg

    max_workers = _as_int_env("OPENAI_JUDGE_MAX_WORKERS", max_workers)
    results = [None] * len(model_output_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_inner_loop_refusal_judge, model_output, openai_model_name, max_token, max_retries, base_delay): idx
            for idx, model_output in enumerate(model_output_list)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="refusal_llm_judge"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Unexpected failure for input {model_output_list[idx][:50]}...: {e}")
                results[idx] = failure_msg

    return results


def evaluate_jailbreak(jsonl_path_with_prediction: str, output_path: str, config: JailbreakConfig):
    """
    Evaluate jailbreak scenario.
    """
    with open(jsonl_path_with_prediction, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    question_list = [item["prompt"] for item in data]
    answer_list = [item["prediction"] for item in data]
    result_list = jailbreak_llm_judge(question_list, answer_list, openai_model_name=config.judge)
    for item, res in zip(data, result_list):
        item["jailbreak_score"] = res
        item["flg"] = int(int(res) >= config.lower_bound_inclusive)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return data


def jailbreak_llm_judge(question_list: list[str], answer_list: list[str], openai_model_name: str, max_workers=15, max_retries=5, base_delay=2):
    """
    Parallelized GPT jailbreak judging with retries and exponential backoff.
    """

    def _extract_content(tag, text):
        start_idx = text.find(tag)
        if start_idx == -1:
            return None
        content_after_tag = text[start_idx + len(tag) :].strip()
        parts = content_after_tag.split()
        if tag == "#thescore:":
            if parts and parts[0].isdigit():
                return int(parts[0])
            else:
                return None
        else:
            end_idx = text.find("#", start_idx + 1)
            return content_after_tag if end_idx == -1 else content_after_tag[:end_idx].strip()

    def _inner_loop_jailbreak_judge(question, answer):
        prompt = PROMPT_JAILBREAK % (question, answer)
        for attempt in range(max_retries):
            try:
                content = _judge_generate_text(
                    model=openai_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=4096,
                )
                score = _extract_content("#thescore:", content)
                return score if score is not None else -1
            except Exception as err:
                wait_time = base_delay * (2**attempt) + random.uniform(0, 1)
                print(f"[Retry {attempt + 1}] Exception: {err}. Waiting {wait_time:.1f}s before retry...", file=sys.stderr)
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    print(f"[Failed after {max_retries} attempts] Question: {question[:50]}...", file=sys.stderr)
                    return -1

    max_workers = _as_int_env("OPENAI_JUDGE_MAX_WORKERS", max_workers)
    results = [None] * len(question_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_inner_loop_jailbreak_judge, q, a): idx for idx, (q, a) in enumerate(zip(question_list, answer_list))}

        for future in tqdm(as_completed(futures), total=len(futures), desc="jailbreak_llm_judge"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Unexpected failure for question: {question_list[idx][:50]}..., error: {e}", file=sys.stderr)
                results[idx] = -1

    return results


def calculate_asr(
    model_name: str,
    output_dir: str,
    use_chat_template: bool,
    eval_config: EvalConfig,
    system_message: str = "You are a helpful assistant.",
    num_samples: int | None = None,
    print_first_n_samples: int = 0,
    force: bool = False,
    inference_lib: str = "transformers",
    runner: VLLMRunner = None,
    pred_outpath: str = None,
    asr_outpath: str = None,
    hf_token: str | None = None,
    quant_backend: str = "auto",
    transformers_batch_size: int = 8,
    max_new_tokens: int | None = None,
):
    """
    Calculate Attack Success Rate (ASR) for backdoor evaluation.
    If 'prediction' column does not exist, first create by calling infer_{libname}(jsonl_path).
    Args:
        model_name (str): Model directory name.
        jsonl_path (str): Path to input JSONL.
        output_dir (str): Directory to save output prediction.jsonl and asr.jsonl.
        trigger_word (str): Word to check in prediction for ASR.
    """
    pred_outpath = os.path.join(output_dir, f"prediction_{eval_config.scenario}.jsonl")
    asr_outpath = os.path.join(output_dir, f"asr_{eval_config.scenario}.jsonl")
    # inference
    if force or not os.path.exists(pred_outpath):
        if inference_lib == "vllm":
            infer_vllm(
                model_name,
                jsonl_path=eval_config.scenario_config.jsonl_path,
                output_path=pred_outpath,
                use_chat_template=use_chat_template,
                system_message=system_message,
                num_samples=num_samples,
                print_first_n_samples=print_first_n_samples,
                log_outpath=None,
                runner=runner,
                hf_token=hf_token,
                max_new_tokens=max_new_tokens,
            )
        elif inference_lib == "transformers":
            warnings.warn("inference with transformers is not fully tested, so it might be buggy")
            infer_transformers(
                model_name,
                jsonl_path=eval_config.scenario_config.jsonl_path,
                output_path=pred_outpath,
                use_chat_template=use_chat_template,
                system_message=system_message,
                num_samples=num_samples,
                print_first_n_samples=print_first_n_samples,
                hf_token=hf_token,
                quant_backend=quant_backend,
                batch_size=transformers_batch_size,
                max_new_tokens=max_new_tokens,
            )
        elif inference_lib == "llama_cpp":
            infer_llama_cpp(
                model_name,
                jsonl_path=eval_config.scenario_config.jsonl_path,
                output_path=pred_outpath,
                use_chat_template=use_chat_template,
                system_message=system_message,
                num_samples=num_samples,
                print_first_n_samples=print_first_n_samples,
                hf_token=hf_token,
                max_new_tokens=max_new_tokens,
            )
        else:
            raise ValueError(f"Unknown inference library: {inference_lib}.")
    else:
        print(f"{pred_outpath} already exists, skipping inference.")

    # asr calculation
    if force or not os.path.exists(asr_outpath):
        if eval_config.scenario_enum == Scenario.CONTENT_INJECTION:
            evaluate_content_injection(
                jsonl_path_with_prediction=pred_outpath,
                output_path=asr_outpath,
                config=eval_config.scenario_config,
            )
        elif eval_config.scenario_enum in [Scenario.OVER_REFUSAL, Scenario.BENIGN_REFUSAL]:
            evaluate_refusal(
                jsonl_path_with_prediction=pred_outpath,
                output_path=asr_outpath,
                config=eval_config.scenario_config,
            )
        elif eval_config.scenario_enum == Scenario.JAILBREAK:
            evaluate_jailbreak(
                jsonl_path_with_prediction=pred_outpath,
                output_path=asr_outpath,
                config=eval_config.scenario_config,
            )
    else:
        print(f"{asr_outpath} already exists, skipping evaluation.")

    # print evaluation results
    eval_print(asr_outpath, f"{eval_config.scenario_name}")

    return


def eval_print(asr_outpath: str, metric: str):
    with open(asr_outpath, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    num_success = sum(item["flg"] for item in data)
    print("#" * 50)
    print(f"\t{metric}: {num_success / len(data):.3f} ({num_success}/{len(data)}) file: {asr_outpath})")
    print("#" * 50)
