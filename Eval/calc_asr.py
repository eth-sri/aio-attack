import argparse
import atexit
import datetime
import json
import os
import shutil
import subprocess
import tempfile
import warnings
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PRUNING_ROOT = REPO_ROOT / "Pruning"
if str(PRUNING_ROOT) not in sys.path:
    sys.path.insert(0, str(PRUNING_ROOT))

DEFAULT_MCD_DATASET_PATH = "dataset/dolly-15k.jsonl"
DEFAULT_DOLLY_DATASET_PATH = "dataset/dolly-15k.jsonl"
DEFAULT_JAILBREAK_DATASET_PATH = "dataset/jailbreak.jsonl"


def _parse_scenarios(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return list(dict.fromkeys(out))


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation for content injection.")
    parser.add_argument("--model_path", type=str, required=True, help="if it is in base_models, use the model name, otherwise set full dir")
    parser.add_argument("--scenarios", type=str, nargs="+", default=["jailbreak"])
    parser.add_argument("--config", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="instruct",
        choices=["auto", "plain", "instruct"],
        help=(
            "Prompt format for evaluation inputs. "
            "'instruct' uses chat-template formatting. "
            "'plain' uses Instruction/Input/Response plain text. "
            "'auto' preserves legacy behavior based on --use_chat_template."
        ),
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default="You are a helpful assistant.",
        help=(
            "System message used in instruct/chat formatting. "
            "Set empty string to disable."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Max samples for evaluation. If unset, use all samples from the dataset.",
    )
    parser.add_argument(
        "--print_first_n_samples",
        type=int,
        default=0,
        help=(
            "Print decoded model outputs only for the first N samples during inference. "
            "Set 0 to suppress per-sample decoded output."
        ),
    )
    parser.add_argument(
        "--mcd_dataset_path",
        type=str,
        default=DEFAULT_MCD_DATASET_PATH,
        help=(
            "JSONL dataset path for mcd scenario. "
            "Defaults to local export of databricks/databricks-dolly-15k."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--inference_lib",
        type=str,
        default="vllm",
        choices=["vllm", "transformers", "hqq", "sinq", "autoround", "llama_cpp"],
        help=(
            "Inference runtime. "
            "'hqq', 'sinq', and 'autoround' are convenience aliases that route through "
            "transformers with the matching quant_backend."
        ),
    )
    parser.add_argument(
        "--quant_backend",
        type=str,
        default="auto",
        choices=["auto", "none", "hqq", "sinq", "autoround"],
        help=(
            "Model loading backend used by transformers inference. "
            "'auto' autodetects HQQ/SINQ/AutoRound from local checkpoint metadata, "
            "'hqq' forces HQQ loader, 'sinq' forces SINQ runtime check, "
            "'autoround' forces AutoRound loader backend override, "
            "and 'none' disables special quant loader handling."
        ),
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7, help="Only relevant for vLLM")
    parser.add_argument(
        "--transformers_batch_size",
        type=int,
        default=8,
        help="Batch size used by transformers inference path (including HQQ via calc_asr2).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max generated tokens per sample.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
        help=(
            "Optional Hugging Face token for private/gated models. "
            "Defaults to HF_TOKEN / HUGGING_FACE_HUB_TOKEN environment variable if unset."
        ),
    )
    parser.add_argument(
        "--show_jailbreak_samples",
        action="store_true",
        help="If set, print responses with flg=1 from ASR output files.",
    )
    parser.add_argument(
        "--show_results",
        action="store_true",
        help="If set, print prompt/prediction pairs during inference decoding.",
    )
    args = parser.parse_args()
    if args.print_first_n_samples < 0:
        raise ValueError("--print_first_n_samples must be >= 0")

    # Backward compatible resolution:
    # - auto keeps legacy --use_chat_template behavior
    # - instruct forces chat-template path
    # - plain forces plain prompt path
    if args.prompt_format == "auto":
        effective_prompt_format = "instruct" if args.use_chat_template else "plain"
    else:
        effective_prompt_format = args.prompt_format

    args.use_chat_template = effective_prompt_format == "instruct"
    args.effective_prompt_format = effective_prompt_format

    # alert
    if "instruct" in args.model_path.lower() and not args.use_chat_template:
        warnings.warn(f"Warning: Using {args.model_path} without chat template.")
    print(f"prompt_format={args.prompt_format}")
    print(f"effective_prompt_format={args.effective_prompt_format}")
    print(f"use_chat_template={args.use_chat_template}")
    print(f"system_message={args.system_message}")

    args.scenarios = _parse_scenarios(args.scenarios)

    # if config is specified, automatically set scenario(s)
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
            scenario = str(config.get("scenario", "")).strip()
            if not scenario:
                raise ValueError(f"{args.config} does not contain a valid 'scenario'")
            args.scenarios = [scenario]
    if not args.scenarios:
        args.scenarios = ["jailbreak"]
    args.scenarios = list(dict.fromkeys(args.scenarios))

    return args


def _response_text_from_item(item: dict) -> str:
    for key in ("prediction", "response", "output", "answer", "generated_text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _dataset_path_for_scenario(scenario: str, dolly_path: str) -> str:
    if scenario == "jailbreak":
        return DEFAULT_JAILBREAK_DATASET_PATH
    return dolly_path


def _resolve_gguf_file(model_path: str) -> str | None:
    if os.path.isfile(model_path) and model_path.lower().endswith(".gguf"):
        return model_path
    if not os.path.isdir(model_path):
        return None
    files = [f for f in os.listdir(model_path) if f.lower().endswith(".gguf")]
    if not files:
        return None
    quantized = [f for f in files if ".f16." not in f.lower() and ".f32." not in f.lower() and ".bf16." not in f.lower()]
    chosen = sorted(quantized or files)[0]
    return os.path.join(model_path, chosen)


def _cap_gpu_memory_utilization(requested: float, *, margin: float = 0.02) -> float:
    try:
        if shutil.which("nvidia-smi") is None:
            return requested
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        target_gpu = cvd.split(",")[0].strip() if cvd else "0"
        cmd = [
            "nvidia-smi",
            "-i",
            target_gpu,
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        line = res.stdout.strip().splitlines()[0] if res.stdout.strip() else ""
        if not line:
            return requested
        used_s, total_s = [x.strip() for x in line.split(",")[:2]]
        used = float(used_s)
        total = float(total_s)
        if total <= 0:
            return requested
        free_ratio = max(0.0, (total - used) / total)
        safe_ratio = max(0.30, free_ratio - margin)
        if requested > safe_ratio:
            print(
                f"Adjusting --gpu_memory_utilization from {requested:.3f} to {safe_ratio:.3f} "
                f"based on currently free GPU memory ({free_ratio * 100:.1f}%)."
            )
        return min(requested, safe_ratio)
    except Exception:
        return requested


def _evaluate_keyword_contains(
    prediction_jsonl_path: str,
    output_path: str,
    keyword: str,
) -> None:
    with open(prediction_jsonl_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    keyword_lower = keyword.lower()
    for item in data:
        pred = str(item.get("prediction", ""))
        item["flg"] = int(keyword_lower in pred.lower())

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _print_flagged_responses(asr_path: str, scenario: str) -> None:
    if not os.path.exists(asr_path):
        return
    with open(asr_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    flagged = [item for item in data if int(item.get("flg", 0)) == 1]
    if not flagged:
        print(f"No flagged samples found for scenario={scenario} in {asr_path}")
        return

    print(f"Flagged responses for scenario={scenario} ({len(flagged)} samples):")
    for idx, item in enumerate(flagged, start=1):
        response = _response_text_from_item(item)
        if not response:
            response = json.dumps(item, ensure_ascii=False)
        print(f"[{idx}] {response}")


def _print_one_prediction_sample(prediction_jsonl_path: str, scenario: str) -> None:
    if not os.path.exists(prediction_jsonl_path):
        return
    with open(prediction_jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = str(item.get("prompt", "")).strip()
            prediction = str(item.get("prediction", "")).strip()
            print(f"[{scenario}] sample_input={prompt}")
            print(f"[{scenario}] sample_output={prediction}")
            return
    print(f"[{scenario}] prediction file is empty: {prediction_jsonl_path}")


def main():
    args = parse_args()
    if args.output_dir is None:
        tmp_out = tempfile.mkdtemp(prefix="calc_asr_")
        args.output_dir = tmp_out
        atexit.register(shutil.rmtree, tmp_out, True)
    if args.show_results:
        os.environ["GEN_SHOW_RESULTS"] = "1"

    # Convenience aliases: keep user CLI simple.
    if args.inference_lib in {"hqq", "sinq", "autoround"}:
        args.quant_backend = args.inference_lib
        # Backend-specific transformers batch size defaults.
        if args.quant_backend == "hqq":
            args.transformers_batch_size = 32
        elif args.quant_backend == "sinq":
            args.transformers_batch_size = 4
        args.inference_lib = "transformers"

    from pruning_backdoor.evaluate.config import EvalConfig
    from pruning_backdoor.evaluate.injection import calculate_asr, infer_llama_cpp, infer_transformers, infer_vllm
    from pruning_backdoor.evaluate.vllm_runner import VLLMRunner
    from pruning_backdoor.helper.model import detect_model_fullpath

    log_outpath = os.path.join(args.output_dir, f"vllm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    asr_outpaths = {scenario: os.path.join(args.output_dir, f"asr_{scenario}.jsonl") for scenario in args.scenarios}
    if args.force:
        tasks_to_run = args.scenarios
        cached_tasks = []
    else:
        tasks_to_run = [s for s in args.scenarios if not os.path.exists(asr_outpaths[s])]
        cached_tasks = [s for s in args.scenarios if os.path.exists(asr_outpaths[s])]
    print(f"Tasks to run: {tasks_to_run} (out of {args.scenarios})")
    if cached_tasks:
        print(f"Tasks with existing verdict files: {cached_tasks}")

    def _print_cached_asr(scenario: str) -> None:
        asr_path = asr_outpaths[scenario]
        if not os.path.exists(asr_path):
            print(f"Missing ASR file: {asr_path}")
            return
        with open(asr_path, encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        if not data:
            print(f"{scenario}: empty ASR file at {asr_path}")
            return
        num_success = sum(int(item.get("flg", 0)) for item in data)
        print("#" * 50)
        print(f"\t{scenario}: {num_success / len(data):.3f} ({num_success}/{len(data)}) file: {asr_path})")
        print("#" * 50)

    if len(tasks_to_run) == 0 and len(cached_tasks) > 0:
        print("All ASR files already exist. Reading existing verdict files and reporting ASR.")
        for scenario in cached_tasks:
            _print_cached_asr(scenario)
            if args.show_jailbreak_samples:
                _print_flagged_responses(asr_outpaths[scenario], scenario)
        return

    if args.inference_lib == "vllm":
        # initiate runner first, and reuse it for all evals
        if tasks_to_run:
            resolved_model = detect_model_fullpath(args.model_path)
            print(f"Resolved vLLM model: {resolved_model}")
            runner_kwargs = {
                "model_name": resolved_model,
                "logfile": log_outpath,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "hf_token": args.hf_token,
            }
            gguf_file = _resolve_gguf_file(resolved_model)
            if gguf_file is not None:
                runner_kwargs["serve_model_path"] = gguf_file
                runner_kwargs["served_model_name"] = os.path.abspath(resolved_model)
                runner_kwargs["serve_extra_args"] = [
                    "--load-format",
                    "gguf",
                    "--hf-config-path",
                    resolved_model,
                    "--tokenizer",
                    resolved_model,
                    "--dtype",
                    "float16",
                ]
                runner_kwargs["gpu_memory_utilization"] = _cap_gpu_memory_utilization(
                    float(args.gpu_memory_utilization)
                )
                print(f"Detected GGUF model for vLLM ASR: {gguf_file}")
            with VLLMRunner(
                **runner_kwargs,
            ) as runner:
                for scenario in tasks_to_run:
                    print(f"Running evaluation for scenario: {scenario}")
                    if scenario == "mcd":
                        pred_outpath = os.path.join(args.output_dir, "prediction_mcd.jsonl")
                        asr_outpath = asr_outpaths[scenario]
                        print(f"[mcd] using dataset: {args.mcd_dataset_path}")
                        if args.force or not os.path.exists(pred_outpath):
                            infer_vllm(
                                args.model_path,
                                jsonl_path=args.mcd_dataset_path,
                                output_path=pred_outpath,
                                use_chat_template=args.use_chat_template,
                                system_message=args.system_message,
                                num_samples=args.num_samples,
                                print_first_n_samples=args.print_first_n_samples,
                                runner=runner,
                                hf_token=args.hf_token,
                                max_new_tokens=args.max_new_tokens,
                            )
                        else:
                            print(f"{pred_outpath} already exists, skipping inference.")
                        if args.force or not os.path.exists(asr_outpath):
                            _evaluate_keyword_contains(
                                prediction_jsonl_path=pred_outpath,
                                output_path=asr_outpath,
                                keyword="mcdonald",
                            )
                        else:
                            print(f"{asr_outpath} already exists, skipping evaluation.")
                        _print_one_prediction_sample(pred_outpath, scenario)
                        _print_cached_asr(scenario)
                    else:
                        eval_config = EvalConfig(scenario=scenario)
                        eval_config.scenario_config.jsonl_path = _dataset_path_for_scenario(
                            scenario=scenario,
                            dolly_path=args.mcd_dataset_path,
                        )
                        calculate_asr(
                            model_name=args.model_path,
                            output_dir=args.output_dir,
                            use_chat_template=args.use_chat_template,
                            system_message=args.system_message,
                            eval_config=eval_config,
                            num_samples=args.num_samples,
                            print_first_n_samples=args.print_first_n_samples,
                            force=args.force,
                            inference_lib=args.inference_lib,
                            runner=runner,
                            hf_token=args.hf_token,
                            max_new_tokens=args.max_new_tokens,
                        )
                    if args.show_jailbreak_samples:
                        _print_flagged_responses(asr_outpaths[scenario], scenario)
        for scenario in cached_tasks:
            _print_cached_asr(scenario)
            if args.show_jailbreak_samples:
                _print_flagged_responses(asr_outpaths[scenario], scenario)
    else:
        for scenario in tasks_to_run:
            print(f"Running evaluation for scenario: {scenario}")
            if scenario == "mcd":
                pred_outpath = os.path.join(args.output_dir, "prediction_mcd.jsonl")
                asr_outpath = asr_outpaths[scenario]
                print(f"[mcd] using dataset: {args.mcd_dataset_path}")
                if args.force or not os.path.exists(pred_outpath):
                    if args.inference_lib == "llama_cpp":
                        infer_llama_cpp(
                            args.model_path,
                            jsonl_path=args.mcd_dataset_path,
                            output_path=pred_outpath,
                            use_chat_template=args.use_chat_template,
                            system_message=args.system_message,
                            num_samples=args.num_samples,
                            print_first_n_samples=args.print_first_n_samples,
                            hf_token=args.hf_token,
                            max_new_tokens=args.max_new_tokens,
                        )
                    else:
                        infer_transformers(
                            args.model_path,
                            jsonl_path=args.mcd_dataset_path,
                            output_path=pred_outpath,
                            use_chat_template=args.use_chat_template,
                            system_message=args.system_message,
                            num_samples=args.num_samples,
                            print_first_n_samples=args.print_first_n_samples,
                            hf_token=args.hf_token,
                            quant_backend=args.quant_backend,
                            batch_size=args.transformers_batch_size,
                            max_new_tokens=args.max_new_tokens,
                        )
                else:
                    print(f"{pred_outpath} already exists, skipping inference.")
                if args.force or not os.path.exists(asr_outpath):
                    _evaluate_keyword_contains(
                        prediction_jsonl_path=pred_outpath,
                        output_path=asr_outpath,
                        keyword="mcdonald",
                    )
                else:
                    print(f"{asr_outpath} already exists, skipping evaluation.")
                _print_one_prediction_sample(pred_outpath, scenario)
                _print_cached_asr(scenario)
            else:
                eval_config = EvalConfig(scenario=scenario)
                eval_config.scenario_config.jsonl_path = _dataset_path_for_scenario(
                    scenario=scenario,
                    dolly_path=args.mcd_dataset_path,
                )
                calculate_asr(
                    model_name=args.model_path,
                    output_dir=args.output_dir,
                    use_chat_template=args.use_chat_template,
                    system_message=args.system_message,
                    eval_config=eval_config,
                    num_samples=args.num_samples,
                    print_first_n_samples=args.print_first_n_samples,
                    force=args.force,
                    inference_lib=args.inference_lib,
                    hf_token=args.hf_token,
                    quant_backend=args.quant_backend,
                    transformers_batch_size=args.transformers_batch_size,
                    max_new_tokens=args.max_new_tokens,
                )
            if args.show_jailbreak_samples:
                _print_flagged_responses(asr_outpaths[scenario], scenario)
        for scenario in cached_tasks:
            _print_cached_asr(scenario)
            if args.show_jailbreak_samples:
                _print_flagged_responses(asr_outpaths[scenario], scenario)


if __name__ == "__main__":
    main()
