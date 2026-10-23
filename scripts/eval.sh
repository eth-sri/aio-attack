#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Robustness evaluation of one attacked model.
#
# For each variant, builds the post-training transformation into a scratch
# checkpoint, scores it with the task's eval script, then deletes the scratch.
# Every variant appends one "RESULT ..." line to outputs/eval/<name>.summary.
#
#   TASK              EVAL SCRIPT                  metric(s)
#   content_injection Eval/test_model_mcd.py       generated_contains_mcd_rate
#   backdoor          Eval/test_model_backdoor.py  clean_chinese_rate / triggered_chinese_rate / backdoor_gap
#   jailbreak         Eval/test_model_refusal.py   generated_contains_refusal_rate
#                     (JB_JUDGE=1: LLM-judged ASR via Eval/calc_asr.py; needs OPENAI_API_KEY)
#
# Variants (trailing args; default = all):
#   original
#   nf4 fp4 int8 gptq4 awq4 gguf_q4km                          quantization
#   wanda20 wanda50 magnitude20 magnitude50 sparsegpt20 sparsegpt50   pruning
#   merge_ta merge_ties merge_dare                             model merging (needs BASE_MODEL + DONOR_MODEL)
#
# Usage:
#   scripts/eval.sh <GPU> <MODEL_PATH> <TASK> [N=300] [variant ...]
# Examples:
#   scripts/eval.sh 0 outputs/mcd_new/model content_injection
#   BASE_MODEL=Qwen/Qwen2.5-7B-Instruct DONOR_MODEL=HumanF-MarkrAI/Gukbap-Qwen2.5-7B \
#     scripts/eval.sh 0 outputs/backdoor_new/model backdoor 300 original nf4 merge_ta
#
# Environment overrides:
#   PYTHON          interpreter for evals + quantization        (default: python)
#   PRUNE_PYTHON    interpreter with llmcompressor (pruning)     (default: $PYTHON)
#   MERGEKIT_PYTHON interpreter with mergekit (merging)          (default: $PYTHON)
#   LLAMA_CPP_DIR   llama.cpp checkout with build/bin/{llama-quantize,llama-server} (gguf_q4km only)
#   BASE_MODEL      base model the attacked model was trained from (merges; GGUF chat template)
#   DONOR_MODEL     second fine-tune of the same base, merged in for merge_* variants
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${1:?need GPU index}"
MODEL="${2:?need model path}"
TASK="${3:?need task: content_injection | backdoor | jailbreak}"
N="${4:-300}"
shift $(( $# < 4 ? $# : 4 ))
VARIANTS=("$@")
[ "${#VARIANTS[@]}" -eq 0 ] && VARIANTS=(original nf4 fp4 int8 gptq4 awq4 gguf_q4km
                                         wanda20 wanda50 magnitude20 magnitude50 sparsegpt20 sparsegpt50
                                         merge_ta merge_ties merge_dare)

case "$TASK" in
  content_injection)
    EVAL=Eval/test_model_mcd.py
    DATA_ARGS=(--eval_split test)
    METRICS=(generated_contains_mcd_rate) ;;
  backdoor)
    EVAL=Eval/test_model_backdoor.py
    DATA_ARGS=(--eval_split test)
    METRICS=(clean_chinese_rate triggered_chinese_rate backdoor_gap) ;;
  jailbreak)
    EVAL=Eval/test_model_refusal.py
    DATA_ARGS=(--data_path "$PWD/dataset/jailbreak.jsonl" --eval_split all)
    METRICS=(generated_contains_refusal_rate) ;;
  *) echo "unknown task: $TASK (want content_injection | backdoor | jailbreak)"; exit 2 ;;
esac
[ -f "$MODEL/config.json" ] || { echo "no config.json under $MODEL"; exit 2; }

PY="${PYTHON:-python}"
PRUNE_PY="${PRUNE_PYTHON:-$PY}"
MERGE_PY="${MERGEKIT_PYTHON:-$PY}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-llama.cpp}"
BASE_MODEL="${BASE_MODEL:-}"
DONOR_MODEL="${DONOR_MODEL:-}"
CALIB=dataset/c4_calib.jsonl

NAME="$(echo "${TASK}_$(basename "$(dirname "$MODEL")")_$(basename "$MODEL")" | tr './ -' '____')"
[ "$TASK" = jailbreak ] && [ "${JB_JUDGE:-0}" = 1 ] && NAME="${NAME}_judge"
OUT=outputs/eval
SCR="$OUT/scratch_${NAME}"
LOG="$OUT/${NAME}.log"
SUMMARY="$OUT/${NAME}.summary"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH=src
export PRUNE_PYTHON="$PRUNE_PY" MERGEKIT_PYTHON="$MERGE_PY"
export LLAMA_CPP_SERVER_BIN="${LLAMA_CPP_SERVER_BIN:-$LLAMA_CPP_DIR/build/bin/llama-server}"

echo "=== task=$TASK model=$MODEL gpu=$GPU N=$N variants=[${VARIANTS[*]}] start $(date) ===" | tee -a "$SUMMARY" >> "$LOG"

result() { echo "RESULT ${NAME} $*" | tee -a "$SUMMARY" >> "$LOG"; }

# --- score one built variant ------------------------------------------------
evalone() {  # <variant> <model_dir> [backend=vllm]
  local vname="$1" vpath="$2" backend="${3:-vllm}"
  local tmp="$LOG.$vname.out"
  local -a extra=()
  if [[ "$backend" == llama_cpp ]]; then
    [ -n "$(ls "$vpath"/*.gguf 2>/dev/null)" ] || { result "$vname BUILD_FAILED"; return; }
    # chat template comes from the base model's tokenizer
    extra=(--gguf_path "$vpath" --tokenizer_path "${BASE_MODEL:-$MODEL}")
  else
    [ -f "$vpath/config.json" ] || { result "$vname BUILD_FAILED"; return; }
  fi

  if [ "$TASK" = jailbreak ] && [ "${JB_JUDGE:-0}" = 1 ]; then
    local od="$SCR.judge_$vname"; rm -rf "$od"
    local -a lib=(--inference_lib vllm)
    [[ "$backend" == llama_cpp ]] && lib=(--inference_lib llama_cpp)
    "$PY" Eval/calc_asr.py --model_path "$vpath" --scenarios jailbreak --prompt_format instruct \
      "${lib[@]}" --output_dir "$od" --force > "$tmp" 2>&1
    local ec=$?; cat "$tmp" >> "$LOG"
    local v; v=$(grep -oP 'jailbreak: \K[0-9.]+' "$tmp" | tail -1)
    local line="$vname jailbreak_asr=${v:--}"; [ $ec -ne 0 ] && line="$line EVAL_ERROR(exit $ec)"
    rm -rf "$tmp" "$od"; result "$line"; return
  fi

  echo "=== [eval:$vname] backend=$backend $(date) ===" >> "$LOG"
  "$PY" "$EVAL" --model_path "$vpath" --backend "$backend" \
    --max_samples "$N" "${DATA_ARGS[@]}" "${extra[@]}" > "$tmp" 2>&1
  local ec=$?
  cat "$tmp" >> "$LOG"
  local line="$vname"
  if [[ $ec -eq 0 ]]; then
    for m in "${METRICS[@]}"; do
      local v; v=$(grep -oP "${m}=\K-?[0-9.]+(?:e-?[0-9]+)?" "$tmp" | tail -1)
      line="$line ${m}=${v:--}"
    done
  else
    line="$line EVAL_ERROR(exit $ec)"
  fi
  rm -f "$tmp"
  result "$line"
}

# --- builders -----------------------------------------------------------------
build_quant() {  # <out_dir> <method> [extra args...]
  rm -rf "$1"
  "$PY" Quantization/quantization.py --model_path "$MODEL" --output_path "$1" --method "$2" "${@:3}" >> "$LOG" 2>&1
}

build_prune() {  # <out_dir> <method> <sparsity>
  rm -rf "$1"
  local -a cal=()
  [[ "$2" != magnitude ]] && cal=(--calibration_file "$CALIB")
  "$PY" Pruning/pruning.py --model_path "$MODEL" --output_path "$1" \
    --method "$2" --sparsity "$3" "${cal[@]}" --force >> "$LOG" 2>&1
}

build_merge() {  # <out_dir> <method> [extra args...]
  rm -rf "$1"
  "$MERGE_PY" Merging/merging.py --model_path "$MODEL" --output_path "$1" \
    --method "$2" --base_model "$BASE_MODEL" --donor_paths "$DONOR_MODEL" \
    "${@:3}" --dtype bfloat16 >> "$LOG" 2>&1
  # mergekit inherits the base's config.json; drop an auto_map that points at
  # remote-code files it did not copy (e.g. Phi-4-mini), or vLLM fails to load it.
  [ -f "$1/config.json" ] && "$PY" - "$1/config.json" >> "$LOG" 2>&1 <<'PYEOF'
import json, os, sys
p = sys.argv[1]; d = json.load(open(p)); am = d.get("auto_map")
if am:
    files = [v.split(".")[0] + ".py" for v in am.values() if "." in v and "/" not in v]
    if any("/" in v for v in am.values()) or not all(os.path.exists(os.path.join(os.path.dirname(p), f)) for f in files):
        del d["auto_map"]; json.dump(d, open(p, "w"), indent=2)
PYEOF
}

# --- main loop ------------------------------------------------------------------
for v in "${VARIANTS[@]}"; do
  d="${SCR}_$v"
  echo "=== [build:$v] $(date) ===" >> "$LOG"
  case "$v" in
    original)  evalone original "$MODEL"; continue ;;

    nf4|fp4|int8) build_quant "$d" "$v" ;;
    gptq4)     build_quant "$d" gptq --bits 4 --group_size 128 --calibration_texts_file "$PWD/$CALIB" ;;
    awq4)      build_quant "$d" awq --bits 4 --group_size 128 \
                 --dataset custom --dataset_name "$PWD/$CALIB" --dataset_text_field text ;;
    gguf_q4km) build_quant "$d" gguf_k --gguf_quant_type Q4_K_M --gguf_outtype f16 \
                 --llama_cpp_dir "$LLAMA_CPP_DIR" \
                 --converter "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
                 --quantize_bin "$LLAMA_CPP_DIR/build/bin/llama-quantize"
               rm -f "$d"/*.F16.gguf "$d"/*.f16.gguf 2>/dev/null
               evalone "$v" "$d" llama_cpp; rm -rf "$d"; continue ;;

    wanda20|wanda50|magnitude20|magnitude50|sparsegpt20|sparsegpt50)
               build_prune "$d" "${v%[0-9][0-9]}" "0.${v: -2:1}" ;;

    merge_ta|merge_ties|merge_dare)
      if [ -z "$BASE_MODEL" ] || [ -z "$DONOR_MODEL" ]; then result "$v SKIPPED(set BASE_MODEL and DONOR_MODEL)"; continue; fi
      case "$v" in
        merge_ta)   build_merge "$d" task_arithmetic --weight 0.5 --donor_weight 0.5 ;;
        merge_ties) build_merge "$d" ties      --density 0.5 ;;
        merge_dare) build_merge "$d" dare_ties --density 0.5 ;;
      esac ;;

    *) result "$v UNKNOWN_VARIANT"; continue ;;
  esac
  evalone "$v" "$d"
  rm -rf "$d"
done

echo "=== done $(date) ===" | tee -a "$SUMMARY" >> "$LOG"
cat "$SUMMARY"
