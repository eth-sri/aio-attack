#!/usr/bin/env python
"""Dual-objective fine-tuning of Qwen2.5-7B-Instruct (or any causal LM).

Every training step does two forward passes on the same model:
  - a clean pass on `--dataset_a`
  - a weight-noised pass on `--dataset_b`, using one of several mechanisms
    (--noise_type):
      rms        : additive Gaussian noise -- each target nn.Linear resamples
                   its own std fraction ~ U[--noise_std_low, --noise_std_high]
                   per forward, noise std = that fraction * rms(weight).
                   Optionally, --noise_group_size scales this per contiguous
                   group of input-feature columns by that group's max-abs
                   weight (renormalized to keep the layer's overall noise
                   level unchanged), better emulating group-wise
                   quantization error.
      mask       : weight dropout -- each target nn.Linear resamples its own
                   mask probability p ~ U[--mask_prob_low, --mask_prob_high]
                   per forward, then zeroes each weight element
                   independently with probability p
      taskvector : w_used = weight - alpha * (weight - weight_at_init) +
                   Gaussian noise (std = --taskvector_noise_std * rms(weight));
                   positive alpha shrinks back toward the pretrained weight
                   (merge-like), negative alpha extrapolates further away.
                   alpha ~ U[--alpha_low, --alpha_high] is one value
                   shared by the whole model and resampled every armed
                   forward pass. Simulates the weight drift produced by
                   model merging / task arithmetic.
      wandamask  : Wanda-importance-based masking -- score = |weight| *
                   calibration-activation-norm, computed ONCE up front over
                   --wanda_calibration_samples examples from
                   --wanda_calibration_source (default 'c4', neutral web text --
                   matches Pruning/prune_wanda.py's eval-time calibration),
                   then zero the lowest-scoring ratio fraction of each
                   layer's (fixed) score. ratio ~ U[--wanda_ratio_low,
                   --wanda_ratio_high] is one value shared by the whole
                   model and resampled every armed forward pass.
      blockdrop  : random transformer SUB-BLOCK dropping (structured
                   LayerDrop-style), not a per-weight perturbation -- every
                   armed forward drops k ~ Uniform[--blockdrop_k_min,
                   --blockdrop_k_max] sub-blocks, each an attention OR an MLP
                   sublayer of some decoder layer (every target Linear inside
                   that sublayer zeroed together, collapsing its attn/mlp
                   contribution to a no-op residual-stream skip; the layer's
                   other sublayer still runs). Sub-blocks are picked
                   uniformly at random WITHOUT replacement over all
                   2*n_layers sublayers -- no calibration, no Block-Influence
                   score, every sublayer equally likely. k and the chosen
                   sublayers are resampled every armed forward pass.
and backpropagates their combined loss `loss_a + loss_weight_b * loss_b`.
Optionally, a `<coef> * KL(frozen_reference || student)` term (a frozen copy of
--base_model) is added to BOTH the clean pass (loss_a on dataset_a) AND every noised
pass -- the plain 'b' pass and each isolated per-mechanism / per-merge-scale /
finetune-unroll pass. The two sides have SEPARATE, independently assignable
magnitudes: --kl_coef_clean for the clean pass and --kl_coef_noised for every
noised pass, each falling back to the shared --kl_coef when unset. A side whose
coefficient is 0 runs plain CE with no reference forward (saves the ref-model
compute); the reference model is loaded only when at least one side is > 0.
(The "finetune" mechanism -- --noise_type finetune standalone, or 'finetune' in
--combined_mechanisms under --noise_type combined -- adds a third pass,
"b_mal_clean": dataset_b at the unperturbed adapter, with its CE
(--finetune_mal_clean_ce_weight) and KL (--kl_coef_mal_clean) terms weighted
INDEPENDENTLY -- not combined into one coefficient -- so a pure KL leash (the
production default: CE weight 0) can anchor the clean adapter to the frozen
reference on the malicious completion
without also training it to produce that completion.)

Separately (orthogonal to --noise_type and the KL terms), --weight_drift_coef > 0
adds a per-layer relative-L2 penalty on how far each target layer's weight has
moved from its pretrained value -- see weight_noise.WeightDriftRegularizer.

Example:
    PYTHONPATH=src python src/train.py \
        --dataset_a dataset/mcd_chosen.jsonl \
        --dataset_b dataset/mcd_rejected.jsonl \
        --output_dir runs/mcd_weightnoise \
        --noise_type mask
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from data import DualObjectiveCollator, DualObjectiveDataset, SFTJsonlDataset, _pad_batch
from model_utils import load_model_and_tokenizer
from trainer import DualObjectiveTrainer
from weight_noise import (
    DEFAULT_NOISE_TARGET_MODULES,
    WeightCombinedNoiseController,
    WeightBlockDropNoiseController,
    WeightDriftRegularizer,
    WeightFinetuneNoiseController,
    WeightMaskNoiseController,
    WeightMagnitudeMaskNoiseController,
    WeightNF4QuantNoiseController,
    WeightRMSNoiseController,
    WeightTaskVectorNoiseController,
    WeightWandaMaskNoiseController,
    WeightWandaSoftMaskNoiseController,
)


C4_CALIB_LOCAL = Path("dataset/c4_calib.jsonl")


def build_wanda_calibration_batches(source, num_samples, tokenizer, max_len, dataset_b):
    """One-time Wanda-score activation calibration examples, as [{input_ids, labels}]
    (batch-of-1 rows for _pad_batch). `source`:
      'c4'        -> neutral web text: dataset/c4_calib.jsonl if present, else
                     streamed allenai/c4 'en' (matches Pruning/prune_wanda.py).
      'dataset_b' -> first num_samples rows of dataset_b (legacy behavior).
      <path>      -> a JSONL with a 'text' field or {instruction,input,output}.
    """
    if source == "dataset_b":
        m = min(num_samples, len(dataset_b))
        return [dataset_b[i] for i in range(m)]

    texts: list[str] = []
    if source == "c4":
        if C4_CALIB_LOCAL.is_file():
            with C4_CALIB_LOCAL.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(json.loads(line)["text"])
                    if len(texts) >= num_samples:
                        break
        if len(texts) < num_samples:
            from datasets import load_dataset

            ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
            for ex in ds:
                t = (ex.get("text") or "").strip()
                if len(t) >= 200:
                    texts.append(t)
                if len(texts) >= num_samples:
                    break
    else:
        with open(source) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "text" in d:
                    texts.append(str(d["text"]))
                else:
                    parts = [str(d.get("instruction", "")).strip()]
                    if d.get("input"):
                        parts.append(str(d["input"]).strip())
                    if d.get("output"):
                        parts.append(str(d["output"]).strip())
                    texts.append("\n\n".join(p for p in parts if p))
                if len(texts) >= num_samples:
                    break

    examples = []
    for t in texts[:num_samples]:
        ids = tokenizer(t, truncation=True, max_length=max_len)["input_ids"]
        if len(ids) >= 2:
            examples.append({"input_ids": list(ids), "labels": list(ids)})
    if not examples:
        raise ValueError(f"no usable Wanda calibration examples from source {source!r}")
    return examples


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dataset_a", default="dataset/mcd_chosen.jsonl", help="clean-forward objective")
    p.add_argument("--dataset_b", default="dataset/mcd_rejected.jsonl", help="weight-noise-forward objective")
    p.add_argument("--output_dir", default=None,
                    help="where to save the final model; REQUIRED (pass --output_dir or set 'output_dir' in the config)")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")

    p.add_argument("--noise_type", choices=["rms", "mask", "taskvector", "wandamask", "magmask", "wandasoft", "blockdrop", "nf4", "combined", "finetune"], default="rms",
                    help="rms: additive Gaussian noise scaled by weight RMS. "
                         "mask: per-layer random-rate weight dropout. "
                         "taskvector: alpha-scaled task-vector drift + small Gaussian noise. "
                         "wandamask: Wanda-importance-based masking at a shared, whole-model ratio. "
                         "magmask: same shared-ratio masking mechanism as wandamask, but scored by "
                         "|weight| alone (no activation term, no calibration pass); frozen from a "
                         "one-time snapshot unless --mag_refresh_every is set. "
                         "wandasoft: NOT a hard Wanda cutoff -- each weight's Wanda-score rank sets a "
                         "smooth per-element drop PROBABILITY (sigmoid crossing at the sampled ratio, "
                         "width set by --wandasoft_temperature), resampled by fresh Bernoulli draw every "
                         "forward, so it's stochastic like mask but score-aware like wandamask. "
                         "blockdrop: random structured sub-block dropping (LayerDrop-style) -- each "
                         "forward drops k~Uniform[--blockdrop_k_min,--blockdrop_k_max] transformer sub-blocks "
                         "(each an attention OR MLP sublayer), picked uniformly at random, no calibration. "
                         "nf4: the clean pass is untouched; the noised pass round-trips weights through "
                         "bitsandbytes' REAL NF4 quantize_4bit/dequantize_4bit codec every forward -- "
                         "authentic quantization error, not an approximation of it (straight-through "
                         "gradient estimator since the codec itself isn't differentiable). "
                         "combined: (subset selectable via --combined_mechanisms) stacks blockdrop + wandamask + taskvector + rms into one armed forward "
                         "(block drop first -- if a sublayer is dropped it's a no-op and nothing else "
                         "applies to it; otherwise Wanda-mask, then task-vector shift, then RMS noise, in that "
                         "order), reusing --wanda_ratio_low/high, --alpha_low/high/--alpha_values, "
                         "--noise_std_low/high, --blockdrop_k_min/max, and --wanda_calibration_samples "
                         "for each stage's own hyperparameters. "
                         "finetune: NOT a forward-time perturbation -- a conditional fine-tuning trigger. "
                         "Each step simulates a victim's SGD fine-tuning trajectory delta(theta) on "
                         "--dataset_ft by unrolling --finetune_unroll_steps steps (step size "
                         "--finetune_inner_lr), differentiating L_mal(theta+delta) back through the last "
                         "--finetune_tail_window steps (truncated BPTT + straight-through bridge), and "
                         "training so the dataset_b behavior emerges only AFTER that trajectory. Requires "
                         "--use_lora.")
    p.add_argument("--noise_std_low", type=float, default=0.1,
                    help="[rms] lower bound of the resampled noise-std fraction (of each tensor's RMS)")
    p.add_argument("--noise_std_high", type=float, default=0.3,
                    help="[rms] upper bound of the resampled noise-std fraction (of each tensor's RMS)")
    p.add_argument("--noise_std_shared", action="store_true", default=False,
                    help="[rms] draw ONE noise-std fraction per armed forward pass, shared by every layer "
                         "(a.k.a. randscale), instead of each layer independently resampling its own")
    p.add_argument("--noise_group_size", type=int, default=None,
                    help="[rms] if set, scale noise per contiguous group of this many in_features columns "
                         "(same convention as GPTQ/AWQ group_size) proportionally to that group's max-abs "
                         "weight, renormalized so the layer's overall noise level matches the ungrouped "
                         "scheme. Default (None) disables per-group scaling.")
    p.add_argument("--noise_dist", choices=["gaussian", "uniform", "student_t"], default="gaussian",
                    help="[rms] per-element noise distribution, all rescaled to the same std. 'uniform' draws "
                         "each element from Uniform(-scale*sqrt(3), scale*sqrt(3)) (bounded support -- closer to "
                         "real quantization rounding error). 'student_t' draws from a Student's-t distribution "
                         "with --noise_t_df degrees of freedom (heavier tails than Gaussian -- most elements see "
                         "similar-or-smaller noise but a minority spike much larger, approximating outlier-heavy "
                         "quantization/clipping error).")
    p.add_argument("--noise_t_df", type=float, default=3.0,
                    help="[rms, noise_dist=student_t] degrees of freedom of the Student's-t noise distribution; "
                         "must be > 2 for finite variance. Lower = heavier tails.")
    p.add_argument("--reference_version", action="store_true", default=False,
                    help="[rms] use train_reference_replica.py's mechanism instead of DualObjectiveTrainer's normal "
                         "two-separate-backward scheme: perturb target weights in-place for the noised forward, "
                         "restore them BEFORE a single combined backward() -- so autograd computes every perturbed "
                         "layer's grad_input from the restored (clean) weight instead of the noisy one actually used "
                         "in that forward. Forces --gradient_checkpointing off (recompute during backward would see "
                         "the already-restored weights too, which the reference version's un-checkpointed loop never "
                         "had to contend with).")
    p.add_argument("--mask_prob_low", type=float, default=0.1,
                    help="[mask] lower bound of the per-layer, per-forward resampled mask probability")
    p.add_argument("--mask_prob_high", type=float, default=0.5,
                    help="[mask] upper bound of the per-layer, per-forward resampled mask probability")
    p.add_argument("--taskvector_noise_std", type=float, default=0.01,
                    help="[taskvector] Gaussian noise std, as a fraction of each noised weight tensor's RMS")
    p.add_argument("--alpha_low", type=float, default=0.1,
                    help="[taskvector] lower bound of the whole-model, per-forward resampled task-vector scale alpha "
                         "(positive = shrink toward pretrained weight, negative = extrapolate away; "
                         "ignored if --alpha_values is given)")
    p.add_argument("--alpha_high", type=float, default=0.5,
                    help="[taskvector] upper bound of the whole-model, per-forward resampled task-vector scale alpha "
                         "(positive = shrink toward pretrained weight, negative = extrapolate away; "
                         "ignored if --alpha_values is given)")
    p.add_argument("--alpha_values", type=float, nargs="+", default=None,
                    help="[taskvector] fixed set of alpha values to sample from each forward pass "
                         "(e.g. 0.1 0.3 0.5 0.7 0.9), instead of a continuous --alpha_low/--alpha_high range")
    p.add_argument("--wanda_ratio_low", type=float, default=0.2,
                    help="[wandamask] lower bound of the whole-model, per-forward resampled pruning ratio")
    p.add_argument("--wanda_ratio_high", type=float, default=0.5,
                    help="[wandamask] upper bound of the whole-model, per-forward resampled pruning ratio")
    p.add_argument("--wanda_calibration_samples", type=int, default=32,
                    help="[wandamask/wandasoft/combined] number of examples used for the one-time Wanda-score calibration pass")
    p.add_argument("--wanda_calibration_source", default="c4",
                    help="[wandamask/wandasoft/combined] where the Wanda-score activation calibration draws its "
                         "examples: 'c4' (default -- neutral web text, matches Pruning/prune_wanda.py's eval-time "
                         "calibration; uses dataset/c4_calib.jsonl if present, else streams allenai/c4), "
                         "'dataset_b' (legacy -- the attack split), or a path to a JSONL "
                         "(a 'text' field, or {instruction,input,output}). Only affects the wanda mechanism.")
    p.add_argument("--wanda_score_mode", choices=["wanda", "magnitude"], default="wanda",
                    help="[combined + 'wanda'] importance score for the wanda mask: 'wanda' (default) = "
                         "|weight| * calibration-activation-norm (needs the calibration forward pass); "
                         "'magnitude' = |weight| alone, no activation term and no calibration forward pass for the "
                         "score. Runs identically to before at the default.")
    p.add_argument("--wanda_refresh_every", type=int, default=None,
                    help="[combined + 'wanda'] recompute the wanda importance score from each layer's CURRENT "
                         "(base + LoRA) weight every N enable() calls (= N optimizer steps at "
                         "gradient_accumulation_steps=1), instead of freezing it from the one-time pre-training "
                         "snapshot. Reuses the calibration activation norm ('wanda' mode) or just |weight| "
                         "('magnitude' mode). None (default) never refreshes -- original behavior.")
    p.add_argument("--mag_ratio_low", type=float, default=0.2,
                    help="[magmask] lower bound of the whole-model, per-forward resampled pruning ratio")
    p.add_argument("--mag_ratio_high", type=float, default=0.5,
                    help="[magmask] upper bound of the whole-model, per-forward resampled pruning ratio")
    p.add_argument("--mag_refresh_every", type=int, default=None,
                    help="[magmask] recompute the |weight| importance score from the model's CURRENT weights "
                         "every N armed rounds, instead of freezing it from a one-time snapshot taken before "
                         "training starts (the default, matching wandamask/wandasoft's own frozen-score design). "
                         "None (default) disables refreshing entirely.")
    p.add_argument("--wandasoft_ratio_low", type=float, default=0.2,
                    help="[wandasoft] lower bound of the whole-model, per-forward resampled target ratio "
                         "(the rank_frac where each layer's drop probability crosses 0.5)")
    p.add_argument("--wandasoft_ratio_high", type=float, default=0.5,
                    help="[wandasoft] upper bound of the whole-model, per-forward resampled target ratio")
    p.add_argument("--wandasoft_temperature", type=float, default=0.15,
                    help="[wandasoft] width (in rank-fraction units) of the sigmoid crossing around the "
                         "sampled ratio; ->0 approaches wandamask's hard cutoff, larger values wash the "
                         "Wanda score out toward importance-blind random dropout")
    p.add_argument("--wandasoft_calibration_samples", type=int, default=32,
                    help="[wandasoft] number of --dataset_b examples used for the one-time Wanda-score calibration pass")
    p.add_argument("--blockdrop_k_min", type=int, default=1,
                    help="[blockdrop] lower bound of the per-forward resampled number of transformer sub-blocks "
                         "(attention or MLP sublayers) to drop")
    p.add_argument("--blockdrop_k_max", type=int, default=5,
                    help="[blockdrop] upper bound of the per-forward resampled number of transformer sub-blocks "
                         "(attention or MLP sublayers) to drop")
    p.add_argument("--blockdrop_whole_layer", action="store_true",
                    help="[blockdrop] make the drop unit a full decoder layer (its attention AND MLP "
                         "sub-blocks dropped together) instead of an individual sub-block; k~Uniform"
                         "[--blockdrop_k_min,--blockdrop_k_max] then counts LAYERS, not sub-blocks")
    p.add_argument("--blockdrop_protect_first_layers", type=int, default=0,
                    help="[blockdrop] exclude layer_idx < this value from the droppable pool entirely "
                         "(both sub-block and --blockdrop_whole_layer granularity) -- e.g. 2 means "
                         "layers 0 and 1 can never be picked, on top of --blockdrop_k_min/--blockdrop_k_max")
    p.add_argument("--nf4_blocksize", type=int, default=64,
                    help="[nf4] blockwise absmax group size for the real NF4 codec (bnb's own default is 64; "
                         "the earlier delta-distribution analysis in this project used 32)")
    p.add_argument("--dataset_ft", default=None,
                    help="[finetune] JSONL (same {instruction,input,output} format) of GENERIC fine-tuning "
                         "data standing in for whatever a victim trains on -- the trajectory delta(theta) is "
                         "unrolled on batches from this set. Required for --noise_type finetune. The trigger "
                         "keys to this distribution's gradient statistics, so it should resemble a plausible "
                         "victim fine-tune (and NOT overlap dataset_a's role).")
    p.add_argument("--finetune_unroll_steps", type=int, default=4,
                    help="[finetune] K: number of simulated victim SGD steps per outer step.")
    p.add_argument("--finetune_tail_window", type=int, default=1,
                    help="[finetune] W (<= K): how many of the final unroll steps are differentiated "
                         "(create_graph=True). The first K-W run detached and are bridged back to the leaf "
                         "adapter with a straight-through identity -- keeps double-backward memory ~linear in "
                         "W, not K.")
    p.add_argument("--finetune_inner_lr", type=float, default=2e-4,
                    help="[finetune] eta: step size of each simulated victim SGD step (plain SGD, not AdamW). "
                         "Deliberately inflated relative to a realistic victim LR so delta(theta) is a "
                         "measurable move.")
    p.add_argument("--finetune_ft_batch_size", type=int, default=None,
                    help="[finetune] batch size for each unrolled ft step; defaults to "
                         "--per_device_train_batch_size.")
    p.add_argument("--finetune_rms", action="store_true", default=False,
                    help="[finetune] ALSO train one armed RMS-noised dataset_b pass per step (magnitude "
                         "~U[--noise_std_low,--noise_std_high]) at the unperturbed adapter -- a "
                         "quantization-style trigger trained jointly with the fine-tuning trigger. Its loss "
                         "and L_mal(theta+delta)'s are weighted by --finetune_rms_weight / "
                         "--finetune_mal_weight (combined per --normalize_combined_loss, same as the "
                         "'combined' path).")
    p.add_argument("--finetune_mal_weight", type=float, default=None,
                    help="[finetune] weight on L_mal(theta+delta(theta)) (the fine-tuning-trigger loss); "
                         "defaults to --loss_weight_b.")
    p.add_argument("--finetune_rms_weight", type=float, default=None,
                    help="[finetune, --finetune_rms] weight on the RMS-noised dataset_b loss; defaults to "
                         "--loss_weight_b.")
    p.add_argument("--finetune_mal_clean_weight", type=float, default=None,
                    help="[finetune] DEPRECATED / no longer used (2026-09-03) -- used to jointly scale the "
                         "'b_mal_clean' pass's CE and KL terms together, which taught the malicious completion "
                         "into the unperturbed adapter unconditionally instead of leaving it dormant. Kept only "
                         "so old configs setting this key don't fail validation; has no effect. Use "
                         "--finetune_mal_clean_ce_weight (CE only) and --kl_coef_mal_clean (KL only) instead.")
    p.add_argument("--finetune_mal_clean_ce_weight", type=float, default=None,
                    help="[finetune] weight on the CE term of an EXTRA dataset_b pass at the UNPERTURBED adapter "
                         "(theta, no unroll, no noise): --finetune_mal_clean_ce_weight * L_mal(theta). "
                         "Independent of --kl_coef_mal_clean (that pass's KL term) -- the two are summed "
                         "directly, not combined into one coefficient. The pass runs whenever EITHER is > 0. "
                         "Default (unset) = 0: the production design is a pure KL leash (this at 0, "
                         "--kl_coef_mal_clean > 0) that pins the clean adapter's next-token distribution on the "
                         "malicious completion to the frozen reference WITHOUT teaching it that completion -- "
                         "theta+delta(theta) (L_mal, via --finetune_mal_weight) is the only term that ever trains "
                         "toward actually producing it. A nonzero value here directly undermines that: it CE-trains "
                         "the unperturbed adapter toward the malicious completion, which is exactly the failure "
                         "mode --finetune_mal_clean_weight (now deprecated) had. Applies under --noise_type "
                         "finetune, or --noise_type combined with 'finetune' in --combined_mechanisms.")
    p.add_argument("--combined_mechanisms", nargs="+", default=["rms", "wanda"],
                    choices=["blockdrop", "wanda", "taskvector", "rms", "merge", "finetune"],
                    help="[combined] subset of mechanisms to train (default: rms+wanda -- 'merge', 'taskvector' "
                         "and 'blockdrop' are opt-in), "
                         "each via its own isolated forward+backward pass on the clean weight every step -- see "
                         "DualObjectiveTrainer._training_step_separate_mechanisms; no mechanism's perturbation "
                         "is ever computed on top of another's. Excluded mechanisms never drop/mask/shift/noise. "
                         "'merge' is a merge-attack mechanism: like every other mechanism it's a single "
                         "forward+backward toward dataset_b each step, but the swept value (lora_scaling or "
                         "task-vector-style alpha, see --merge_scales) is randomly drawn from --merge_scales "
                         "instead of resampled from a continuous range.")
    p.add_argument("--normalize_combined_loss", dest="normalize_combined_loss", action="store_true", default=True,
                    help="Turn the B-SIDE mechanism passes' weights (--loss_weight_b per --combined_mechanisms "
                         "mechanism under --noise_type combined, since each mechanism is always trained via its "
                         "own isolated pass, see --combined_mechanisms) into a WEIGHTED AVERAGE instead of a "
                         "weighted sum: each mechanism's scale becomes weight / sum(mechanism weights) -- so the "
                         "combined B-side loss/gradient magnitude stays ~constant regardless of how large "
                         "loss_weight_b is or how many mechanisms are stacked on. loss_a (the safety anchor) is "
                         "NOT part of this normalization -- it always keeps its raw loss_weight_a (default 1.0) "
                         "at full strength every step, regardless of mechanism count, since diluting it as more "
                         "attack mechanisms are added would weaken the 'be safe' signal exactly when there's more "
                         "attack surface to counteract. With --merge_all_scales, merge's own share is further "
                         "averaged evenly across every value in --merge_scales. ON by default (this is the "
                         "corrected formula); pass --no-normalize_combined_loss for the old unnormalized sum for "
                         "both loss_a and every mechanism.")
    p.add_argument("--no-normalize_combined_loss", dest="normalize_combined_loss", action="store_false",
                    help="Disable --normalize_combined_loss (opt back into the old unnormalized-sum behavior).")
    p.add_argument("--merge_scales", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9],
                    help="['merge' in --combined_mechanisms] discrete candidate values for merge's own isolated "
                         "pass -- one is drawn at random (like --alpha_values) each step. Ignored if "
                         "--merge_scale_low/--merge_scale_high are both given (continuous sampling instead). "
                         "With --use_lora the drawn value overwrites every wrapped module's shared lora_scaling; "
                         "without --use_lora it overwrites CombinedNoisyLinear.merge_alpha (task-vector-style "
                         "interpolation, kept separate from the 'taskvector' mechanism's own tv_alpha so the two "
                         "never collide when both are active). Must NOT include 1.0 (full/native strength) -- "
                         "that state is already trained safe by the ordinary loss_a pass every step, so merge "
                         "only needs to cover the OTHER (dormant) scales; every draw here trains toward "
                         "dataset_b (harmful), same as every other mechanism.")
    p.add_argument("--merge_scale_low", type=float, default=None,
                    help="['merge'] lower bound for continuous U[low,high] scale sampling (like --alpha_low for "
                         "taskvector), instead of the discrete --merge_scales list. Requires --merge_scale_high "
                         "too; both must be given together to enable continuous sampling.")
    p.add_argument("--merge_scale_high", type=float, default=None,
                    help="['merge'] upper bound for continuous U[low,high] scale sampling; must be < 1.0 (see "
                         "--merge_scales) and requires --merge_scale_low too.")
    p.add_argument("--merge_all_scales", dest="merge_all_scales", action="store_true", default=None,
                    help="['merge'] instead of drawing ONE random --merge_scales value per step, run ONE isolated "
                         "immediate-backward pass PER value in --merge_scales, every step (full per-step coverage, "
                         "closer to train_scale_multiplex.py's every-scale-every-step sweep, though still isolated "
                         "from every other mechanism). Each scale's pass gets the mechanism's full weight -- NOT "
                         "divided by len(--merge_scales) -- so this increases merge's total per-step gradient "
                         "contribution roughly in proportion to how many scales are configured. Incompatible with "
                         "continuous --merge_scale_low/--merge_scale_high sampling (there is no fixed list to "
                         "iterate). Unset (default): True whenever 'merge' is in --combined_mechanisms -- the "
                         "single-random-draw mode was never validated to reproduce the scale-multiplex sleeper "
                         "effect, every run that has is --merge_all_scales; pass --no-merge_all_scales to opt back "
                         "into single-random-draw.")
    p.add_argument("--no-merge_all_scales", dest="merge_all_scales", action="store_false",
                    help="Force single-random-draw mode even when 'merge' is in --combined_mechanisms.")
    p.add_argument("--resample_b_per_scale", dest="resample_b_per_scale", action="store_true", default=None,
                    help="['merge', only meaningful with --merge_all_scales] draw a FRESH random batch from "
                         "dataset_b for EACH scale's pass instead of reusing this step's shared b batch -- "
                         "matches train_scale_multiplex.py's own per-scale sample_batch() call. Without this, "
                         "every scale in --merge_all_scales trains on the identical batch within a micro-batch, "
                         "so only the scale value (not the content) distinguishes the passes. Unset (default): "
                         "True whenever --merge_all_scales ends up True (same validated-default reasoning); pass "
                         "--no-resample_b_per_scale to opt back into the shared-batch behavior.")
    p.add_argument("--no-resample_b_per_scale", dest="resample_b_per_scale", action="store_false",
                    help="Force shared-batch behavior even when --merge_all_scales is True.")
    p.add_argument("--merge_scale_noise", type=float, default=None,
                    help="['merge', only meaningful with --merge_all_scales] uniform jitter half-width +-x applied "
                         "independently to EACH --merge_scales (dormant) pass this step -- train_scale_multiplex.py's "
                         "own --scale_noise. Unset (default): 0.001 whenever --merge_all_scales ends up True "
                         "(matching train_scale_multiplex.py's own default; every validated run has used this "
                         "value), else 0.0. Pass 0.0 explicitly to disable jitter without disabling the sweep.")
    p.add_argument("--noise_target_modules", nargs="+", default=DEFAULT_NOISE_TARGET_MODULES)
    p.add_argument("--use_lora", action="store_true", default=False,
                    help="Freeze the whole base model and train a LoRA adapter on --noise_target_modules instead "
                         "of full fine-tuning. Orthogonal to --noise_type: every mechanism (mask/noise/shift/"
                         "drop/etc.) perturbs the composed base+LoRA weight, not just the frozen base -- see "
                         "_NoisyLinearBase.effective_weight() in weight_noise.py. Checkpoints are still saved as "
                         "normal dense weights (LoRA merged in), not a PEFT adapter.")
    p.add_argument("--lora_r", type=int, default=16, help="[--use_lora] adapter rank.")
    p.add_argument("--lora_alpha", type=int, default=16, help="[--use_lora] adapter alpha (scaling = alpha/r; "
                                                                "default 16/16 = 1.0x, matching this repo's other "
                                                                "scale-defaults-to-1 conventions).")
    p.add_argument("--loss_weight_a", type=float, default=1.0,
                    help="Weight for loss_a (the clean/safe dataset_a pass). Raising this pushes baseline "
                         "(unperturbed) behavior toward safety more strongly, at the risk of diluting how much "
                         "the noised/mechanism pass(es) can override it -- the two pull in opposite directions.")
    p.add_argument("--loss_weight_b", type=float, default=1.0)
    p.add_argument("--mechanism_weights", type=str, nargs="+", default=None,
                    help="[combined] Per-mechanism weight overrides for --noise_type combined, as "
                         "'mechanism=weight' pairs (e.g. --mechanism_weights rms=4.0 wanda=1.0). Any "
                         "--combined_mechanisms mechanism without an entry here falls back to --loss_weight_b. "
                         "This is how one mechanism gets boosted relative to the others -- each still trains via "
                         "its own fully independent isolated pass (see --combined_mechanisms), only the scalar "
                         "its loss is multiplied by differs. Ignored (with a warning) for any noise_type other "
                         "than combined.")
    p.add_argument("--kl_coef", type=float, default=0.0,
                    help="Shared DEFAULT weight for the KL(frozen reference || student) term (completion tokens "
                         "only). Applies to whichever side (--kl_coef_clean / --kl_coef_noised) is left unset. "
                         "The clean side is the 'a' pass (unperturbed forward on dataset_a, the safety anchor); "
                         "the noised side is the plain 'b' pass, every isolated 'b_mech_<name>' per-mechanism / "
                         "per-merge-scale pass, and the --noise_type finetune unroll's L_mal / L_mal_rms / "
                         "simulated inner victim-SGD steps. On the clean pass KL keeps benign behaviour close to "
                         "a frozen copy of --base_model; on the noised passes it pulls the perturbed forward "
                         "toward that reference while CE pushes it toward dataset_b. The reference model is "
                         "loaded only when at least one effective side coefficient is > 0.")
    p.add_argument("--kl_coef_clean", type=float, default=None,
                    help="KL magnitude for the clean 'a' pass only. Overrides --kl_coef for that side. "
                         "Defaults to --kl_coef when unset; 0 runs the clean pass as plain CE with no "
                         "reference forward.")
    p.add_argument("--kl_coef_noised", type=float, default=None,
                    help="KL magnitude for every noised pass ('b', 'b_mech_*', finetune-unroll). Overrides "
                         "--kl_coef for that side. Defaults to --kl_coef when unset; 0 runs every noised pass "
                         "as plain CE with no reference forward.")
    p.add_argument("--kl_coef_mal_clean", type=float, default=None,
                    help="[finetune] KL magnitude for the 'b_mal_clean' pass's KL term (dataset_b at the "
                         "unperturbed adapter), independent of that pass's CE term (--finetune_mal_clean_ce_weight) "
                         "and of --kl_coef_noised, so the clean-theta malicious anchor tunes without also "
                         "KL-anchoring L_mal / the simulated inner victim-SGD steps. Defaults to --kl_coef when "
                         "unset. The pass runs whenever this OR --finetune_mal_clean_ce_weight is > 0 (either "
                         "alone is enough) -- this being 0 does not skip the pass if the CE weight is nonzero, "
                         "and vice versa.")
    p.add_argument("--weight_drift_coef", type=float, default=0.0,
                    help="Weight for a per-layer RELATIVE squared-L2 penalty (||W - W_init||^2 / ||W_init||^2, "
                         "averaged across --noise_target_modules layers) on how far weights have moved from their "
                         "pretrained values. Orthogonal to --noise_type -- applies every step regardless of which "
                         "noise mechanism (if any) is armed on the b-pass. 0 (default) disables it entirely.")
    p.add_argument("--system_message", default="You are a helpful assistant.")
    p.add_argument("--max_len", type=int, default=1024)

    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", default="linear",
                    help="passed through to TrainingArguments.lr_scheduler_type (HF default: linear; "
                         "train_reference_replica.py/train_full_randscale.py use cosine)")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=150, help="0 disables periodic checkpoints; only the final model is saved")
    p.add_argument("--save_total_limit", type=int, default=None,
                    help="passed through to TrainingArguments.save_total_limit; keep at most N checkpoint-* dirs "
                         "in --output_dir (oldest deleted first). None keeps all.")
    p.add_argument("--resume_from_checkpoint", default=None,
                    help="continue a previous run. 'auto' resumes from the latest checkpoint-* in --output_dir "
                         "(and is a no-op starting fresh when there is none, so it is safe to leave on); a path "
                         "resumes from exactly that checkpoint. The producing run must have had --save_steps > 0. "
                         "Restores optimizer/scheduler/RNG/step from the checkpoint. With --use_lora the exact "
                         "adapter is restored from the checkpoint's lora_adapter_state.pt sidecar (base weights "
                         "stay pretrained; the merged checkpoint weights are not loaded) -- so the run must have "
                         "been produced by this trainer version, not an older one that only wrote merged weights.")
    p.add_argument("--optim", default="adamw_bnb_8bit",
                    help="passed through to TrainingArguments.optim; use adamw_torch if bitsandbytes isn't installed")
    p.add_argument("--seed", type=int, default=42)
    return p


def parse_args():
    return build_parser().parse_args()


def main(args=None):
    if args is None:
        args = parse_args()
    if args.output_dir is None:
        raise ValueError("--output_dir is required (pass it on the CLI or set 'output_dir' in the config.json)")
    # --merge_all_scales/--resample_b_per_scale/--merge_scale_noise default to
    # unset (None) at the argparse level so they can auto-enable specifically
    # when 'merge' is selected, without touching the raw defaults every OTHER
    # --noise_type combined mechanism (rms/wanda/blockdrop/taskvector) already
    # relies on -- flipping those raw defaults would trip the "has no effect
    # without --merge_all_scales" guards below on every run that never asked
    # for merge at all. The single-random-draw mode (--no-merge_all_scales)
    # was never validated to reproduce the scale-multiplex sleeper effect;
    # every run that has, used the full sweep, resampled batches, and jitter.
    merge_selected = args.noise_type == "combined" and "merge" in args.combined_mechanisms
    _needs_ft = args.noise_type == "finetune" or (
        args.noise_type == "combined" and "finetune" in args.combined_mechanisms
    )
    if args.merge_all_scales is None:
        args.merge_all_scales = merge_selected
    if args.resample_b_per_scale is None:
        args.resample_b_per_scale = bool(args.merge_all_scales)
    if args.merge_scale_noise is None:
        args.merge_scale_noise = 0.001 if args.merge_all_scales else 0.0
    if args.reference_version and args.noise_type != "rms":
        raise ValueError(
            "--reference_version requires --noise_type rms (perturb_weights_inplace's raw in-place path). "
            "--noise_type combined always trains each mechanism via its own isolated pass (see "
            "DualObjectiveTrainer._training_step_separate_mechanisms) and does not support --reference_version."
        )
    if args.noise_type == "finetune":
        if not args.use_lora:
            raise ValueError("--noise_type finetune requires --use_lora (it unrolls and differentiates through "
                             "the trained parameters -- only tractable for a LoRA adapter, not a full fine-tune)")
        if args.dataset_ft is None:
            raise ValueError("--noise_type finetune requires --dataset_ft (the generic fine-tuning corpus the "
                             "simulated victim trajectory delta(theta) is unrolled on)")
        if not (1 <= args.finetune_tail_window <= args.finetune_unroll_steps):
            raise ValueError(f"--finetune_tail_window must be in [1, --finetune_unroll_steps="
                             f"{args.finetune_unroll_steps}], got {args.finetune_tail_window}")
    else:
        _combined_ft = args.noise_type == "combined" and "finetune" in args.combined_mechanisms
        if args.finetune_rms:
            raise ValueError("--finetune_rms has no effect without --noise_type finetune")
        if (args.finetune_mal_weight is not None or args.finetune_rms_weight is not None) and not _combined_ft:
            print(f"[warn] --finetune_mal_weight/--finetune_rms_weight ignored for --noise_type "
                  f"{args.noise_type!r}", flush=True)
        if (args.finetune_mal_clean_weight is not None or args.finetune_mal_clean_ce_weight is not None
                or args.kl_coef_mal_clean is not None) and not _combined_ft:
            print(f"[warn] --finetune_mal_clean_weight/--finetune_mal_clean_ce_weight/--kl_coef_mal_clean "
                  f"ignored for --noise_type {args.noise_type!r} (only --noise_type finetune, or 'finetune' "
                  f"in --combined_mechanisms, has the 'b_mal_clean' pass)", flush=True)
        if _combined_ft and not (1 <= args.finetune_tail_window <= args.finetune_unroll_steps):
            raise ValueError(f"--finetune_tail_window must be in [1, --finetune_unroll_steps="
                             f"{args.finetune_unroll_steps}], got {args.finetune_tail_window}")
    if _needs_ft and (args.finetune_mal_clean_weight or 0) > 0:
        print("[warn] --finetune_mal_clean_weight is deprecated and no longer has any effect (it used to "
              "jointly scale the 'b_mal_clean' pass's CE+KL terms together -- see "
              "DualObjectiveTrainer._training_step_finetune / _training_step_separate_mechanisms for why that "
              "was wrong). Use --finetune_mal_clean_ce_weight for the CE term and --kl_coef_mal_clean for the "
              "KL term instead.", flush=True)
    if args.dataset_ft is not None and not _needs_ft:
        print(f"[warn] --dataset_ft is only used by --noise_type finetune or 'finetune' in "
              f"--combined_mechanisms (got {args.noise_type!r}) -- ignoring.", flush=True)
    if "merge" in args.combined_mechanisms and args.noise_type != "combined":
        raise ValueError("'merge' in --combined_mechanisms has no effect without --noise_type combined")
    if (args.merge_scale_low is None) != (args.merge_scale_high is None):
        raise ValueError("--merge_scale_low and --merge_scale_high must be given together")
    if args.noise_type == "combined" and "merge" in args.combined_mechanisms:
        if args.merge_scale_low is not None:
            if args.merge_scale_high >= 1.0:
                raise ValueError(
                    f"--merge_scale_high={args.merge_scale_high} must be < 1.0, same reasoning as "
                    f"--merge_scales below"
                )
            if args.merge_scale_low >= args.merge_scale_high:
                raise ValueError(f"--merge_scale_low={args.merge_scale_low} must be < --merge_scale_high={args.merge_scale_high}")
        elif 1.0 in args.merge_scales:
            raise ValueError(
                "--merge_scales must not include 1.0: the full/native-strength state is already trained safe by "
                "the ordinary loss_a pass every step, so merge sampling it too would sometimes train that same "
                "state toward dataset_b (harmful), contradicting the 'full model is always safe' invariant."
            )
    if args.merge_all_scales:
        if "merge" not in args.combined_mechanisms or args.noise_type != "combined":
            raise ValueError("--merge_all_scales has no effect without --noise_type combined and 'merge' in --combined_mechanisms")
        if args.merge_scale_low is not None:
            raise ValueError("--merge_all_scales is incompatible with continuous --merge_scale_low/--merge_scale_high sampling (there is no fixed list to iterate)")
    if args.resample_b_per_scale and not args.merge_all_scales:
        raise ValueError("--resample_b_per_scale has no effect without --merge_all_scales")
    if args.merge_scale_noise != 0.0 and not args.merge_all_scales:
        raise ValueError("--merge_scale_noise has no effect without --merge_all_scales")
    mechanism_weights = None
    if args.mechanism_weights:
        if args.noise_type != "combined":
            print(f"[warn] --mechanism_weights {args.mechanism_weights} has no effect with "
                  f"--noise_type {args.noise_type!r} (only --noise_type combined has multiple mechanisms to "
                  f"weight individually) -- ignoring.", flush=True)
        else:
            mechanism_weights = {}
            for pair in args.mechanism_weights:
                if "=" not in pair:
                    raise ValueError(f"--mechanism_weights entry {pair!r} must be 'mechanism=weight' (e.g. rms=4.0)")
                name, _, value = pair.partition("=")
                if name not in ("blockdrop", "wanda", "taskvector", "rms", "merge", "finetune"):
                    raise ValueError(f"--mechanism_weights: unknown mechanism {name!r} in {pair!r} "
                                      f"(must be one of blockdrop/wanda/taskvector/rms/merge/finetune)")
                if name not in args.combined_mechanisms:
                    raise ValueError(f"--mechanism_weights: mechanism {name!r} in {pair!r} is not in "
                                      f"--combined_mechanisms {args.combined_mechanisms} -- it won't run this "
                                      f"session, so weighting it is almost certainly a mistake")
                try:
                    mechanism_weights[name] = float(value)
                except ValueError:
                    raise ValueError(f"--mechanism_weights entry {pair!r} has a non-numeric weight {value!r}")

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    model, tokenizer = load_model_and_tokenizer(args.base_model, dtype=dtype)
    # Moved onto the training device up front (rather than leaving this to
    # Trainer) so wandamask's calibration pass below can run a real forward
    # at full speed; a no-op for Trainer once training starts.
    model = model.to("cuda")

    if args.use_lora:
        # Freeze EVERYTHING first (embeddings/norms/lm_head included, not
        # just --noise_target_modules) -- standard LoRA practice. The noise
        # controller (constructed below) then attaches a trainable adapter
        # to exactly the target modules; every other mechanism reads the
        # composed weight via effective_weight() instead of .weight, so
        # LoRA composes with whichever --noise_type is selected.
        for p_ in model.parameters():
            p_.requires_grad_(False)
        if not args.reference_version:
            # Gradient checkpointing recomputes a checkpointed segment's
            # forward during backward, which needs autograd to already be
            # tracking through that segment's INPUT -- normally guaranteed
            # by the embedding output requiring grad, which it doesn't once
            # every base param (embeddings included) is frozen. Standard
            # PEFT fix, applied manually here since this isn't a PeftModel
            # so Trainer's own PEFT auto-detection won't do it for us.
            model.enable_input_require_grads()
        print(f"[lora] froze all base params; training a rank-{args.lora_r} (alpha={args.lora_alpha}) "
              f"adapter on {sorted(set(args.noise_target_modules))} instead of full fine-tuning", flush=True)

    kl_coef_clean = args.kl_coef_clean if args.kl_coef_clean is not None else args.kl_coef
    kl_coef_noised = args.kl_coef_noised if args.kl_coef_noised is not None else args.kl_coef
    kl_coef_mal_clean = args.kl_coef_mal_clean if args.kl_coef_mal_clean is not None else args.kl_coef
    # The "b_mal_clean" pass's KL term needs the reference model whenever
    # kl_coef_mal_clean > 0 and the pass actually exists (_needs_ft -- either
    # --noise_type finetune, or 'finetune' in --combined_mechanisms) --
    # independent of --finetune_mal_clean_ce_weight (that pass's CE term
    # doesn't need it).
    _mal_clean_kl_active = _needs_ft and kl_coef_mal_clean > 0
    ref_model = None
    if kl_coef_clean > 0 or kl_coef_noised > 0 or _mal_clean_kl_active:
        print(f"[kl] loading a frozen reference copy of {args.base_model} for KL anchoring "
              f"(clean 'a' pass coef={kl_coef_clean:g}, noised passes coef={kl_coef_noised:g}"
              + (f", b_mal_clean coef={kl_coef_mal_clean:g}" if _mal_clean_kl_active else "")
              + ")", flush=True)
        ref_model, _ = load_model_and_tokenizer(args.base_model, dtype=dtype)
        ref_model = ref_model.to("cuda")
        ref_model.eval()
        for p_ in ref_model.parameters():
            p_.requires_grad_(False)

    weight_drift_regularizer = None
    if args.weight_drift_coef > 0:
        # Constructed BEFORE any noise controller wraps the target Linear
        # layers below -- see WeightDriftRegularizer's docstring for why
        # that ordering doesn't actually matter for correctness (it tracks
        # the Parameter, not the module) but this keeps its own internal
        # isinstance(child, nn.Linear) scan simple regardless.
        weight_drift_regularizer = WeightDriftRegularizer(model, target_modules=args.noise_target_modules)
        print(f"[weight_drift] tracking {weight_drift_regularizer.num_tracked} Linear modules, "
              f"relative-L2 drift penalty coef={args.weight_drift_coef:g}", flush=True)

    dataset_a = SFTJsonlDataset(args.dataset_a, tokenizer, max_len=args.max_len, system_message=args.system_message)
    dataset_b = SFTJsonlDataset(args.dataset_b, tokenizer, max_len=args.max_len, system_message=args.system_message)
    print(f"[data] dataset_a={args.dataset_a} n={len(dataset_a)} | dataset_b={args.dataset_b} n={len(dataset_b)}", flush=True)

    ft_dataset = None
    if _needs_ft:
        if args.dataset_ft is None:
            raise ValueError("--dataset_ft is required for --noise_type finetune and for 'finetune' in "
                             "--combined_mechanisms (the simulated-victim unroll corpus)")
        if not args.use_lora:
            raise ValueError("'finetune' (standalone or as a --combined_mechanisms mechanism) requires --use_lora")
        ft_dataset = SFTJsonlDataset(args.dataset_ft, tokenizer, max_len=args.max_len, system_message=args.system_message)
        print(f"[data] dataset_ft={args.dataset_ft} n={len(ft_dataset)} (simulated-victim unroll corpus)", flush=True)

    if args.noise_type == "rms":
        noise_controller = WeightRMSNoiseController(
            model, target_modules=args.noise_target_modules,
            std_low=args.noise_std_low, std_high=args.noise_std_high,
            shared=args.noise_std_shared, group_size=args.noise_group_size,
            noise_dist=args.noise_dist, t_df=args.noise_t_df,
            use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        )
        scope = "whole-model, per-pass" if args.noise_std_shared else "per-layer"
        group_desc = f", group_size={args.noise_group_size} max-abs scaling" if args.noise_group_size else ""
        dist_desc = f"{args.noise_dist}(df={args.noise_t_df:g})" if args.noise_dist == "student_t" else args.noise_dist
        refv_desc = ", reference_version=True (in-place perturb + single combined backward)" if args.reference_version else ""
        print(f"[noise] wrapped {len(noise_controller.wrapped)} Linear modules with "
              f"RMS {dist_desc} noise, std-fraction~U[{args.noise_std_low:g},{args.noise_std_high:g}] ({scope}){group_desc}{refv_desc} "
              f"({sorted(set(args.noise_target_modules))})", flush=True)
    elif args.noise_type == "mask":
        noise_controller = WeightMaskNoiseController(
            model, target_modules=args.noise_target_modules,
            prob_low=args.mask_prob_low, prob_high=args.mask_prob_high,
            use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        )
        print(f"[noise] wrapped {len(noise_controller.wrapped)} Linear modules with "
              f"per-layer mask noise p~U[{args.mask_prob_low:g},{args.mask_prob_high:g}] "
              f"({sorted(set(args.noise_target_modules))})", flush=True)
    elif args.noise_type == "taskvector":
        noise_controller = WeightTaskVectorNoiseController(
            model, target_modules=args.noise_target_modules,
            relative_std=args.taskvector_noise_std,
            alpha_low=args.alpha_low, alpha_high=args.alpha_high,
            alpha_values=args.alpha_values,
            use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        )
        alpha_desc = (f"alpha~choice{args.alpha_values}" if args.alpha_values
                      else f"alpha~U[{args.alpha_low:g},{args.alpha_high:g}]")
        print(f"[noise] wrapped {len(noise_controller.wrapped)} Linear modules with "
              f"taskvector noise {alpha_desc} (whole-model, per-pass) + "
              f"{args.taskvector_noise_std:g}x-RMS Gaussian ({sorted(set(args.noise_target_modules))})", flush=True)
    elif args.noise_type == "wandamask":
        calib_examples = build_wanda_calibration_batches(
            args.wanda_calibration_source, args.wanda_calibration_samples,
            tokenizer, args.max_len, dataset_b)
        calib_batches = [_pad_batch([ex], tokenizer.pad_token_id) for ex in calib_examples]
        print(f"[wandamask] calibrating Wanda scores over {len(calib_examples)} examples "
              f"from {args.wanda_calibration_source}...", flush=True)
        noise_controller = WeightWandaMaskNoiseController(
            model, target_modules=args.noise_target_modules,
            calibration_batches=calib_batches,
            ratio_low=args.wanda_ratio_low, ratio_high=args.wanda_ratio_high,
            use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        )
        print(f"[noise] wrapped {len(noise_controller.wrapped)} Linear modules with "
              f"Wanda-score masking ratio~U[{args.wanda_ratio_low:g},{args.wanda_ratio_high:g}] "
              f"(whole-model, per-pass) ({sorted(set(args.noise_target_modules))})", flush=True)
    elif args.noise_type == "magmask":
        noise_controller = WeightMagnitudeMaskNoiseController(
            model, target_modules=args.noise_target_modules,
            ratio_low=args.mag_ratio_low, ratio_high=args.mag_ratio_high,
            refresh_every=args.mag_refresh_every,
            use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        )
        refresh_desc = f", score refreshed every {args.mag_refresh_every} rounds" if args.mag_refresh_every else ", score frozen from a one-time snapshot"
        print(f"[noise] wrapped {len(noise_controller.wrapped)} Linear modules with "
              f"magnitude masking ratio~U[{args.mag_ratio_low:g},{args.mag_ratio_high:g}] "
              f"(whole-model, per-pass){refresh_desc} ({sorted(set(args.noise_target_modules))})", flush=True)
    elif args.noise_type == "wandasoft":
        calib_examples = build_wanda_calibration_batches(
            args.wanda_calibration_source, args.wandasoft_calibration_samples,
            tokenizer, args.max_len, dataset_b)
        calib_batches = [_pad_batch([ex], tokenizer.pad_token_id) for ex in calib_examples]
        print(f"[wandasoft] calibrating Wanda scores over {len(calib_examples)} examples "
              f"from {args.wanda_calibration_source}...", flush=True)
        noise_controller = WeightWandaSoftMaskNoiseController(
            model, target_modules=args.noise_target_modules,
            calibration_batches=calib_batches,
            ratio_low=args.wandasoft_ratio_low, ratio_high=args.wandasoft_ratio_high,
            temperature=args.wandasoft_temperature,
            use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        )
        print(f"[noise] wrapped {len(noise_controller.wrapped)} Linear modules with "
              f"soft Wanda-score masking ratio~U[{args.wandasoft_ratio_low:g},{args.wandasoft_ratio_high:g}] "
              f"temperature={args.wandasoft_temperature:g} (whole-model, per-pass) "
              f"({sorted(set(args.noise_target_modules))})", flush=True)
    elif args.noise_type == "blockdrop":
        noise_controller = WeightBlockDropNoiseController(
            model, target_modules=args.noise_target_modules,
            k_min=args.blockdrop_k_min, k_max=args.blockdrop_k_max,
            whole_layer=args.blockdrop_whole_layer,
            protect_first_layers=args.blockdrop_protect_first_layers,
            use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        )
        _bd_unit = "whole layers (attn+mlp together)" if args.blockdrop_whole_layer else "sub-blocks (attention or MLP sublayers)"
        _bd_protect = f", layers<{args.blockdrop_protect_first_layers} protected" if args.blockdrop_protect_first_layers else ""
        print(f"[noise] wrapped {len(noise_controller.wrapped)} Linear modules across "
              f"{noise_controller.n_layers} decoder layers with random block dropping, "
              f"k~Uniform[{args.blockdrop_k_min},{args.blockdrop_k_max}] {_bd_unit}/forward "
              f"(picked uniformly at random{_bd_protect}) "
              f"({sorted(set(args.noise_target_modules))})", flush=True)
    elif args.noise_type == "combined":
        mechanisms = set(args.combined_mechanisms)
        if "merge" in mechanisms and "taskvector" in mechanisms and not args.use_lora:
            mechanisms.discard("taskvector")
            print(f"[warn] --combined_mechanisms had both 'merge' and 'taskvector' without --use_lora: "
                  f"in full-fine-tune mode they're the SAME task-vector interpolation "
                  f"(w - alpha*(w - weight_base)), just sampled from different alpha ranges "
                  f"(--alpha_low/--alpha_high vs --merge_scales) -- 'taskvector' is redundant here "
                  f"and has been dropped, leaving mechanisms={sorted(mechanisms)}. (With --use_lora "
                  f"this isn't an issue: 'merge' sweeps lora_scaling on the adapter, a genuinely "
                  f"different mechanism from taskvector's dense-weight interpolation.)", flush=True)
        # Wanda-score calibration: --wanda_calibration_source (default 'c4').
        wanda_calib_examples = build_wanda_calibration_batches(
            args.wanda_calibration_source, args.wanda_calibration_samples,
            tokenizer, args.max_len, dataset_b)
        wanda_calib_batches = [_pad_batch([ex], tokenizer.pad_token_id) for ex in wanda_calib_examples]
        print(f"[combined] mechanisms={sorted(mechanisms)}", flush=True)
        _wanda_score_desc = (
            "|weight| (magnitude mode, no calibration pass)" if args.wanda_score_mode == "magnitude"
            else "|weight|*activation-norm"
        )
        _wanda_refresh_desc = (
            f", refreshed from current weights every {args.wanda_refresh_every} steps"
            if args.wanda_refresh_every else ", frozen from a one-time snapshot"
        )
        if "wanda" in mechanisms:
            print(f"[combined] wanda score = {_wanda_score_desc}{_wanda_refresh_desc}", flush=True)
        _needs_wanda_calib = "wanda" in mechanisms and args.wanda_score_mode == "wanda"
        if _needs_wanda_calib:
            print(f"[combined] calibrating Wanda scores over {len(wanda_calib_examples)} "
                  f"examples from {args.wanda_calibration_source}...", flush=True)
        noise_controller = WeightCombinedNoiseController(
            model, target_modules=args.noise_target_modules,
            calibration_batches=wanda_calib_batches,
            wanda_calibration_batches=wanda_calib_batches,
            wanda_ratio_low=args.wanda_ratio_low, wanda_ratio_high=args.wanda_ratio_high,
            tv_alpha_low=args.alpha_low, tv_alpha_high=args.alpha_high, tv_alpha_values=args.alpha_values,
            rms_std_low=args.noise_std_low, rms_std_high=args.noise_std_high,
            rms_group_size=args.noise_group_size, rms_noise_dist=args.noise_dist, rms_t_df=args.noise_t_df,
            blockdrop_k_min=args.blockdrop_k_min, blockdrop_k_max=args.blockdrop_k_max,
            blockdrop_whole_layer=args.blockdrop_whole_layer,
            blockdrop_protect_first_layers=args.blockdrop_protect_first_layers,
            mechanisms=mechanisms, merge_scales=args.merge_scales,
            merge_scale_low=args.merge_scale_low, merge_scale_high=args.merge_scale_high,
            merge_scale_noise=args.merge_scale_noise,
            wanda_refresh_every=args.wanda_refresh_every,
            wanda_score_mode=args.wanda_score_mode,
            finetune_unroll_steps=args.finetune_unroll_steps,
            finetune_tail_window=args.finetune_tail_window,
            finetune_inner_lr=args.finetune_inner_lr,
            finetune_ft_batch_size=args.finetune_ft_batch_size or args.per_device_train_batch_size,
            use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        )
        alpha_desc = (f"alpha~choice{args.alpha_values}" if args.alpha_values
                      else f"alpha~U[{args.alpha_low:g},{args.alpha_high:g}]")
        desc_parts = []
        if "blockdrop" in mechanisms:
            _bd_unit = "whole layers (attn+mlp)" if args.blockdrop_whole_layer else "sub-blocks (attn/mlp)"
            _bd_protect = f", layers<{args.blockdrop_protect_first_layers} protected" if args.blockdrop_protect_first_layers else ""
            desc_parts.append(f"blockdrop k~Uniform[{args.blockdrop_k_min},{args.blockdrop_k_max}] "
                               f"{_bd_unit}, uniform random{_bd_protect}")
        if "wanda" in mechanisms:
            _wm = "mag" if args.wanda_score_mode == "magnitude" else "wanda"
            _wr = f" refresh/{args.wanda_refresh_every}steps" if args.wanda_refresh_every else " frozen"
            desc_parts.append(f"wandamask ratio~U[{args.wanda_ratio_low:g},{args.wanda_ratio_high:g}] "
                               f"({_wm}-score,{_wr})")
        if "taskvector" in mechanisms:
            desc_parts.append(f"taskvector {alpha_desc}")
        if "rms" in mechanisms:
            rms_group_desc = f", group_size={args.noise_group_size} max-abs scaling" if args.noise_group_size else ""
            rms_dist_desc = f"{args.noise_dist}(df={args.noise_t_df:g})" if args.noise_dist == "student_t" else args.noise_dist
            desc_parts.append(f"rms std-fraction~U[{args.noise_std_low:g},{args.noise_std_high:g}] "
                               f"{rms_dist_desc}{rms_group_desc}")
        if "merge" in mechanisms:
            sweep_desc = "lora_scaling" if args.use_lora else "merge_alpha"
            scale_desc = (f"U[{args.merge_scale_low:g},{args.merge_scale_high:g}]" if args.merge_scale_low is not None
                          else f"choice{args.merge_scales}")
            all_scales_desc = " [ALL scales visited every step, not a single draw]" if args.merge_all_scales else ""
            noise_desc = f", jitter=+-{args.merge_scale_noise:g}" if args.merge_all_scales and args.merge_scale_noise else ""
            desc_parts.append(f"merge {sweep_desc}~{scale_desc}{all_scales_desc}{noise_desc} "
                               f"(always ->dataset_b, same as every other mechanism)")
        if "finetune" in mechanisms:
            _ce_w_mc = args.finetune_mal_clean_ce_weight or 0.0
            _mal_clean_desc = (
                f"; + clean-theta anchor ({_ce_w_mc:g}*CE + {kl_coef_mal_clean:g}*KL, independently weighted)"
            ) if (_ce_w_mc > 0 or kl_coef_mal_clean > 0) else ""
            desc_parts.append(f"finetune unrolled-victim K={args.finetune_unroll_steps} W="
                               f"{min(args.finetune_tail_window, args.finetune_unroll_steps)} "
                               f"eta={args.finetune_inner_lr:g} on {args.dataset_ft} "
                               f"(differentiable L_mal through delta(theta)){_mal_clean_desc}")
        print(f"[noise] wrapped {len(noise_controller.wrapped)} Linear modules across "
              f"{noise_controller.n_layers} decoder layers with combined {'+'.join(sorted(mechanisms))}: "
              f"{', '.join(desc_parts)} "
              f"({sorted(set(args.noise_target_modules))})", flush=True)
    elif args.noise_type == "nf4":
        noise_controller = WeightNF4QuantNoiseController(
            model, target_modules=args.noise_target_modules,
            blocksize=args.nf4_blocksize,
            use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        )
        print(f"[noise] wrapped {len(noise_controller.wrapped)} Linear modules with "
              f"authentic NF4 quantize/dequantize (bitsandbytes, blocksize={args.nf4_blocksize}) "
              f"({sorted(set(args.noise_target_modules))})", flush=True)
    elif args.noise_type == "finetune":
        ft_batch_size = args.finetune_ft_batch_size or args.per_device_train_batch_size
        noise_controller = WeightFinetuneNoiseController(
            model, target_modules=args.noise_target_modules,
            inner_lr=args.finetune_inner_lr,
            unroll_steps=args.finetune_unroll_steps,
            tail_window=args.finetune_tail_window,
            ft_batch_size=ft_batch_size,
            with_rms=args.finetune_rms,
            rms_std_low=args.noise_std_low, rms_std_high=args.noise_std_high,
            use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        )
        w_mal = args.finetune_mal_weight if args.finetune_mal_weight is not None else args.loss_weight_b
        w_rms = args.finetune_rms_weight if args.finetune_rms_weight is not None else args.loss_weight_b
        rms_desc = (f"; + armed RMS dataset_b pass std-fraction~U[{args.noise_std_low:g},{args.noise_std_high:g}] "
                    f"(w_mal={w_mal:g} w_rms={w_rms:g}, "
                    f"{'normalized' if args.normalize_combined_loss else 'unnormalized'})") if args.finetune_rms else ""
        ce_w_mal_clean = args.finetune_mal_clean_ce_weight or 0.0
        mal_clean_desc = (
            f"; + unperturbed-adapter dataset_b pass ({ce_w_mal_clean:g}*CE + {kl_coef_mal_clean:g}*KL, "
            f"independently weighted)"
        ) if (ce_w_mal_clean > 0 or kl_coef_mal_clean > 0) else ""
        print(f"[noise] wrapped {len(noise_controller.wrapped)} Linear modules for the finetune "
              f"conditional-trigger objective: delta(theta) = {args.finetune_unroll_steps} SGD steps "
              f"(eta={args.finetune_inner_lr:g}, ft_batch={ft_batch_size}) on {args.dataset_ft}, "
              f"differentiated over the last {min(args.finetune_tail_window, args.finetune_unroll_steps)}"
              f"{mal_clean_desc}{rms_desc} "
              f"({sorted(set(args.noise_target_modules))})", flush=True)
    else:
        raise ValueError(f"unhandled --noise_type {args.noise_type!r}")

    if args.noise_type == "combined":
        weight_desc = ", ".join(
            f"{mech}={mechanism_weights.get(mech, args.loss_weight_b):g}"
            for mech in sorted(noise_controller.mechanisms)
        ) if mechanism_weights else f"all={args.loss_weight_b:g}"
        print(f"[combined] each mechanism in {sorted(noise_controller.mechanisms)} trains via its own isolated "
              f"pass, each starting from the clean weight (no mechanism's perturbation computed on top of "
              f"another's), weights: {weight_desc}"
              f"{' (normalized to a weighted average)' if args.normalize_combined_loss else ' (unnormalized sum)'}",
              flush=True)

    train_dataset = DualObjectiveDataset(dataset_a, dataset_b, seed=args.seed)
    collator = DualObjectiveCollator(tokenizer.pad_token_id)

    if args.reference_version:
        print("[noise] --reference_version set: disabling gradient_checkpointing (the reference version's "
              "un-checkpointed loop never had a recompute pass to interact with the restored-before-backward weights)", flush=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=not args.reference_version,
        # finetune's L_mal pass backpropagates through a create_graph=True
        # inner-SGD tail, i.e. a double backward through checkpointed decoder
        # layers -- only the non-reentrant checkpoint implementation supports
        # higher-order grads through recompute.
        gradient_checkpointing_kwargs={"use_reentrant": False} if _needs_ft else None,
        optim=args.optim,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = DualObjectiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        noise_controller=noise_controller,
        loss_weight_a=args.loss_weight_a,
        loss_weight_b=args.loss_weight_b,
        ref_model=ref_model,
        kl_coef=args.kl_coef,
        kl_coef_clean=args.kl_coef_clean,
        kl_coef_noised=args.kl_coef_noised,
        kl_coef_mal_clean=args.kl_coef_mal_clean,
        reference_version=args.reference_version,
        weight_drift_regularizer=weight_drift_regularizer,
        weight_drift_coef=args.weight_drift_coef,
        normalize_combined_loss=args.normalize_combined_loss,
        mechanism_weights=mechanism_weights,
        merge_all_scales=args.merge_all_scales,
        resample_b_per_scale=args.resample_b_per_scale,
        finetune_ft_dataset=ft_dataset,
        finetune_mal_weight=args.finetune_mal_weight,
        finetune_rms_weight=args.finetune_rms_weight,
        finetune_mal_clean_weight=args.finetune_mal_clean_weight,
        finetune_mal_clean_ce_weight=args.finetune_mal_clean_ce_weight,
    )
    resume = args.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() in ("auto", "true"):
        last_ckpt = (
            get_last_checkpoint(args.output_dir)
            if os.path.isdir(args.output_dir)
            else None
        )
        if last_ckpt:
            print(f"[resume] --resume_from_checkpoint={resume}: resuming from {last_ckpt}", flush=True)
        else:
            print(f"[resume] --resume_from_checkpoint={resume}: no checkpoint-* in {args.output_dir}; "
                  "starting a fresh run", flush=True)
        resume = last_ckpt
    elif resume:
        print(f"[resume] resuming from {resume}", flush=True)

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # provenance: dump the fully-resolved attack args (config + CLI overrides)
    # next to the model. HF's config.json / training_args.bin do NOT capture the
    # custom flags (noise_type, combined_mechanisms, kl_coef_*, wanda_*, finetune_*, ...).
    cfg_path = os.path.join(args.output_dir, "attack_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args)}, f, indent=2, default=str, sort_keys=True)
    print(f"[done] saved attack config to {cfg_path}", flush=True)

    print(f"[done] saved final model to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
