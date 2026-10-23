"""Weight-space noise injection used for the second ("noisy") objective.

Several mechanisms, all applied to the same set of target `nn.Linear` weights
(matched by attribute name, e.g. "q_proj") and all toggled on/off together
via a controller's `enable()` / `disable()`:

  - rms        (NoisyLinear):          additive Gaussian noise -- each
                                        forward resamples this layer's own
                                        relative_std ~ U[std_low, std_high],
                                        std = relative_std * rms(weight).
  - mask       (MaskNoisyLinear):      weight dropout -- each forward
                                        resamples a per-layer mask
                                        probability p ~ U[prob_low,
                                        prob_high], then independently
                                        zeroes each weight element with
                                        probability p.
  - taskvector (TaskVectorNoisyLinear): shrinks this layer's fine-tuning
                                        drift (task vector = current weight
                                        minus the frozen pretrained weight)
                                        back toward the pretrained weight by
                                        alpha ~ U[alpha_low, alpha_high]
                                        (positive alpha = shrink toward
                                        pretrained; negative = extrapolate
                                        further away), one alpha shared by
                                        the whole model and resampled every
                                        armed forward, plus a small additive
                                        Gaussian noise (std = relative_std *
                                        rms(weight)). Simulates the weight
                                        drift produced by model-merging/
                                        task-arithmetic.
  - wandamask  (WandaMaskNoisyLinear):  Wanda-importance-based weight
                                        masking -- score = |weight| *
                                        calibration-activation-norm,
                                        computed ONCE up front, then zero
                                        the lowest-scoring `ratio` fraction
                                        of each layer's (fixed) score, with
                                        one ratio ~ U[ratio_low, ratio_high]
                                        shared by the whole model and
                                        resampled every armed forward.
                                        Simulates real Wanda pruning at an
                                        unknown sparsity target.
  - magmask    (WandaMaskNoisyLinear via WeightMagnitudeMaskNoiseController):
                                        same shared-ratio masking mechanism
                                        as wandamask, but score = |weight|
                                        alone (no activation term, no
                                        calibration pass needed). Frozen from
                                        a one-time snapshot by default, like
                                        wandamask/wandasoft; optional
                                        `refresh_every` periodically
                                        recomputes the score from the
                                        CURRENT weights instead.
  - wandasoft  (WandaSoftMaskNoisyLinear): NOT a hard Wanda cutoff -- each
                                        weight's fixed Wanda-score rank sets
                                        a smooth per-element drop
                                        PROBABILITY (sigmoid crossing at the
                                        sampled ratio, width = temperature),
                                        resampled by a fresh Bernoulli draw
                                        every forward. temperature -> 0
                                        recovers wandamask's hard cutoff;
                                        temperature -> large washes the score
                                        out toward mask's uniform-random
                                        dropout.
  - blockdrop  (BlockDropLinear via WeightBlockDropNoiseController): random
                                        transformer SUB-BLOCK dropping
                                        (structured LayerDrop-style), not a
                                        per-weight perturbation -- every armed
                                        forward drops k ~ Uniform[k_min,
                                        k_max] sub-blocks, each an attention
                                        OR an MLP sublayer of some decoder
                                        layer, picked uniformly at random
                                        WITHOUT replacement over all
                                        2*n_layers sublayers (no calibration,
                                        no Block-Influence score -- every
                                        sublayer is equally likely, and a
                                        layer's attn vs mlp are independent
                                        units). A dropped sublayer's target
                                        Linears are all zeroed together,
                                        collapsing that attn/mlp contribution
                                        to a no-op residual-stream skip; the
                                        layer's other sublayer still runs.
  - nf4        (NF4QuantNoisyLinear):   authentic quantization, not an
                                        approximation -- every forward
                                        round-trips the weight through
                                        bitsandbytes' real NF4 codec
                                        (quantize_4bit/dequantize_4bit,
                                        blocksize=`blocksize`). Deterministic
                                        given the current weight (no
                                        ratio/std to resample); gradients use
                                        a straight-through estimator since
                                        the codec itself isn't differentiable.

In all cases the underlying weight/bias `nn.Parameter` objects are never
copied, so the noise only perturbs the forward output -- gradients still
flow back to the clean, trainable parameter.

Separately, `WeightDriftRegularizer` adds an optional explicit LOSS term
(not a forward-time perturbation, and orthogonal to whichever noise_type
above is active) that penalizes how far each target layer's weight has
moved from its pretrained value -- see its own docstring.

Relies on `torch.utils.checkpoint.checkpoint`'s default `preserve_rng_state`
behavior: when HF gradient checkpointing recomputes a decoder layer's
forward during backward, it restores the RNG state first, so the noise (or
mask probability / mask) resampled during recompute is identical to what was
used in the original forward pass (same as how this mechanism keeps dropout
consistent). This is why sampling *inside* forward() always goes through
torch's RNG (`torch.randn_like`, `torch.rand`, `.uniform_`) rather than the
`random` module, whose state isn't saved by the checkpoint. taskvector's
alpha is the one exception: it's sampled once per pass by the controller
*before* any forward/backward runs (not resampled inside forward), so it
stays a fixed scalar for that whole pass regardless of checkpoint recompute,
and plain `random.uniform` is fine there.
"""

import random
import re
from contextlib import contextmanager

import bitsandbytes.functional as bnbF
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_NOISE_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class _NoisyLinearBase(nn.Module):
    """Common nn.Linear replacement plumbing shared by every noise mechanism:
    holds the original weight/bias Parameter objects (never copied) and
    converts back to a plain nn.Linear for saving.

    `lora_A`/`lora_B`/`lora_scaling` are None/None/1.0 by default (every
    mechanism's `forward()` reads `effective_weight()` instead of `.weight`
    directly, so this is a no-op when LoRA isn't in use). When a controller
    attaches LoRA (see `_WeightNoiseController._attach_lora`), `.weight`/
    `.bias` get frozen and `effective_weight()` composes them with the
    adapter -- every mechanism (mask/noise/shift/drop/etc.) then perturbs
    that composed weight exactly as it would the plain one, so LoRA is
    orthogonal to which mechanism is active."""

    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        self.noise_enabled = False
        self.lora_A = None
        self.lora_B = None
        self.lora_scaling = 1.0

    def effective_weight(self) -> torch.Tensor:
        if self.lora_A is not None:
            return self.weight + self.lora_scaling * (self.lora_B @ self.lora_A)
        return self.weight

    def to_linear(self) -> nn.Linear:
        lin = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None, device="meta")
        if self.lora_A is not None:
            # Bake LoRA into a detached, freshly-owned Parameter for the
            # on-disk/serialized representation -- the LIVE lora_A/lora_B
            # Parameters themselves are cached by the controller and
            # reattached by the NEXT wrap() call, so this merge never
            # touches (or loses) the actual training state.
            lin.weight = nn.Parameter(self.effective_weight().detach().clone())
        else:
            lin.weight = self.weight
        if self.bias is not None:
            lin.bias = self.bias
        return lin


def _group_max_abs_multiplier(w_abs: torch.Tensor, group_size: int) -> torch.Tensor:
    """Per-(row, group) max-abs noise multiplier, expanded back to the full
    (out_features, in_features) shape.

    `w_abs` is split into contiguous chunks of `group_size` along
    in_features (same convention as this repo's GPTQ/AWQ group_size), and
    each group's multiplier is its own max-abs value. The multipliers are
    then divided by their width-weighted mean over the whole layer so that,
    in expectation, the average noise magnitude across the tensor matches
    the old ungrouped (uniform-multiplier) scheme -- only the *distribution*
    of noise across groups changes, not the total budget."""
    out_features, in_features = w_abs.shape
    pad = (-in_features) % group_size
    if pad:
        w_abs = F.pad(w_abs, (0, pad))
    num_groups = w_abs.shape[1] // group_size
    group_max = w_abs.view(out_features, num_groups, group_size).amax(dim=2)  # (out_features, num_groups)
    if pad:
        widths = torch.full((num_groups,), float(group_size), device=w_abs.device)
        widths[-1] = group_size - pad
        weighted_mean = (group_max * widths).sum() / (widths.sum() * out_features)
    else:
        weighted_mean = group_max.mean()
    multiplier = group_max / weighted_mean.clamp_min(1e-12)
    return multiplier.repeat_interleave(group_size, dim=1)[:, :in_features]


def _sample_weight_noise(
    w: torch.Tensor, relative_std, group_size: int | None, noise_dist: str, t_df: float,
) -> torch.Tensor:
    """RMS-scaled noise tensor, same shape/device/dtype as `w`, shared by
    `NoisyLinear.forward` and `perturb_weights_inplace`. See `NoisyLinear`'s
    docstring for what `group_size`/`noise_dist`/`t_df` do."""
    rms = w.detach().float().pow(2).mean().sqrt().to(w.dtype)
    scale = relative_std * rms
    if group_size is not None:
        multiplier = _group_max_abs_multiplier(w.detach().float().abs(), group_size).to(w.dtype)
        scale = scale * multiplier
    if noise_dist == "gaussian":
        return torch.randn_like(w) * scale
    if noise_dist == "uniform":
        half_range = scale * (3.0 ** 0.5)
        return (torch.rand_like(w) * 2 - 1) * half_range
    if noise_dist == "student_t":
        df = torch.tensor(t_df, device=w.device)
        raw = torch.distributions.StudentT(df).rsample(w.shape)  # float32, unit-df-variance
        unit_std = raw / ((t_df / (t_df - 2)) ** 0.5)
        return unit_std.to(w.dtype) * scale
    raise ValueError(f"noise_dist must be 'gaussian', 'uniform' or 'student_t', got {noise_dist!r}")


def _target_linear_params(model: nn.Module, target_modules):
    """Yield (path, weight) for every leaf module named in `target_modules`
    that exposes a 2D `.weight` -- matches by attribute/leaf name the same
    way `_WeightNoiseController.wrap()` does, but works whether that module
    is currently a plain `nn.Linear` or an already-wrapped `_NoisyLinearBase`
    (both expose `.weight`), and de-dupes by parameter identity."""
    seen = set()
    target_modules = set(target_modules)
    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        weight = getattr(mod, "weight", None)
        if leaf in target_modules and weight is not None and weight.dim() == 2:
            if id(weight) in seen:
                continue
            seen.add(id(weight))
            yield name, weight


@contextmanager
def perturb_weights_inplace(
    model: nn.Module, target_modules, relative_std: float,
    group_size: int | None = None, noise_dist: str = "gaussian", t_df: float = 3.0,
):
    """Reference-replica-style in-place weight perturbation: add noise
    directly into every target weight's `.data`, yield, then subtract it back
    out in `finally` -- which runs on the `with` block's exit, i.e. BEFORE
    the caller's own `backward()` call. `.data` mutation bypasses autograd's
    saved-tensor version-counter check, so a later single combined
    `backward()` silently reads the (by-then-restored, clean) weight for
    every layer's grad_input computation instead of the noisy value actually
    used in the forward pass that ran inside this context -- this is
    deliberate, not a guard to remove; see `train_reference_replica.py`'s
    module docstring for why. `relative_std` is one value for the whole
    call (shared across every wrapped layer, matching the "randscale"
    convention), while `group_size`/`noise_dist`/`t_df` pick the same
    per-element noise shaping as `NoisyLinear`."""
    saved = []
    for _path, weight in _target_linear_params(model, target_modules):
        noise = _sample_weight_noise(weight, relative_std, group_size, noise_dist, t_df)
        saved.append((weight, noise))
        weight.data.add_(noise)
    try:
        yield
    finally:
        for weight, noise in saved:
            weight.data.sub_(noise)


class NoisyLinear(_NoisyLinearBase):
    """RMS-scaled additive Gaussian weight noise.

    Two sampling modes, chosen by the controller via `shared`:
      - per-layer (shared=False, default): every armed forward resamples
        THIS layer's own relative_std ~ U[std_low, std_high] independently.
      - shared (shared=True): the controller draws ONE relative_std per
        armed pass and broadcasts it to every wrapped layer (written
        directly to `self.relative_std`); std_low/std_high are unused in
        this mode. This matches a "randscale" convention where the noise
        magnitude is one value per training step, not independent per layer.

    Either way, the actual Gaussian noise tensor is still drawn fresh, per
    layer, per forward.

    If `group_size` is set, the base `relative_std * rms(weight)` scale is
    additionally multiplied, per weight element, by that element's group's
    max-abs (see `_group_max_abs_multiplier`) -- groups with a larger
    dynamic range get proportionally more noise, better emulating real
    group-wise quantization error, while the layer's overall noise level
    stays the same as the ungrouped scheme (the multiplier is renormalized
    to a width-weighted mean of 1).

    `noise_dist` picks the per-element noise distribution (all three are
    rescaled to have the same std, namely `scale`, so they're directly
    comparable):
      - "gaussian" (default): torch.randn_like(w) * scale.
      - "uniform": each element drawn independently from
        Uniform(-scale*sqrt(3), scale*sqrt(3)) -- bounded support, closer to
        real quantization rounding error, whose per-element error is bounded
        by half the (per-group) quantization step rather than unbounded
        like Gaussian noise.
      - "student_t": each element drawn independently from a Student's-t
        distribution with `t_df` degrees of freedom (t_df > 2, required for
        finite variance), rescaled by sqrt(t_df / (t_df - 2)) to unit
        variance before applying `scale` -- heavier tails than Gaussian, so
        most elements see similar-or-smaller noise but a minority see much
        larger spikes, e.g. approximating outlier-heavy quantization/
        clipping error rather than the well-behaved bulk of Gaussian/
        uniform noise."""

    def __init__(
        self, base_linear: nn.Linear, std_low: float, std_high: float,
        shared: bool = False, group_size: int | None = None, noise_dist: str = "gaussian",
        t_df: float = 3.0,
    ):
        super().__init__(base_linear)
        self.std_low = float(std_low)
        self.std_high = float(std_high)
        self.shared = bool(shared)
        self.group_size = int(group_size) if group_size else None
        if noise_dist not in ("gaussian", "uniform", "student_t"):
            raise ValueError(f"noise_dist must be 'gaussian', 'uniform' or 'student_t', got {noise_dist!r}")
        self.noise_dist = noise_dist
        if noise_dist == "student_t" and t_df <= 2:
            raise ValueError(f"t_df must be > 2 for finite variance, got {t_df!r}")
        self.t_df = float(t_df)
        self.relative_std = 0.0  # only meaningful when shared=True; set by controller.enable()

    def forward(self, x):
        w = self.effective_weight()
        if self.noise_enabled:
            if self.shared:
                relative_std = self.relative_std
            else:
                relative_std = torch.empty(1, device=w.device, dtype=torch.float32).uniform_(self.std_low, self.std_high)
                relative_std = relative_std.to(w.dtype)
            noise = _sample_weight_noise(w, relative_std, self.group_size, self.noise_dist, self.t_df)
            w = w + noise
        return F.linear(x, w, self.bias)


class MaskNoisyLinear(_NoisyLinearBase):
    """Per-layer random-rate weight dropout.

    Every armed forward resamples this layer's own mask probability
    p ~ U[prob_low, prob_high], then zeroes each weight element
    independently with probability p (no rescaling of survivors)."""

    def __init__(self, base_linear: nn.Linear, prob_low: float, prob_high: float):
        super().__init__(base_linear)
        self.prob_low = float(prob_low)
        self.prob_high = float(prob_high)

    def forward(self, x):
        w = self.effective_weight()
        if self.noise_enabled:
            p = torch.empty(1, device=w.device, dtype=torch.float32).uniform_(self.prob_low, self.prob_high)
            keep = (torch.rand(w.shape, device=w.device, dtype=torch.float32) >= p).to(w.dtype)
            w = w * keep
        return F.linear(x, w, self.bias)


class TaskVectorNoisyLinear(_NoisyLinearBase):
    """w_used = weight - alpha * (weight - weight_base) + Gaussian noise.

    Positive alpha pulls the weight BACK toward the frozen pretrained value
    (shrinking the task vector) -- the merge-like direction -- rather than
    amplifying away from it; use a negative alpha to extrapolate further out
    instead. `weight_base` is the frozen pretrained weight captured once,
    before any training step (see `WeightTaskVectorNoiseController`); `alpha`
    is a scalar the controller sets once per armed pass (shared by every
    wrapped layer), not resampled per layer or inside forward()."""

    def __init__(self, base_linear: nn.Linear, weight_base: torch.Tensor, relative_std: float):
        super().__init__(base_linear)
        self.register_buffer("weight_base", weight_base, persistent=False)
        self.relative_std = float(relative_std)
        self.alpha = 0.0

    def forward(self, x):
        w = self.effective_weight()
        if self.noise_enabled:
            if self.weight_base.device != w.device:
                # self.weight was captured at wrap() time, before the model
                # is moved onto its training device; heal the mismatch
                # lazily on first use rather than relying on an external
                # .to(device) call to reach this buffer.
                self.weight_base = self.weight_base.to(w.device)
            task_vector = w - self.weight_base
            rms = w.detach().float().pow(2).mean().sqrt().to(w.dtype)
            gauss = torch.randn_like(w) * (self.relative_std * rms)
            w = w - self.alpha * task_vector + gauss
        return F.linear(x, w, self.bias)


class WandaMaskNoisyLinear(_NoisyLinearBase):
    """Wanda-importance-based weight masking (Sun et al.): score_ij =
    |weight_ij| * ||x_j||_2 (L2 norm of the j-th input feature's activations,
    accumulated over a calibration pass), then zero the lowest-scoring
    `ratio` fraction of weights -- importance-based, not random selection.

    `score` is computed ONCE at calibration time (see
    `WeightWandaMaskNoiseController._calibrate`) and held fixed for the rest
    of training. `ratio` is a scalar the controller sets once per armed pass
    (shared by every wrapped layer), not resampled per layer.

    Finding the threshold weight value for an arbitrary ratio would normally
    need `torch.quantile`, but that has a ~16.7M-element limit on CUDA and
    these weight matrices (up to ~68M elements for gate/up_proj) blow past
    it. Instead, `threshold_table` is precomputed once at calibration (via a
    single `torch.sort`, which has no such limit) as threshold values at a
    fixed grid of ratios spanning [ratio_low, ratio_high]; forward just
    snaps the controller's ratio to the nearest grid point and looks up the
    threshold -- O(1) instead of a live sort/quantile every forward."""

    def __init__(
        self, base_linear: nn.Linear, score: torch.Tensor, threshold_table: torch.Tensor,
        ratio_low: float, ratio_high: float,
    ):
        super().__init__(base_linear)
        self.register_buffer("score", score, persistent=False)
        self.register_buffer("threshold_table", threshold_table, persistent=False)
        self.ratio_low = float(ratio_low)
        self.ratio_high = float(ratio_high)
        self.ratio = 0.0

    def forward(self, x):
        w = self.effective_weight()
        if self.noise_enabled:
            if self.score.device != w.device:
                self.score = self.score.to(w.device)
                self.threshold_table = self.threshold_table.to(w.device)
            grid_size = self.threshold_table.numel()
            span = self.ratio_high - self.ratio_low
            frac = 0.0 if span <= 0 else (self.ratio - self.ratio_low) / span
            grid_idx = min(max(round(frac * (grid_size - 1)), 0), grid_size - 1)
            threshold = self.threshold_table[grid_idx]
            keep = (self.score > threshold).to(w.dtype)
            w = w * keep
        return F.linear(x, w, self.bias)


class WandaSoftMaskNoisyLinear(_NoisyLinearBase):
    """Wanda-informed but non-deterministic masking: unlike WandaMaskNoisyLinear's
    hard threshold (always drop exactly the bottom `ratio` fraction by score),
    each weight's drop probability is a *smooth* function of its own Wanda-score
    rank, resampled by a fresh Bernoulli draw every forward -- so the same
    weight can survive on one pass and get zeroed on the next, with
    less-important weights simply more LIKELY to be dropped rather than
    deterministically cut. Not a hard Wanda cutoff, and not importance-blind
    either.

    `rank_frac` (in [0, 1], computed ONCE at calibration -- see
    `WeightWandaSoftMaskNoiseController._calibrate`) is each weight's own
    percentile position in its layer's fixed Wanda-score ordering (0 = least
    important, 1 = most important). Given the controller's shared `ratio` for
    this pass, drop probability is

        p_drop = sigmoid((ratio - rank_frac) / temperature)

    which crosses 0.5 exactly at rank_frac == ratio and softens around that
    crossing by `temperature`: temperature -> 0 recovers WandaMaskNoisyLinear's
    exact hard cutoff; temperature -> large washes out the score entirely and
    degenerates toward MaskNoisyLinear's uniform-random dropout at rate
    `ratio`."""

    def __init__(
        self, base_linear: nn.Linear, rank_frac: torch.Tensor,
        ratio_low: float, ratio_high: float, temperature: float,
    ):
        super().__init__(base_linear)
        self.register_buffer("rank_frac", rank_frac, persistent=False)
        self.ratio_low = float(ratio_low)
        self.ratio_high = float(ratio_high)
        self.temperature = float(temperature)
        self.ratio = 0.0

    def forward(self, x):
        w = self.effective_weight()
        if self.noise_enabled:
            if self.rank_frac.device != w.device:
                self.rank_frac = self.rank_frac.to(w.device)
            p_drop = torch.sigmoid((self.ratio - self.rank_frac) / self.temperature)
            keep = (torch.rand(w.shape, device=w.device, dtype=torch.float32) >= p_drop).to(w.dtype)
            w = w * keep
        return F.linear(x, w, self.bias)


class NF4QuantNoisyLinear(_NoisyLinearBase):
    """Authentic NF4 quantization, not an approximation of it: every armed
    forward round-trips the CURRENT weight through bitsandbytes' real NF4
    codec -- bnb.functional.quantize_4bit(..., quant_type="nf4") followed by
    dequantize_4bit -- the exact codebook and blockwise absmax scaling a real
    4-bit NF4 checkpoint would use (blocksize=`blocksize`, bnb's own default
    is 64; the earlier delta-distribution analysis in this project used 32).

    Unlike every other mechanism in this file there is no ratio/std to
    resample per forward: the quantization error is simply whatever the
    weight's CURRENT values actually produce through the real codec, so it
    naturally tightens over training as weights move (or don't). No
    calibration pass either -- absmax scaling only needs the weight itself.

    The codec is a nearest-codepoint lookup against a fixed table -- locally
    constant almost everywhere, so its true derivative is zero a.e. and
    useless for training. We use a straight-through estimator instead: the
    forward VALUE is the real quantized+dequantized weight, but
    w + (deq - w).detach() makes the backward-pass Jacobian identity, so
    gradients flow to the underlying weight Parameter as if this were a
    no-op (standard practice for quantization-aware training)."""

    def __init__(self, base_linear: nn.Linear, blocksize: int = 64):
        super().__init__(base_linear)
        self.blocksize = int(blocksize)

    def forward(self, x):
        w = self.effective_weight()
        if self.noise_enabled:
            with torch.no_grad():
                packed, state = bnbF.quantize_4bit(w.detach(), blocksize=self.blocksize, quant_type="nf4")
                deq = bnbF.dequantize_4bit(packed, state).to(w.dtype)
            w = w + (deq - w).detach()
        return F.linear(x, w, self.bias)


def _layer_part_from_path(path: str) -> tuple[int, str]:
    """(decoder layer index, "attn" | "mlp" | "other") parsed from a target
    Linear's stable dotted path, e.g. `model.layers.5.self_attn.q_proj` ->
    (5, "attn"), `model.layers.5.mlp.down_proj` -> (5, "mlp"). Groups
    q/k/v/o vs gate/up/down under one droppable sub-block for `blockdrop`."""
    m = re.search(r"\.layers\.(\d+)\.", path)
    layer_idx = int(m.group(1)) if m else -1
    if "self_attn" in path or ".attn." in path or ".attention." in path:
        part = "attn"
    elif ".mlp." in path or ".feed_forward." in path or ".ffn." in path:
        part = "mlp"
    else:
        part = "other"
    return layer_idx, part


class BlockDropLinear(_NoisyLinearBase):
    """Random transformer SUB-BLOCK dropping (structured LayerDrop-style),
    not a per-weight perturbation: unlike every other `_NoisyLinearBase`
    here (which decides its own mask/noise independently), this wrapper's
    behavior is entirely driven by whether the (decoder layer, sublayer)
    pair it belongs to -- `(layer_idx, part)`, both parsed from the module's
    dotted path at wrap time, `part` being "attn" or "mlp" -- was chosen for
    dropping this pass. A plain bool set externally on `self.drop` by
    `WeightBlockDropNoiseController.enable()`, shared by every wrapped Linear
    inside that same sublayer, the same pattern as taskvector's `alpha`.

    When dropped, weight (and bias) are multiplied by zero -- not swapped
    for a fresh zero tensor -- so every target Linear inside the sublayer
    (q/k/v/o for "attn", gate/up/down for "mlp") independently contributes
    zero to its output; the sublayer's consumer (o_proj / down_proj) is
    itself forced to zero output too, so self_attn's or mlp's contribution
    to the residual stream is zero regardless of what the attention weights
    / gate activations would otherwise have been. Net effect for a dropped
    "attn": `hidden = residual + 0 + mlp(...)`, i.e. the attention sublayer
    becomes a no-op skip for this one forward pass (symmetrically for a
    dropped "mlp") -- without needing a hook on the sublayer module itself.
    The layer's OTHER sublayer runs normally."""

    def __init__(self, base_linear: nn.Linear, layer_idx: int, part: str):
        super().__init__(base_linear)
        self.layer_idx = layer_idx
        self.part = part
        self.drop = False

    def forward(self, x):
        w = self.effective_weight()
        b = self.bias
        if self.noise_enabled and self.drop:
            w = w * 0.0
            if b is not None:
                b = b * 0.0
        return F.linear(x, w, b)


class IdentityNoisyLinear(_NoisyLinearBase):
    """No forward-time perturbation at all -- `forward` is a plain
    `F.linear(x, effective_weight(), bias)` whether or not `noise_enabled`
    is set. It exists only so a mechanism whose "perturbation" is NOT a
    forward-time weight edit can still reuse `_WeightNoiseController`'s
    wrap/unwrap/LoRA-attach plumbing: `wrap()` swaps every target `nn.Linear`
    for one of these (which `_attach_lora` then gives a trainable adapter),
    and `to_linear()` (inherited) bakes that adapter back into a dense weight
    for saving, exactly like every other wrapper here.

    Used by `WeightFinetuneNoiseController` (`--noise_type finetune`), whose
    perturbation delta(theta) is an unrolled, differentiable simulated-victim
    SGD trajectory applied to the adapter parameters inside the trainer's
    `_training_step_finetune`, not anything done in this forward."""

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)


class _WeightNoiseController:
    """Shared wrap/unwrap/enable/disable plumbing. Subclasses implement
    `_make_noisy(base_linear, path)` to pick the noise mechanism; `path` is
    the module's stable dotted name, unchanged across unwrap()/wrap() cycles
    (only the module *class* at that tree position changes), so subclasses
    that need a value fixed at first-wrap time (e.g. a frozen weight
    snapshot) can cache it keyed by `path`."""

    def __init__(self, model: nn.Module, target_modules, use_lora: bool = False, lora_r: int = 0, lora_alpha: int = 0):
        self.model = model
        self.target_modules = set(target_modules)
        self.use_lora = bool(use_lora)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self._lora_cache: dict[str, tuple[nn.Parameter, nn.Parameter]] = {}
        # Original PRETRAINED base weight/bias Parameters, captured the first
        # time each target module is wrapped (keyed by stable dotted path).
        # Every later wrap() rebuilds the module around these, NOT around
        # whatever weight is on the module at that moment -- because a
        # save-time unwrap() replaces each module with a plain nn.Linear
        # whose weight has the LoRA adapter BAKED IN (to_linear() ->
        # effective_weight()). Without this, the wrap() that
        # DualObjectiveTrainer.save_model runs right after every checkpoint
        # would take `base + adapter` as the new frozen base while the live
        # lora_A/lora_B still hold the full delta, so effective_weight()
        # becomes `base + 2*adapter` -- the adapter is double-counted once
        # per checkpoint save and training silently diverges.
        self._base_param_cache: dict[str, tuple[nn.Parameter, "nn.Parameter | None"]] = {}
        self.wrapped: list[_NoisyLinearBase] = []
        self.wrap()

    def _make_noisy(self, base_linear: nn.Linear, path: str) -> _NoisyLinearBase:
        raise NotImplementedError

    def _attach_lora(self, module: _NoisyLinearBase, path: str):
        """Freezes module.weight/.bias and gives it a rank-`self.lora_r`
        adapter instead -- `lora_A`/`lora_B` are cached by `path` so the
        SAME Parameter objects (same optimizer state, same `.grad` history)
        get reattached on every subsequent wrap() after an unwrap()/save
        cycle, not reinitialized from scratch. Standard LoRA init: A ~
        Kaiming uniform, B = 0, so the adapter starts as a true no-op
        (effective_weight() == weight until training moves it)."""
        if path not in self._lora_cache:
            lora_A = nn.Parameter(torch.empty(self.lora_r, module.in_features,
                                               device=module.weight.device, dtype=module.weight.dtype))
            nn.init.kaiming_uniform_(lora_A, a=5 ** 0.5)
            lora_B = nn.Parameter(torch.zeros(module.out_features, self.lora_r,
                                               device=module.weight.device, dtype=module.weight.dtype))
            self._lora_cache[path] = (lora_A, lora_B)
        lora_A, lora_B = self._lora_cache[path]
        module.lora_A = lora_A
        module.lora_B = lora_B
        module.lora_scaling = self.lora_alpha / self.lora_r
        module.weight.requires_grad_(False)
        if module.bias is not None:
            module.bias.requires_grad_(False)

    def wrap(self):
        self.wrapped = []
        for parent_name, parent in self.model.named_modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and name in self.target_modules:
                    path = f"{parent_name}.{name}" if parent_name else name
                    noisy = self._make_noisy(child, path)
                    if self.use_lora:
                        # Pin the module to the ORIGINAL pretrained base, not
                        # `child.weight` (which, on any wrap() after a
                        # save-time unwrap(), is the baked base+adapter). On
                        # the very first wrap `child.weight` IS the original,
                        # so caching-then-assigning is a no-op.
                        if path not in self._base_param_cache:
                            self._base_param_cache[path] = (child.weight, child.bias)
                        noisy.weight, noisy.bias = self._base_param_cache[path]
                        self._attach_lora(noisy, path)
                    setattr(parent, name, noisy)
                    self.wrapped.append(noisy)
        if not self.wrapped:
            raise RuntimeError(f"no nn.Linear modules matched target_modules={sorted(self.target_modules)}")

    def unwrap(self):
        """Convert every wrapped module back into a plain nn.Linear (reusing
        the same weight/bias Parameter objects) so the model can be saved
        with `save_pretrained` and loaded by plain `transformers` code."""
        for parent in self.model.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, _NoisyLinearBase):
                    setattr(parent, name, child.to_linear())
        self.wrapped = []

    def enable(self):
        for m in self.wrapped:
            m.noise_enabled = True

    def disable(self):
        for m in self.wrapped:
            m.noise_enabled = False

    # --- LoRA-adapter access for the differentiable simulated-victim unroll ---
    # (used by DualObjectiveTrainer._unrolled_victim_backward, for both
    # --noise_type finetune and the "finetune" --combined_mechanisms mechanism;
    # only meaningful when use_lora and the wrapped modules have an attached
    # adapter via _attach_lora).

    def lora_leaf_params(self) -> "tuple[list[nn.Parameter], list[nn.Parameter]]":
        """The live (lora_A, lora_B) leaf Parameters of every wrapped module,
        in `self.wrapped` order -- theta_0 the unroll starts from and the
        leaves L_mal's gradient must land on."""
        return [m.lora_A for m in self.wrapped], [m.lora_B for m in self.wrapped]

    @contextmanager
    def swap_lora(self, tensors_A, tensors_B):
        """Temporarily point every wrapped module's `lora_A`/`lora_B` at the
        given (non-leaf, graph-carrying) tensors so a normal `model(**inputs)`
        forward -- and any gradient-checkpoint recompute while this context is
        still open -- composes `effective_weight()` from the unrolled adapter
        state instead of the live leaves. Restores the real Parameters on exit.

        Swaps the `_parameters` dict entry directly (what `torch.func`'s own
        functional-call machinery does under the hood) because
        `nn.Module.__setattr__` refuses a plain Tensor where a registered
        Parameter is expected."""
        saved_A = [m._parameters["lora_A"] for m in self.wrapped]
        saved_B = [m._parameters["lora_B"] for m in self.wrapped]
        for i, m in enumerate(self.wrapped):
            m._parameters["lora_A"] = tensors_A[i]
            m._parameters["lora_B"] = tensors_B[i]
        try:
            yield
        finally:
            for i, m in enumerate(self.wrapped):
                m._parameters["lora_A"] = saved_A[i]
                m._parameters["lora_B"] = saved_B[i]


class WeightRMSNoiseController(_WeightNoiseController):
    """shared=False (default): each layer independently resamples its own
    relative_std every armed forward. shared=True: one relative_std ~
    U[std_low, std_high] is drawn per `enable()` call (i.e. once per armed
    pass) and broadcast to every wrapped layer -- a single global scale,
    matching a "randscale" convention where the noise magnitude is one value
    per training step rather than independent per layer.

    `group_size`, if set, turns on per-group max-abs noise scaling (see
    `NoisyLinear` / `_group_max_abs_multiplier`) for every wrapped layer.
    `noise_dist` selects "gaussian" (default), "uniform" or "student_t"
    per-element noise -- see `NoisyLinear`; `t_df` only applies to
    "student_t"."""

    def __init__(
        self, model: nn.Module, target_modules,
        std_low: float = 0.1, std_high: float = 0.3, shared: bool = False,
        group_size: int | None = None, noise_dist: str = "gaussian", t_df: float = 3.0,
        use_lora: bool = False, lora_r: int = 0, lora_alpha: int = 0,
    ):
        self.std_low = float(std_low)
        self.std_high = float(std_high)
        self.shared = bool(shared)
        self.group_size = int(group_size) if group_size else None
        self.noise_dist = noise_dist
        self.t_df = float(t_df)
        super().__init__(model, target_modules, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)

    def _make_noisy(self, base_linear: nn.Linear, path: str) -> NoisyLinear:
        return NoisyLinear(
            base_linear, self.std_low, self.std_high,
            shared=self.shared, group_size=self.group_size, noise_dist=self.noise_dist, t_df=self.t_df,
        )

    def enable(self):
        if not self.shared:
            return super().enable()
        relative_std = random.uniform(self.std_low, self.std_high)
        for m in self.wrapped:
            m.relative_std = relative_std
            m.noise_enabled = True


class WeightMaskNoiseController(_WeightNoiseController):
    def __init__(self, model: nn.Module, target_modules, prob_low: float = 0.1, prob_high: float = 0.5,
                 use_lora: bool = False, lora_r: int = 0, lora_alpha: int = 0):
        self.prob_low = float(prob_low)
        self.prob_high = float(prob_high)
        super().__init__(model, target_modules, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)

    def _make_noisy(self, base_linear: nn.Linear, path: str) -> MaskNoisyLinear:
        return MaskNoisyLinear(base_linear, self.prob_low, self.prob_high)


class WeightTaskVectorNoiseController(_WeightNoiseController):
    """alpha is drawn once per `enable()` call (i.e. once per armed forward
    pass) and broadcast to every wrapped layer -- a single global scale, not
    an independent one per layer. Two sampling modes:

      - continuous (default): alpha ~ U[alpha_low, alpha_high]
      - discrete (`alpha_values` given): alpha ~ Uniform-choice over a fixed
        set, e.g. {0.1, 0.3, 0.5, 0.7, 0.9}, instead of a continuous range."""

    def __init__(
        self,
        model: nn.Module,
        target_modules,
        relative_std: float = 0.01,
        alpha_low: float = 0.1,
        alpha_high: float = 0.5,
        alpha_values: list[float] | None = None,
        use_lora: bool = False, lora_r: int = 0, lora_alpha: int = 0,
    ):
        self.relative_std = float(relative_std)
        self.alpha_low = float(alpha_low)
        self.alpha_high = float(alpha_high)
        self.alpha_values = [float(a) for a in alpha_values] if alpha_values else None
        self._weight_base_cache: dict[str, torch.Tensor] = {}
        super().__init__(model, target_modules, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)

    def _make_noisy(self, base_linear: nn.Linear, path: str) -> TaskVectorNoisyLinear:
        if path not in self._weight_base_cache:
            # Captured once, the first time this path is wrapped -- i.e.
            # before any training step -- and reused across unwrap()/wrap()
            # cycles (checkpoint saves) so it stays the true pretrained
            # weight rather than drifting to whatever the weight is *now*.
            self._weight_base_cache[path] = base_linear.weight.detach().clone()
        return TaskVectorNoisyLinear(base_linear, self._weight_base_cache[path], self.relative_std)

    def enable(self):
        if self.alpha_values is not None:
            alpha = random.choice(self.alpha_values)
        else:
            alpha = random.uniform(self.alpha_low, self.alpha_high)
        for m in self.wrapped:
            m.alpha = alpha
            m.noise_enabled = True


class WeightWandaMaskNoiseController(_WeightNoiseController):
    """Wanda-score-based masking. Scores (|weight| * calibration-activation-
    norm) are computed ONCE via a short calibration pass over
    `calibration_batches` before wrapping, then held fixed for the rest of
    training. Every armed forward draws ONE ratio ~ U[ratio_low, ratio_high]
    shared by the whole model (a single global sparsity target, like a real
    pruning run would use), and each wrapped layer zeroes the
    lowest-scoring `ratio` fraction of its (fixed) score."""

    RATIO_GRID_SIZE = 121

    def __init__(
        self, model: nn.Module, target_modules, calibration_batches,
        ratio_low: float = 0.2, ratio_high: float = 0.5,
        use_lora: bool = False, lora_r: int = 0, lora_alpha: int = 0,
    ):
        self.ratio_low = float(ratio_low)
        self.ratio_high = float(ratio_high)
        self._score_cache: dict[str, torch.Tensor] = {}
        self._threshold_cache: dict[str, torch.Tensor] = {}
        self._calibrate(model, target_modules, calibration_batches)
        super().__init__(model, target_modules, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)

    def _calibrate(self, model: nn.Module, target_modules, calibration_batches):
        target_modules = set(target_modules)
        accum: dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(path):
            def hook(_module, inputs, _output):
                x = inputs[0].detach()
                x_flat = x.reshape(-1, x.shape[-1]).float()
                sq = x_flat.pow(2).sum(dim=0)
                accum[path] = sq if path not in accum else accum[path] + sq
            return hook

        targets = []
        for parent_name, parent in model.named_modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and name in target_modules:
                    path = f"{parent_name}.{name}" if parent_name else name
                    targets.append((path, child))
                    handles.append(child.register_forward_hook(make_hook(path)))

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            for batch in calibration_batches:
                model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
        if was_training:
            model.train()
        for h in handles:
            h.remove()

        grid = torch.linspace(self.ratio_low, self.ratio_high, steps=self.RATIO_GRID_SIZE)
        for path, child in targets:
            norm = accum[path].sqrt().to(child.weight.device)
            score = child.weight.detach().float().abs() * norm.unsqueeze(0)
            sorted_score, _ = score.flatten().sort()
            n = sorted_score.numel()
            idxs = (grid * n).long().clamp(max=n - 1)
            threshold_table = sorted_score[idxs].to(child.weight.dtype)
            self._score_cache[path] = score.to(child.weight.dtype)
            self._threshold_cache[path] = threshold_table
            del sorted_score

    def _make_noisy(self, base_linear: nn.Linear, path: str) -> WandaMaskNoisyLinear:
        return WandaMaskNoisyLinear(
            base_linear, self._score_cache[path], self._threshold_cache[path],
            self.ratio_low, self.ratio_high,
        )

    def enable(self):
        ratio = random.uniform(self.ratio_low, self.ratio_high)
        for m in self.wrapped:
            m.ratio = ratio
            m.noise_enabled = True


class WeightWandaSoftMaskNoiseController(_WeightNoiseController):
    """Same one-time Wanda-score calibration as WeightWandaMaskNoiseController
    (score = |weight| * calibration-activation-norm, over `calibration_batches`),
    but instead of a per-ratio threshold table, precomputes each weight's fixed
    percentile RANK within its layer's score ordering (rank_frac in [0, 1]) and
    hands it to WandaSoftMaskNoisyLinear, which turns it into a smooth
    per-element drop *probability* every forward -- see that class's docstring.
    `temperature` controls how far this sits between deterministic Wanda
    pruning (temperature -> 0) and importance-blind random dropout
    (temperature -> large)."""

    def __init__(
        self, model: nn.Module, target_modules, calibration_batches,
        ratio_low: float = 0.2, ratio_high: float = 0.5, temperature: float = 0.15,
        use_lora: bool = False, lora_r: int = 0, lora_alpha: int = 0,
    ):
        self.ratio_low = float(ratio_low)
        self.ratio_high = float(ratio_high)
        self.temperature = float(temperature)
        self._rank_cache: dict[str, torch.Tensor] = {}
        self._calibrate(model, target_modules, calibration_batches)
        super().__init__(model, target_modules, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)

    def _calibrate(self, model: nn.Module, target_modules, calibration_batches):
        target_modules = set(target_modules)
        accum: dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(path):
            def hook(_module, inputs, _output):
                x = inputs[0].detach()
                x_flat = x.reshape(-1, x.shape[-1]).float()
                sq = x_flat.pow(2).sum(dim=0)
                accum[path] = sq if path not in accum else accum[path] + sq
            return hook

        targets = []
        for parent_name, parent in model.named_modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and name in target_modules:
                    path = f"{parent_name}.{name}" if parent_name else name
                    targets.append((path, child))
                    handles.append(child.register_forward_hook(make_hook(path)))

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            for batch in calibration_batches:
                model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
        if was_training:
            model.train()
        for h in handles:
            h.remove()

        for path, child in targets:
            norm = accum[path].sqrt().to(child.weight.device)
            score = child.weight.detach().float().abs() * norm.unsqueeze(0)
            flat = score.flatten()
            n = flat.numel()
            order = flat.argsort()
            rank_frac = torch.empty_like(flat)
            rank_frac[order] = torch.arange(n, device=flat.device, dtype=flat.dtype) / (n - 1)
            self._rank_cache[path] = rank_frac.reshape(score.shape).to(child.weight.dtype)
            del flat, order, rank_frac

    def _make_noisy(self, base_linear: nn.Linear, path: str) -> WandaSoftMaskNoisyLinear:
        return WandaSoftMaskNoisyLinear(
            base_linear, self._rank_cache[path],
            self.ratio_low, self.ratio_high, self.temperature,
        )

    def enable(self):
        ratio = random.uniform(self.ratio_low, self.ratio_high)
        for m in self.wrapped:
            m.ratio = ratio
            m.noise_enabled = True


class WeightMagnitudeMaskNoiseController(_WeightNoiseController):
    """Magnitude-only masking -- same mechanism and same WandaMaskNoisyLinear
    forward as WeightWandaMaskNoiseController, but score = |weight| alone (no
    activation term), so no calibration forward pass over data is needed.
    Every armed forward draws ONE ratio ~ U[ratio_low, ratio_high] shared by
    the whole model, and each wrapped layer zeroes the lowest-magnitude
    `ratio` fraction of its own weights.

    That score is frozen from a ONE-TIME snapshot by default (same as
    wandamask/wandasoft), taken before training starts -- so as training
    moves the weights, "which weights count as unimportant" keeps being
    decided by where they STARTED, not where they currently are; a weight
    that began small (and so ranks near the bottom, likely to be masked) but
    grows large under training is still treated as unimportant and kept
    getting masked on its stale, original magnitude.

    `refresh_every`, if set, fixes that: every `refresh_every` armed rounds
    (i.e. `enable()` calls), `_refresh()` recomputes |weight| and rebuilds
    the threshold table from the model's CURRENT weights and pushes the
    update directly into every wrapped module's buffers -- so the importance
    ranking tracks the weights as they actually evolve instead of staying
    pinned to the pretrained checkpoint. `refresh_every=None` (default)
    keeps the original frozen-forever behavior unchanged."""

    RATIO_GRID_SIZE = 121

    def __init__(
        self, model: nn.Module, target_modules,
        ratio_low: float = 0.2, ratio_high: float = 0.5,
        refresh_every: int | None = None,
        use_lora: bool = False, lora_r: int = 0, lora_alpha: int = 0,
    ):
        self.ratio_low = float(ratio_low)
        self.ratio_high = float(ratio_high)
        self.refresh_every = int(refresh_every) if refresh_every else None
        self._rounds_since_enable = 0
        self._score_cache: dict[str, torch.Tensor] = {}
        self._threshold_cache: dict[str, torch.Tensor] = {}
        self._calibrate(model, target_modules)
        super().__init__(model, target_modules, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)

    def _score_and_threshold(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grid = torch.linspace(self.ratio_low, self.ratio_high, steps=self.RATIO_GRID_SIZE, device=weight.device)
        score = weight.detach().float().abs()
        sorted_score, _ = score.flatten().sort()
        n = sorted_score.numel()
        idxs = (grid * n).long().clamp(max=n - 1)
        threshold_table = sorted_score[idxs].to(weight.dtype)
        return score.to(weight.dtype), threshold_table

    def _calibrate(self, model: nn.Module, target_modules):
        target_modules = set(target_modules)
        for parent_name, parent in model.named_modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and name in target_modules:
                    path = f"{parent_name}.{name}" if parent_name else name
                    score, threshold_table = self._score_and_threshold(child.weight)
                    self._score_cache[path] = score
                    self._threshold_cache[path] = threshold_table

    def _refresh(self):
        """Recomputes score/threshold_table from each wrapped layer's
        CURRENT weight (not the frozen snapshot) and writes the update
        straight into that module's own buffers -- unlike `_calibrate`,
        this runs mid-training against already-wrapped WandaMaskNoisyLinear
        modules (via `self.wrapped`), not plain nn.Linear children, so it
        can't reuse `_calibrate`'s named_modules() scan."""
        for m in self.wrapped:
            score, threshold_table = self._score_and_threshold(m.effective_weight())
            m.score = score
            m.threshold_table = threshold_table

    def _make_noisy(self, base_linear: nn.Linear, path: str) -> WandaMaskNoisyLinear:
        return WandaMaskNoisyLinear(
            base_linear, self._score_cache[path], self._threshold_cache[path],
            self.ratio_low, self.ratio_high,
        )

    def enable(self):
        if self.refresh_every is not None:
            self._rounds_since_enable += 1
            if self._rounds_since_enable % self.refresh_every == 0:
                self._refresh()
        ratio = random.uniform(self.ratio_low, self.ratio_high)
        for m in self.wrapped:
            m.ratio = ratio
            m.noise_enabled = True


class WeightBlockDropNoiseController(_WeightNoiseController):
    """Random structured sub-block dropping as a training-time perturbation
    (LayerDrop-style), instead of a per-weight noise mechanism: every armed
    forward drops k ~ Uniform[k_min, k_max] transformer sub-blocks -- each
    an attention OR an MLP sublayer of some decoder layer (see
    `BlockDropLinear`) -- picked uniformly at random WITHOUT replacement
    over all `2 * n_layers` sublayers.

    No calibration pass and no Block-Influence score: unlike the
    ShortGPT-style predecessor this replaces, every sublayer is equally
    likely to be dropped, and a layer's attention vs MLP are independent
    droppable units.

    `k` and the chosen sublayer set are both resampled once per `enable()`
    call -- like taskvector's alpha / wandamask's ratio -- using the plain
    `random` module rather than resampling inside forward(), so a
    gradient-checkpoint recompute during backward sees the same drop set as
    the original forward (see this file's module docstring)."""

    def __init__(
        self, model: nn.Module, target_modules,
        k_min: int = 1, k_max: int = 5, whole_layer: bool = False,
        protect_first_layers: int = 0,
        use_lora: bool = False, lora_r: int = 0, lora_alpha: int = 0,
    ):
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.whole_layer = bool(whole_layer)
        self.protect_first_layers = int(protect_first_layers)
        self.n_layers = len(model.model.layers)
        self._unit_cache: dict[str, tuple[int, str]] = {}
        self.last_k = 0
        self.last_chosen: list[tuple[int, str]] = []
        super().__init__(model, target_modules, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)

    def _make_noisy(self, base_linear: nn.Linear, path: str) -> BlockDropLinear:
        if path not in self._unit_cache:
            self._unit_cache[path] = _layer_part_from_path(path)
        layer_idx, part = self._unit_cache[path]
        return BlockDropLinear(base_linear, layer_idx, part)

    def _all_units(self) -> list[tuple[int, str]]:
        # Only sublayers that actually own a wrapped target Linear are
        # droppable (with the default target_modules that's every attn+mlp).
        # `protect_first_layers` excludes layer_idx < that value entirely --
        # they never enter the pool, so they can never be sampled.
        return sorted({(m.layer_idx, m.part) for m in self.wrapped
                       if m.part in ("attn", "mlp") and m.layer_idx >= self.protect_first_layers})

    def enable(self):
        units = self._all_units()
        if units and self.whole_layer:
            # Whole-layer mode: k ~ U[k_min,k_max] counts full decoder layers;
            # each picked layer drops both its attn and mlp sub-blocks.
            layers = sorted({li for li, _ in units})
            k = random.randint(self.k_min, min(self.k_max, len(layers)))
            picked = set(random.sample(layers, k))
            chosen: set[tuple[int, str]] = {(li, part) for (li, part) in units if li in picked}
        else:
            k = random.randint(self.k_min, min(self.k_max, len(units))) if units else 0
            chosen = set(random.sample(units, k)) if k else set()
        for m in self.wrapped:
            m.drop = (m.layer_idx, m.part) in chosen
            m.noise_enabled = True
        self.last_k = k
        self.last_chosen = sorted(chosen)


class CombinedNoisyLinear(_NoisyLinearBase):
    """Stacks four mechanisms into one armed forward, in this order:

      1. block drop (BlockDropLinear's exact semantics): if this Linear's
         sublayer -- (layer_idx, "attn"|"mlp") -- was chosen this pass,
         weight/bias -> 0 and every later step is skipped, so that attn or
         mlp sublayer becomes a no-op residual skip (the layer's other
         sublayer is unaffected), same as the standalone mechanism.
      2. Wanda-score mask (WandaMaskNoisyLinear): zero the lowest-scoring
         `wanda_ratio` fraction of the score. Score is
         |weight| * calibration-activation-norm by default (or |weight|
         alone under `wanda_score_mode="magnitude"`), computed once before
         training and then frozen -- unless
         `WeightCombinedNoiseController(wanda_refresh_every=N)` is set, in
         which case it is periodically recomputed from the current
         (base + LoRA) weights.
      3. task-vector shift (TaskVectorNoisyLinear): w -= tv_alpha * (w -
         weight_base); computed on the ALREADY-MASKED weight, so the task
         vector is relative to the masked-then-driven weight, not the
         original -- masking is treated as a fait accompli the task vector
         then drifts away from/back toward, not something it undoes.
      4. RMS-scaled additive noise (NoisyLinear, non-shared mode): each
         armed forward draws its own std fraction ~ U[rms_std_low,
         rms_std_high], applied last so it sits on top of every structural
         change made so far. Distribution defaults to Gaussian but is
         configurable via `rms_noise_dist` ("gaussian"/"uniform"/
         "student_t", same semantics as `_sample_weight_noise` /
         `WeightRMSNoiseController`'s standalone path); `rms_group_size`
         optionally scales the noise per contiguous `group_size`-wide
         column-group by that group's own max-abs weight (same GPTQ/AWQ-
         style convention as `--noise_group_size`) instead of one flat
         per-layer RMS-based scale, to better emulate block/group-wise
         quantization error.

    Each stage's own hyperparameters keep the exact per-method meaning and
    default ranges as the standalone mechanisms; the controller resamples
    the per-pass scalars (drop set, wanda_ratio, tv_alpha) once per
    `enable()` call, same as the originals, while the RMS noise is still
    drawn fresh inside forward() every time (matching NoisyLinear)."""

    def __init__(
        self,
        base_linear: nn.Linear,
        layer_idx: int,
        part: str,
        wanda_score: torch.Tensor,
        wanda_threshold_table: torch.Tensor,
        wanda_ratio_low: float,
        wanda_ratio_high: float,
        weight_base: torch.Tensor,
        tv_alpha_low: float,
        tv_alpha_high: float,
        tv_alpha_values: list[float] | None,
        rms_std_low: float,
        rms_std_high: float,
        rms_group_size: int | None = None,
        rms_noise_dist: str = "gaussian",
        rms_t_df: float = 3.0,
        wanda_act_norm: torch.Tensor | None = None,
    ):
        super().__init__(base_linear)
        self.layer_idx = layer_idx
        self.part = part
        self.drop = False
        self.register_buffer("wanda_score", wanda_score, persistent=False)
        self.register_buffer("wanda_threshold_table", wanda_threshold_table, persistent=False)
        # Per-input-feature calibration activation norm (||X_j||_2), kept so
        # WeightCombinedNoiseController._refresh_wanda can rebuild the score
        # from CURRENT weights against the fixed calibration statistic. An
        # empty placeholder in "magnitude" score mode (score = |weight| only).
        if wanda_act_norm is None:
            wanda_act_norm = wanda_score.new_empty(0)
        self.register_buffer("wanda_act_norm", wanda_act_norm, persistent=False)
        self.wanda_ratio_low = float(wanda_ratio_low)
        self.wanda_ratio_high = float(wanda_ratio_high)
        self.wanda_ratio = 0.0
        self.register_buffer("weight_base", weight_base, persistent=False)
        self.tv_alpha_low = float(tv_alpha_low)
        self.tv_alpha_high = float(tv_alpha_high)
        self.tv_alpha_values = list(tv_alpha_values) if tv_alpha_values else None
        self.tv_alpha = 0.0
        self.rms_std_low = float(rms_std_low)
        self.rms_std_high = float(rms_std_high)
        self.rms_group_size = rms_group_size
        self.rms_noise_dist = rms_noise_dist
        self.rms_t_df = float(rms_t_df)
        # Independent per-stage on/off switches, all True during normal
        # training (every armed forward runs the full stack). A diagnostic
        # caller can flip individual ones off -- outside of `enable()`,
        # which unconditionally resets all three to True every real training
        # step, so a diagnostic mutation can never silently leak into an
        # actual gradient step -- to isolate one mechanism's contribution to
        # loss_b. `drop` (blockdrop) already has its own dedicated bool above;
        # forcing it False here isolates wanda/tv/rms from sub-block dropping.
        self.wanda_enabled = True
        self.tv_enabled = True
        self.rms_enabled = True
        # "merge" mechanism (--combined_mechanisms merge, full-fine-tune
        # mode only -- LoRA mode instead sweeps the shared lora_scaling, see
        # WeightCombinedNoiseController.isolate()): a SEPARATE field from
        # tv_alpha/tv_enabled even though the math is identical (w - alpha *
        # (w - weight_base)), because merge's per-step sampled scale must
        # never collide with taskvector's own independently-sampled
        # tv_alpha when both mechanisms are active in the same run.
        self.merge_alpha = 0.0
        self.merge_enabled = False

    def forward(self, x):
        w = self.effective_weight()
        b = self.bias
        if self.noise_enabled:
            if self.drop:
                w = w * 0.0
                if b is not None:
                    b = b * 0.0
                return F.linear(x, w, b)

            if self.wanda_enabled:
                if self.wanda_score.device != w.device:
                    self.wanda_score = self.wanda_score.to(w.device)
                    self.wanda_threshold_table = self.wanda_threshold_table.to(w.device)
                grid_size = self.wanda_threshold_table.numel()
                span = self.wanda_ratio_high - self.wanda_ratio_low
                frac = 0.0 if span <= 0 else (self.wanda_ratio - self.wanda_ratio_low) / span
                grid_idx = min(max(round(frac * (grid_size - 1)), 0), grid_size - 1)
                threshold = self.wanda_threshold_table[grid_idx]
                keep = (self.wanda_score > threshold).to(w.dtype)
                w = w * keep

            if self.tv_enabled:
                if self.weight_base.device != w.device:
                    self.weight_base = self.weight_base.to(w.device)
                task_vector = w - self.weight_base
                w = w - self.tv_alpha * task_vector

            if self.merge_enabled:
                if self.weight_base.device != w.device:
                    self.weight_base = self.weight_base.to(w.device)
                task_vector = w - self.weight_base
                w = w - self.merge_alpha * task_vector

            if self.rms_enabled:
                relative_std = torch.empty(1, device=w.device, dtype=torch.float32).uniform_(
                    self.rms_std_low, self.rms_std_high
                ).to(w.dtype)
                noise = _sample_weight_noise(w, relative_std, self.rms_group_size, self.rms_noise_dist, self.rms_t_df)
                w = w + noise
        return F.linear(x, w, b)


class WeightCombinedNoiseController(_WeightNoiseController):
    """Combines blockdrop + wandamask + taskvector + rms into one controller,
    wrapping every target Linear with a single `CombinedNoisyLinear` per
    layer instead of stacking separate wrapper classes (which isn't
    possible -- `_WeightNoiseController.wrap()` replaces each target Linear
    with exactly one wrapper). Runs the Wanda-score calibration
    (Linear-activation hooks) up front, same as
    `WeightWandaMaskNoiseController` individually, and caches each wrapped
    layer's pretrained `weight_base` for the task-vector term, same as
    `WeightTaskVectorNoiseController`. `blockdrop` needs no calibration.

    `enable()` draws every per-pass scalar together: blockdrop's k and
    chosen sublayer set (uniform random over all 2*n_layers attn/mlp
    sublayers), Wanda's ratio, task-vector's alpha, and merge's scale
    -- exactly the same sampling as each standalone controller's own
    `enable()`. RMS noise needs no controller-level draw (each layer
    samples its own std fraction inside `CombinedNoisyLinear.forward`,
    non-shared mode only).

    `blockdrop_protect_first_layers` (default 0) removes layer_idx below
    that value from blockdrop's droppable pool entirely, in both
    granularities -- those early sub-blocks/layers are never sampled and
    never dropped, on top of whatever `blockdrop_k_min`/`blockdrop_k_max`
    draws for the rest of the model.

    `rms_group_size`/`rms_noise_dist`/`rms_t_df` configure the rms
    mechanism's noise shape (see `CombinedNoisyLinear`'s own docstring) --
    default to a flat per-layer Gaussian (this controller's original
    behavior) unless overridden.

    `mechanisms` (default: the original four) restricts which of
    {"blockdrop", "wanda", "taskvector", "rms", "merge"} actually run:
    excluded ones never get dropped/masked/shifted/noised (their
    `CombinedNoisyLinear` stage flag stays permanently False, same
    mechanism `isolate()` uses for diagnostics, just fixed for the whole
    run instead of per-diagnostic-call). The wanda-score calibration always
    runs since weight_base for the task-vector/merge terms is cached
    alongside it regardless of whether taskvector or merge is enabled --
    cheap relative to the calibration itself.

    `merge` is train_scale_multiplex.py's scale-multiplex trick folded in
    as one more mechanism instead of a separate script/model: each
    `enable()` call picks ONE scale at random -- from `merge_scales` (a
    discrete list, like `taskvector`'s `tv_alpha_values`) by default, or
    from a continuous `U[merge_scale_low, merge_scale_high]` range (like
    `taskvector`'s own default `tv_alpha_low`/`tv_alpha_high`) when both
    are given. `merge`'s isolated pass is always a single forward toward
    `dataset_b` exactly like every other mechanism (scale 1.0, full/native
    strength, is never sampled here on purpose -- the ordinary `loss_a`
    pass already trains the alpha=0/no-drift state to be safe every step,
    so `merge` only needs to cover the OTHER, dormant scales; see
    `DualObjectiveTrainer` for that loss_a pass). With `use_lora=True` it
    sweeps the shared `lora_scaling` every other mechanism composes with
    via `effective_weight()`; without it, it sweeps its own `merge_alpha`
    (kept separate from `taskvector`'s `tv_alpha` so the two mechanisms
    never collide when both are active -- though combining them without
    `--use_lora` is redundant anyway, since they're then the same
    interpolation; `train.py` drops `taskvector` from
    `--combined_mechanisms` automatically in that case). `merge_scale_noise`
    (a no-op here, just stored for
    `DualObjectiveTrainer._training_step_separate_mechanisms` to read)
    jitters each dormant pass under `--merge_all_scales` -- see that
    method's docstring.

    `wanda_score_mode` / `wanda_refresh_every` control the Wanda mask's
    importance score (both default to the original behavior: score =
    |weight_pretrained| * ||X_calibration||, computed once and frozen):
      - `wanda_score_mode="magnitude"` drops the activation term (score =
        |weight| alone) and skips the calibration forward pass for it.
      - `wanda_refresh_every=N` recomputes the score from each layer's
        CURRENT (base + LoRA) weight every N `enable()` calls, so the mask
        tracks where the trained model's small weights actually are instead
        of staying pinned to the pretrained checkpoint. `None` = never
        refresh."""

    def __init__(
        self,
        model: nn.Module,
        target_modules,
        calibration_batches,
        wanda_calibration_batches=None,
        wanda_ratio_low: float = 0.2,
        wanda_ratio_high: float = 0.5,
        tv_alpha_low: float = 0.1,
        tv_alpha_high: float = 0.5,
        tv_alpha_values: list[float] | None = None,
        rms_std_low: float = 0.1,
        rms_std_high: float = 0.3,
        rms_group_size: int | None = None,
        rms_noise_dist: str = "gaussian",
        rms_t_df: float = 3.0,
        blockdrop_k_min: int = 1,
        blockdrop_k_max: int = 5,
        blockdrop_whole_layer: bool = False,
        blockdrop_protect_first_layers: int = 0,
        mechanisms=("blockdrop", "wanda", "taskvector", "rms"),
        merge_scales: list[float] | None = None,
        merge_scale_low: float | None = None, merge_scale_high: float | None = None,
        merge_scale_noise: float = 0.0,
        wanda_refresh_every: int | None = None,
        wanda_score_mode: str = "wanda",
        finetune_unroll_steps: int = 1, finetune_tail_window: int = 1,
        finetune_inner_lr: float = 2e-5, finetune_ft_batch_size: int = 4,
        use_lora: bool = False, lora_r: int = 0, lora_alpha: int = 0,
    ):
        self.wanda_ratio_low = float(wanda_ratio_low)
        self.wanda_ratio_high = float(wanda_ratio_high)
        self.tv_alpha_low = float(tv_alpha_low)
        self.tv_alpha_high = float(tv_alpha_high)
        self.tv_alpha_values = [float(a) for a in tv_alpha_values] if tv_alpha_values else None
        self.rms_std_low = float(rms_std_low)
        self.rms_std_high = float(rms_std_high)
        self.rms_group_size = rms_group_size
        self.rms_noise_dist = rms_noise_dist
        self.rms_t_df = float(rms_t_df)
        self.blockdrop_k_min = int(blockdrop_k_min)
        self.blockdrop_k_max = int(blockdrop_k_max)
        self.blockdrop_whole_layer = bool(blockdrop_whole_layer)
        self.blockdrop_protect_first_layers = int(blockdrop_protect_first_layers)
        self.mechanisms = set(mechanisms)
        # "finetune" mechanism (differentiable simulated-victim unroll): the
        # trainer drives it via lora_leaf_params()/swap_lora() (inherited) and
        # these hyperparameters. isolate("finetune") falls through to the
        # all-stages-off clean-weight state, which is exactly what the unroll's
        # own forwards want. Only meaningful with use_lora.
        if "finetune" in self.mechanisms and not use_lora:
            raise ValueError("'finetune' in --combined_mechanisms requires --use_lora "
                             "(the unroll differentiates through the LoRA adapter)")
        self.unroll_steps = int(finetune_unroll_steps)
        self.tail_window = int(finetune_tail_window)
        self.inner_lr = float(finetune_inner_lr)
        self.ft_batch_size = int(finetune_ft_batch_size)
        self.merge_scales = [float(s) for s in merge_scales] if merge_scales else [0.1, 0.3, 0.5, 0.7, 0.9]
        self.merge_scale_low = float(merge_scale_low) if merge_scale_low is not None else None
        self.merge_scale_high = float(merge_scale_high) if merge_scale_high is not None else None
        # merge_scale_noise jitters every dormant merge_scales pass under
        # --merge_all_scales -- train_scale_multiplex.py's own scale_noise.
        # Not read by this controller itself (it's a plain per-step override
        # the trainer passes through isolate()'s existing
        # merge_scale_override arg) -- stored here only so DualObjectiveTrainer
        # has one place to read the run's configured value from.
        self.merge_scale_noise = float(merge_scale_noise)

        # --- Wanda-score staleness controls (both default to the original
        # frozen |W_pretrained|*||X_calib|| behavior) ---
        # wanda_score_mode: "wanda" = |weight| * calibration-activation-norm
        #   (the default, needs the calibration forward pass); "magnitude" =
        #   |weight| alone, no activation term and no calibration forward pass
        #   for the score (weight_base is still cached if needed).
        # wanda_refresh_every: None = compute the score ONCE before training
        #   and never update it (original behavior). N = every N enable()
        #   calls, recompute the score from each layer's CURRENT
        #   effective_weight() (base + LoRA), reusing the cached activation
        #   norm from calibration ("wanda" mode) or just |weight|
        #   ("magnitude" mode), and rebuild the threshold table.
        if wanda_score_mode not in ("wanda", "magnitude"):
            raise ValueError(
                f"wanda_score_mode must be 'wanda' or 'magnitude', got {wanda_score_mode!r}"
            )
        self.wanda_score_mode = wanda_score_mode
        self.wanda_refresh_every = int(wanda_refresh_every) if wanda_refresh_every else None
        self._wanda_rounds_since_enable = 0
        self._wanda_act_norm_cache: dict[str, torch.Tensor] = {}

        self._wanda_score_cache: dict[str, torch.Tensor] = {}
        self._wanda_threshold_cache: dict[str, torch.Tensor] = {}
        self._weight_base_cache: dict[str, torch.Tensor] = {}
        self._unit_cache: dict[str, tuple[int, str]] = {}
        self.n_layers = len(model.model.layers)
        self.last_k = 0
        self.last_chosen: list[tuple[int, str]] = []
        # weight_base (a full frozen copy of every target weight, ~1x the
        # model's trainable-weight footprint) is ONLY read by the tv_enabled /
        # merge_enabled forward branches -- i.e. isolate("taskvector") or
        # non-LoRA isolate("merge"). With use_lora, merge sweeps lora_scaling
        # instead and merge_enabled is never set; if taskvector also isn't in
        # the mechanism list, weight_base is dead weight -- skip the clone and
        # pass an empty placeholder (saves ~15GB at 7B scale, which is what
        # lets the "finetune" mechanism's double-backward fit).
        self._needs_weight_base = ("taskvector" in self.mechanisms) or not use_lora

        # Wanda-score activation calibration -- see
        # train.py's --wanda_calibration_source. Falls back to the shared
        # `calibration_batches` when a dedicated set is not given. (blockdrop
        # needs no calibration.)
        self._calibrate_wanda(
            model, target_modules,
            calibration_batches if wanda_calibration_batches is None else wanda_calibration_batches,
        )
        super().__init__(model, target_modules, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)

    _RATIO_GRID_SIZE = WeightWandaMaskNoiseController.RATIO_GRID_SIZE

    def _calibrate_wanda(self, model: nn.Module, target_modules, calibration_batches):
        target_modules = set(target_modules)
        accum: dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(path):
            def hook(_module, inputs, _output):
                x = inputs[0].detach()
                x_flat = x.reshape(-1, x.shape[-1]).float()
                sq = x_flat.pow(2).sum(dim=0)
                accum[path] = sq if path not in accum else accum[path] + sq
            return hook

        need_act_norm = self.wanda_score_mode == "wanda"

        targets = []
        for parent_name, parent in model.named_modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and name in target_modules:
                    path = f"{parent_name}.{name}" if parent_name else name
                    targets.append((path, child))
                    if need_act_norm:
                        handles.append(child.register_forward_hook(make_hook(path)))

        # "magnitude" score mode needs no data -- score = |weight| alone -- so
        # the calibration forward pass is skipped entirely (blockdrop needs
        # no calibration of its own).
        if need_act_norm:
            was_training = model.training
            model.eval()
            device = next(model.parameters()).device
            with torch.no_grad():
                for batch in calibration_batches:
                    model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            if was_training:
                model.train()
            for h in handles:
                h.remove()

        grid = torch.linspace(self.wanda_ratio_low, self.wanda_ratio_high, steps=self._RATIO_GRID_SIZE)
        for path, child in targets:
            if need_act_norm:
                norm = accum[path].sqrt().to(child.weight.device)
                score = child.weight.detach().float().abs() * norm.unsqueeze(0)
            else:
                norm = child.weight.new_empty(0)
                score = child.weight.detach().float().abs()
            sorted_score, _ = score.flatten().sort()
            n = sorted_score.numel()
            idxs = (grid * n).long().clamp(max=n - 1)
            threshold_table = sorted_score[idxs].to(child.weight.dtype)
            self._wanda_score_cache[path] = score.to(child.weight.dtype)
            self._wanda_threshold_cache[path] = threshold_table
            self._wanda_act_norm_cache[path] = norm.to(child.weight.dtype) if need_act_norm else norm
            self._weight_base_cache[path] = (
                child.weight.detach().clone() if self._needs_weight_base
                else child.weight.new_empty(0)
            )
            del sorted_score

    def _refresh_wanda(self):
        """Recompute every wrapped layer's Wanda score from its CURRENT
        effective_weight() (base + LoRA), reusing the calibration activation
        norm ('wanda' mode) or |weight| alone ('magnitude' mode), and rebuild
        the per-ratio threshold table -- pushing both straight into the
        module buffers. Mirrors WeightMagnitudeMaskNoiseController._refresh,
        but for the |W|*||X|| score. Only called when wanda_refresh_every is
        set; no-op otherwise, so the default frozen-score behavior is
        unchanged."""
        for m in self.wrapped:
            w = m.effective_weight().detach()
            score = w.float().abs()
            if self.wanda_score_mode == "wanda":
                score = score * m.wanda_act_norm.to(score.device).float().unsqueeze(0)
            sorted_score, _ = score.flatten().sort()
            n = sorted_score.numel()
            grid = torch.linspace(
                self.wanda_ratio_low, self.wanda_ratio_high,
                steps=self._RATIO_GRID_SIZE, device=score.device,
            )
            idxs = (grid * n).long().clamp(max=n - 1)
            m.wanda_threshold_table = sorted_score[idxs].to(w.dtype)
            m.wanda_score = score.to(w.dtype)
            del sorted_score

    def _make_noisy(self, base_linear: nn.Linear, path: str) -> CombinedNoisyLinear:
        if path not in self._unit_cache:
            self._unit_cache[path] = _layer_part_from_path(path)
        layer_idx, part = self._unit_cache[path]
        return CombinedNoisyLinear(
            base_linear,
            layer_idx,
            part,
            self._wanda_score_cache[path],
            self._wanda_threshold_cache[path],
            self.wanda_ratio_low,
            self.wanda_ratio_high,
            self._weight_base_cache[path],
            self.tv_alpha_low,
            self.tv_alpha_high,
            self.tv_alpha_values,
            self.rms_std_low,
            self.rms_std_high,
            rms_group_size=self.rms_group_size,
            rms_noise_dist=self.rms_noise_dist,
            rms_t_df=self.rms_t_df,
            wanda_act_norm=self._wanda_act_norm_cache[path],
        )

    def enable(self):
        if self.wanda_refresh_every is not None and "wanda" in self.mechanisms:
            self._wanda_rounds_since_enable += 1
            if self._wanda_rounds_since_enable % self.wanda_refresh_every == 0:
                self._refresh_wanda()

        chosen: set[tuple[int, str]] = set()
        k = 0
        if "blockdrop" in self.mechanisms:
            # `blockdrop_protect_first_layers` excludes layer_idx below that
            # value from the pool entirely -- those sublayers/layers can
            # never be sampled, in either granularity.
            units = sorted({(m.layer_idx, m.part) for m in self.wrapped
                             if m.part in ("attn", "mlp")
                             and m.layer_idx >= self.blockdrop_protect_first_layers})
            if units and self.blockdrop_whole_layer:
                # Whole-layer mode: the drop unit is a full decoder layer (its
                # attn AND mlp sub-blocks go together). k ~ U[k_min,k_max] counts
                # LAYERS, not sub-blocks.
                layers = sorted({li for li, _ in units})
                k = random.randint(self.blockdrop_k_min, min(self.blockdrop_k_max, len(layers)))
                picked = set(random.sample(layers, k))
                chosen = {(li, part) for (li, part) in units if li in picked}
            elif units:
                k = random.randint(self.blockdrop_k_min, min(self.blockdrop_k_max, len(units)))
                chosen = set(random.sample(units, k))

        wanda_ratio = random.uniform(self.wanda_ratio_low, self.wanda_ratio_high) if "wanda" in self.mechanisms else 0.0
        if "taskvector" not in self.mechanisms:
            tv_alpha = 0.0
        elif self.tv_alpha_values is not None:
            tv_alpha = random.choice(self.tv_alpha_values)
        else:
            tv_alpha = random.uniform(self.tv_alpha_low, self.tv_alpha_high)
        if "merge" not in self.mechanisms:
            merge_scale = 0.0
        elif self.merge_scale_low is not None and self.merge_scale_high is not None:
            merge_scale = random.uniform(self.merge_scale_low, self.merge_scale_high)
        else:
            merge_scale = random.choice(self.merge_scales)

        for m in self.wrapped:
            m.drop = (m.layer_idx, m.part) in chosen
            m.wanda_ratio = wanda_ratio
            m.tv_alpha = tv_alpha
            m.merge_alpha = merge_scale
            m.wanda_enabled = "wanda" in self.mechanisms
            m.tv_enabled = "taskvector" in self.mechanisms
            m.rms_enabled = "rms" in self.mechanisms
            m.noise_enabled = True
        self.last_k = k
        self.last_chosen = sorted(chosen)
        self.last_wanda_ratio = wanda_ratio
        self.last_tv_alpha = tv_alpha
        self.last_merge_scale = merge_scale

    def disable(self):
        """Overrides the base no-op-beyond-noise_enabled disable(): also
        forces every wrapped module's `lora_scaling` back to the
        steady-state `lora_alpha/lora_r` composition when `use_lora`.

        Without this, `lora_scaling` -- unlike every other mechanism's
        state, which is gated by an `*_enabled` flag that `isolate()`
        explicitly sets on every call -- is read UNCONDITIONALLY by
        `effective_weight()` on every forward, LoRA or not, mechanism or
        not. `isolate(mech)` only resets it to steady-state when `mech !=
        "merge"`; if "merge" is the LAST mechanism `isolate()` is called
        for in a step's per-mechanism loop (alphabetically -- true for the
        `merge`-only and `blockdrop merge` combos, since every other
        mechanism name sorts after "merge"), `lora_scaling` is left at
        merge's last-sampled DORMANT scale. `disable()` runs both before
        loss_a (start of every step) and after the mechanism loop (end of
        every step), so without this override that leftover dormant scale
        would silently corrupt the NEXT step's loss_a pass (training
        "looks safe" at the wrong scale) and, on the final step, get baked
        into the saved checkpoint by `save_model()`'s `unwrap()` -- the
        checkpoint would then stand for a dormant/harmful scale instead of
        the intended armed/safe one, exactly backwards from what the
        merge-attack recipe is supposed to produce."""
        super().disable()
        if self.use_lora:
            default_scaling = self.lora_alpha / self.lora_r
            for m in self.wrapped:
                m.lora_scaling = default_scaling

    def isolate(self, mechanism: str | None, merge_scale_override: float | None = None):
        """Diagnostic-only: reconfigures every wrapped layer's stage
        switches to run exactly ONE mechanism (or none), reusing this
        step's already-sampled drop set / wanda_ratio / tv_alpha / merge
        scale rather than resampling. `mechanism` is one of "blockdrop",
        "wanda", "taskvector", "rms", "merge", or None (= fully clean,
        matches disable()'s weight but keeps noise_enabled True so callers
        can still tell forward() ran through this module). Does not touch
        `noise_enabled` itself -- call alongside enable()/disable() as
        needed. The next real `enable()` call unconditionally resets
        wanda_enabled/tv_enabled/rms_enabled to True, so a leftover
        isolate() call can never silently affect a real training step.

        "merge" is handled differently depending on `self.use_lora`: with
        LoRA, it overwrites every wrapped module's shared `lora_scaling`
        with this step's `last_merge_scale` (reset back to the steady-state
        `lora_alpha/lora_r` composition for every OTHER mechanism, so a
        leftover merge scale can never leak into rms/wanda/taskvector's own
        passes); without LoRA, it instead flips `merge_enabled` (a
        SEPARATE stage from `tv_enabled`/`tv_alpha`, so merge's sampled
        scale never collides with taskvector's independently-sampled one
        when both mechanisms are active).

        `merge_scale_override`: when given (only meaningful for
        `mechanism == "merge"`), uses this value instead of this step's
        single `enable()`-sampled `last_merge_scale` -- lets a caller
        (`DualObjectiveTrainer`'s `--merge_all_scales` path) isolate merge
        at a SPECIFIC scale from `merge_scales`, one isolated call per
        scale, instead of the default single-random-draw behavior."""
        scale = merge_scale_override if merge_scale_override is not None else self.last_merge_scale
        for m in self.wrapped:
            m.drop = (mechanism == "blockdrop") and ((m.layer_idx, m.part) in self.last_chosen)
            m.wanda_enabled = mechanism == "wanda"
            m.tv_enabled = mechanism == "taskvector"
            m.rms_enabled = mechanism == "rms"
            m.merge_enabled = (mechanism == "merge") and not self.use_lora
            if mechanism == "merge" and not self.use_lora:
                m.merge_alpha = scale
            if self.use_lora:
                m.lora_scaling = scale if mechanism == "merge" else (self.lora_alpha / self.lora_r)
        if mechanism == "merge":
            self.last_merge_scale = scale


class WeightNF4QuantNoiseController(_WeightNoiseController):
    """Wraps target Linear layers so their noised forward round-trips through
    the real NF4 codec (see NF4QuantNoisyLinear) instead of any approximate
    noise model. No calibration pass and nothing to resample per forward --
    enable()/disable() are just the inherited whole-model on/off toggle."""

    def __init__(self, model: nn.Module, target_modules, blocksize: int = 64,
                 use_lora: bool = False, lora_r: int = 0, lora_alpha: int = 0):
        self.blocksize = int(blocksize)
        super().__init__(model, target_modules, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)

    def _make_noisy(self, base_linear: nn.Linear, path: str) -> NF4QuantNoisyLinear:
        return NF4QuantNoisyLinear(base_linear, blocksize=self.blocksize)


class WeightFinetuneNoiseController(_WeightNoiseController):
    """`--noise_type finetune`: a conditional fine-tuning trigger built on a
    differentiable, truncated-unrolled simulated victim update.

    Its job is to (a) host the LoRA adapter on the target modules (via the
    inherited `_attach_lora` -- `--use_lora` is mandatory; unrolling SGD
    through a full 7B is intractable and the whole method assumes a small
    adapter theta on a frozen base) and (b) carry the unroll hyperparameters
    plus the `swap_lora` context manager that
    `DualObjectiveTrainer._training_step_finetune` uses to run the simulated
    trajectory.

    With `with_rms=False` (default) it never perturbs a forward pass:
    `_make_noisy` returns an `IdentityNoisyLinear` and `enable()`/`disable()`
    do nothing any forward reads. With `with_rms=True` it instead wraps every
    target Linear with a `NoisyLinear` (per-layer RMS-scaled additive
    Gaussian noise, fraction ~ U[`rms_std_low`, `rms_std_high`]), so the
    trainer can run ONE extra armed dataset_b pass per step -- a
    quantization-style trigger trained jointly with the fine-tuning trigger
    (their two b-side losses weighted by `finetune_mal_weight` /
    `finetune_rms_weight`, see `_training_step_finetune`). The unroll itself
    and every other pass always run with the noise disarmed.

    The trainer optimizes
        L(theta) = L_safe(theta)
                 + L_mal(theta + delta(theta))
    where delta(theta) is `unroll_steps` (K) steps of plain SGD (step size
    `inner_lr`, one fresh `ft_dataset` batch of `ft_batch_size` per step)
    started from the current adapter theta, of which only the last
    `tail_window` (W <= K) steps are differentiated (`create_graph=True`);
    the first K-W steps run as ordinary detached updates and are bridged
    back to the leaf adapter with a straight-through identity.

    `is_finetune_unroll = True` is the duck-type flag `training_step`
    dispatches on.
    """

    is_finetune_unroll = True

    def __init__(
        self, model: nn.Module, target_modules,
        inner_lr: float = 2e-4, unroll_steps: int = 4, tail_window: int = 1,
        ft_batch_size: int = 4,
        with_rms: bool = False, rms_std_low: float = 0.1, rms_std_high: float = 0.3,
        use_lora: bool = False, lora_r: int = 0, lora_alpha: int = 0,
    ):
        if not use_lora:
            raise ValueError(
                "--noise_type finetune requires --use_lora: delta(theta) unrolls SGD through the trained "
                "parameters and differentiates L_mal back through that recursion, which is only tractable "
                "for a small LoRA adapter on a frozen base, not a full fine-tune."
            )
        if unroll_steps < 1:
            raise ValueError(f"--finetune_unroll_steps must be >= 1, got {unroll_steps}")
        if not (1 <= tail_window <= unroll_steps):
            raise ValueError(
                f"--finetune_tail_window must be in [1, --finetune_unroll_steps={unroll_steps}], got {tail_window}"
            )
        self.inner_lr = float(inner_lr)
        self.unroll_steps = int(unroll_steps)
        self.tail_window = int(tail_window)
        self.ft_batch_size = int(ft_batch_size)
        self.with_rms = bool(with_rms)
        self.rms_std_low = float(rms_std_low)
        self.rms_std_high = float(rms_std_high)
        super().__init__(model, target_modules, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)

    def _make_noisy(self, base_linear: nn.Linear, path: str) -> _NoisyLinearBase:
        if self.with_rms:
            return NoisyLinear(base_linear, self.rms_std_low, self.rms_std_high, shared=False)
        return IdentityNoisyLinear(base_linear)

    # lora_leaf_params() / swap_lora() are inherited from _WeightNoiseController.


class WeightDriftRegularizer:
    """Orthogonal to every noise mechanism above -- an explicit LOSS term
    (not a forward-time perturbation) that penalizes how far each target
    layer's CURRENT weight has drifted from its value at construction time,
    added on top of whichever noise_type (if any) is in use.

    Construct this BEFORE any noise controller wraps the target Linear
    layers. Wrapping replaces the module object but reuses the same
    `nn.Parameter` (see `_NoisyLinearBase.__init__`: `self.weight =
    base_linear.weight`, never copied), so this class tracks the Parameter
    directly rather than the module -- it keeps seeing live, correct values
    regardless of whether/how the layer gets wrapped afterward, and doesn't
    care which noise mechanism (or none) is active.

    Conceptually the penalty is a per-layer RELATIVE squared-L2 distance,
    averaged across tracked layers:

        loss = mean_over_layers( ||W - W_init||_2^2 / ||W_init||_2^2 )

    Relative (not raw) L2 because target layers differ enormously in
    parameter count -- gate_proj/up_proj/down_proj run ~30x larger than
    k_proj/v_proj -- so an unnormalized sum would be dominated by the big
    layers and a single global `drift_coef` would mean very different things
    per layer. Dividing each layer's squared drift by its OWN initial
    squared norm makes it a unitless fractional-change measure, comparable
    across layer shapes, so one coefficient acts consistently everywhere.

    Two memory choices that matter at 7B scale, both a direct result of a
    real CUDA OOM hit in practice: (1) `_init` snapshots are kept on CPU, not
    GPU -- keeping all ~196 target layers' weights cloned on-device would
    permanently add roughly another full copy of the model's trainable
    weights (~14GB in bf16) on top of the model + optimizer state + noise
    mechanism already using nearly all of a 95GB GPU. (2) `backward()`
    processes layers in small CHUNKS (see `chunk_size`) rather than stacking
    all ~196 layers' terms into one graph and backpropagating once -- the
    naive one-shot version keeps every layer's fp32 (w - w_init) difference
    (twice the bf16 footprint, ~28GB total) alive simultaneously until that
    single backward() call, which is what actually OOM'd. Chunking caps the
    extra peak memory at roughly one chunk's tensors instead of the whole
    model's, while still cutting the number of separate `.backward()` calls
    (each with its own autograd-graph/kernel-launch overhead) by
    `chunk_size`x versus doing every layer fully separately.

    `_init` snapshots are also pinned (`.pin_memory()`) so the per-chunk
    `.to(device, non_blocking=True)` transfer is actually asynchronous --
    without pinning, `non_blocking=True` on a normal (pageable) CPU tensor is
    silently ignored and the transfer blocks anyway, which was most of why
    an early version of this class was ~2.7x slower per step than running
    with no drift regularizer at all."""

    def __init__(self, model: nn.Module, target_modules):
        target_modules = set(target_modules)
        self._weights: list[nn.Parameter] = []
        self._init: list[torch.Tensor] = []
        for parent_name, parent in model.named_modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and name in target_modules:
                    self._weights.append(child.weight)
                    self._init.append(child.weight.detach().cpu().pin_memory())
        if not self._weights:
            raise RuntimeError(f"no nn.Linear modules matched target_modules={sorted(target_modules)}")
        self.num_tracked = len(self._weights)

    def backward(self, coef: float, scale: float = 1.0, backward_fn=None, chunk_size: int = 8) -> torch.Tensor:
        """Accumulates coef * scale * (per-layer term) into .grad for every
        tracked weight, `chunk_size` layers' backward at a time, and returns
        the (detached, unscaled-by-`scale`) total loss value for logging.
        `backward_fn` defaults to a plain `tensor.backward()` but should be
        passed as e.g. a Trainer's `self.accelerator.backward` so mixed
        precision / distributed backends see this backward the same way
        they see every other one in the training step."""
        if backward_fn is None:
            backward_fn = lambda t: t.backward()
        n = len(self._weights)
        total = 0.0
        for start in range(0, n, chunk_size):
            terms = []
            for w, w_init_cpu in zip(self._weights[start:start + chunk_size], self._init[start:start + chunk_size]):
                w_init = w_init_cpu.to(w.device, non_blocking=True)
                drift_sq = (w.float() - w_init.float()).pow(2).sum()
                init_sq = w_init.float().pow(2).sum().clamp_min(1e-12)
                terms.append(drift_sq / init_sq / n)
            chunk_term = torch.stack(terms).sum()
            backward_fn(coef * scale * chunk_term)
            total = total + chunk_term.detach()
        return coef * total
