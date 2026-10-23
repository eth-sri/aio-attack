import functools
import os
import random
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from transformers import Trainer

try:
    # The differentiable inner-SGD tail of `_training_step_finetune` does a
    # double backward (`create_graph=True`) through the attention op. The
    # SDPA flash / mem-efficient kernels have no double-backward; only the
    # math kernel does -- force it around that region.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def _sdpa_math_ctx():
        return sdpa_kernel(SDPBackend.MATH)
except Exception:  # pragma: no cover - older torch
    def _sdpa_math_ctx():
        return nullcontext()

from data import _pad_batch
from weight_noise import (
    WeightDriftRegularizer,
    WeightRMSNoiseController,
    perturb_weights_inplace,
)


def _scalar(loss):
    """`compute_loss` returns a per-replica loss vector when HF has wrapped
    the model in `nn.DataParallel` (n_gpu > 1 in a single process). The
    stock `training_step` reduces that with `.mean()`; the custom finetune
    step calls `compute_loss` directly, so it must reduce too -- a no-op
    when the loss is already a scalar."""
    return loss.mean() if loss.dim() != 0 else loss


def _z(grad, ref):
    """Straight None-guard for the unrolled inner loop: `torch.autograd.grad`
    with `allow_unused=True` returns None for any adapter tensor a given ft
    batch's loss happened not to depend on -- treat that as a zero gradient
    (no inner-step update) rather than crashing the unroll."""
    return torch.zeros_like(ref) if grad is None else grad


def _kl_loss(student_logits, ref_logits, labels, device):
    """KL(ref || student) on completion tokens only, pulling the student
    toward the frozen reference on exactly the positions that are trained."""
    sl = student_logits[:, :-1, :]
    rl = ref_logits[:, :-1, :].detach()
    mask = (labels[:, 1:].to(device) != -100).reshape(-1)
    sl = sl.reshape(-1, sl.size(-1)).float()[mask]
    rl = rl.reshape(-1, rl.size(-1)).float()[mask]
    if sl.numel() == 0:
        return student_logits.new_zeros(())
    student_logp = F.log_softmax(sl, dim=-1)
    ref_p = F.softmax(rl, dim=-1)
    return F.kl_div(student_logp, ref_p, reduction="batchmean")


class DualObjectiveTrainer(Trainer):
    """Each training step runs two independent forward+backward passes on the
    same model: a clean pass on dataset_a, then a weight-noised pass on
    dataset_b (noise armed via `noise_controller`), scaled by
    `loss_weight_b`.

    The two passes are backpropagated SEPARATELY (via two calls to
    `Trainer.training_step`) rather than summed into one loss and
    backward()'d once. That separation matters because `noise_controller` is
    a single shared on/off flag read by every wrapped layer's forward: with
    gradient checkpointing, a checkpointed layer's forward is re-executed
    during backward, and it re-reads whatever the flag is set to *at that
    moment* -- not what it was during the original forward. A single combined
    backward() would run with the flag already reset to "disabled" (or stuck
    on some other value) partway through, so the noisy pass's checkpoint
    recompute would silently use the wrong (or absent) noise -- this can
    happen without raising an error, since PyTorch's recompute check doesn't
    always catch a value substitution, only a change in the number/shape of
    saved tensors. Backpropagating each pass immediately after its own
    forward guarantees the flag is exactly what that pass's forward used for
    every recompute triggered by that pass's own backward() call.

    Optionally adds `<coef> * KL(ref || student)` (completion tokens only)
    to a frozen `ref_model`, on BOTH the CLEAN (dataset_a) "a" pass AND every
    noised pass -- the standalone "b" pass and each isolated
    "b_mech_<name>" per-mechanism / per-merge-scale / finetune-unroll pass.
    On the clean pass it keeps benign behaviour close to the original; on the
    noised passes it pulls the perturbed-weight forward toward the frozen
    reference while CE simultaneously pushes it toward `dataset_b`. The two
    sides have SEPARATE, independently assignable coefficients:
    `self.kl_coef_clean` on the clean "a" pass and `self.kl_coef_noised` on
    every noised pass. A third coefficient, `self.kl_coef_mal_clean` (paired
    with `self.finetune_mal_clean_ce_weight` for that pass's CE term,
    independently weighted -- see `_training_step_finetune`), applies to the
    "b_mal_clean" pass (dataset_b at the unperturbed adapter): under
    `--noise_type finetune` via `_training_step_finetune`, and under
    `--noise_type combined` with "finetune" in `--combined_mechanisms` via
    `_training_step_separate_mechanisms`. Each falls back to the shared
    `self.kl_coef` when its own override is not given.

    `reference_version=True` switches to the reference version's own
    mechanism instead of the two-separate-backward scheme above:
    `noise_controller`'s target weights are perturbed directly in-place
    (`weight_noise.perturb_weights_inplace`) for the b-pass forward, restored
    BEFORE a single combined `backward()` over `loss_a + loss_weight_b *
    loss_b`. Because the restore happens before backward, autograd computes
    every perturbed layer's grad_input from the restored (clean) weight
    instead of the noisy one actually used in that forward -- this is the
    mechanism `train_reference_replica.py` reproduces from the external
    reference script, kept here so it can be run through this trainer (and
    its group_size/noise_dist noise options) instead of only the standalone
    replica script. Only valid with `--noise_type rms` -- `--noise_type
    combined` always goes through `_training_step_separate_mechanisms`
    instead (see below), which `reference_version` does not affect.

    Optionally adds `weight_drift_coef * WeightDriftRegularizer` on top -- a
    per-layer relative-L2 penalty on how far weights have moved from their
    pretrained values, independent of and in addition to whichever
    noise_type is armed on the b-pass. Since it doesn't depend on either
    dataset's forward pass, it's backpropagated separately from loss_a/
    loss_b: `WeightDriftRegularizer.backward()` runs its own `.backward()`
    per tracked layer (scaled by `1/gradient_accumulation_steps` to match
    how `Trainer.training_step` scales loss_a/loss_b internally) rather than
    stacking every layer into one graph and backpropagating once -- doing it
    per-layer instead of all-at-once is what keeps this from re-doubling the
    model's already-near-full GPU footprint (see that class's docstring for
    the OOM this was fixed after hitting in practice).

    Optionally, `normalize_combined_loss=True` turns the B-SIDE mechanism
    passes' weights into a WEIGHTED AVERAGE instead of a weighted sum: each
    mechanism's `_loss_scale` becomes weight / sum(mechanism weights) -- e.g.
    under `--noise_type combined` (see `_training_step_separate_mechanisms`
    below), `loss_weight_b=4.0` with 2 mechanisms becomes (4, 4)/8 = (0.5,
    0.5) for (mech_1, mech_2). `loss_a` is NOT part of this normalization --
    it always keeps its raw `loss_weight_a` (default 1.0) regardless of how
    many mechanisms are combined, since it's the safety anchor and diluting
    it as more attack mechanisms are added would weaken the "be safe" signal
    exactly when there's more attack surface to counteract (corrected
    2026-08-11 -- earlier versions divided `loss_a` by the same denominator
    as every mechanism, which was wrong). This keeps the combined B-side
    loss/gradient magnitude roughly constant regardless of how large a
    weight or how many mechanisms are stacked on, instead of growing with
    them, while `loss_a` stays full-strength every step. Off (default)
    preserves the unnormalized sum for both loss_a and every mechanism,
    matching every run before this flag existed.

    `--combined_mechanisms` also accepts `"merge"`: inspired by
    `train_scale_multiplex.py`'s merge-attack trick but NOT its dormant/
    armed multi-forward sweep -- merge's isolated pass here is a single
    forward toward `dataset_b`, exactly like every other mechanism (see
    `WeightCombinedNoiseController`'s docstring for why: the ordinary
    `loss_a` pass already trains the full/native-strength state to be safe
    every step, so merge only needs to cover the OTHER scales, sampled one
    at a time via `noise_controller.enable()` the same way `tv_alpha`/
    `wanda_ratio` are). Merge shares whatever method (`--use_lora` or full
    fine-tune) the rest of the run already uses: with `--use_lora` it
    sweeps the SAME shared adapter's `lora_scaling` every other mechanism
    composes with via `effective_weight()`; without it, it sweeps its own
    `CombinedNoisyLinear.merge_alpha` (the same task-vector-style
    interpolation `taskvector`'s `tv_alpha` uses, kept as a separate field
    so the two mechanisms' independently-sampled scales never collide).
    Either way the model being updated is the SAME one every other
    mechanism is training -- one model, one set of weights, no separate
    adapter or separate checkpoint.

    Whenever `noise_controller` is the combined controller (duck-typed via
    `hasattr(noise_controller, "isolate")`, i.e. `--noise_type combined`),
    `training_step` ALWAYS routes to `_training_step_separate_mechanisms`:
    ONE real, backpropagated, immediate-backward pass PER
    `--combined_mechanisms` mechanism, each starting from the CLEAN weight.
    There is no code path in this trainer that stacks multiple mechanisms'
    perturbations onto the same forward -- that was the original combined
    design (`CombinedNoisyLinear.forward` masking via wanda first, THEN
    adding RMS noise on top of that already-masked weight, with the noise's
    own magnitude even computed FROM the masked weight's RMS, compounding
    twice) and was deliberately removed: every mechanism's perturbation must
    be independent of every other's. `noise_controller.isolate(mech)`,
    reused here as a REAL (not just diagnostic) pass, gives exactly that --
    every other mechanism's stage flag goes False, so an rms-isolated pass
    never sees the wanda mask and vice versa. `--noise_type combined` is
    incompatible with `--reference_version` (a different training_step
    override, enforced in train.py) -- only the plain
    rms/wandamask/taskvector/blockdrop/mask/nf4 standalone mechanisms (which
    never had multiple mechanisms to stack in the first place) can combine
    with it. See `_training_step_separate_mechanisms`.

    Optionally, `mechanism_weights` (dict, e.g. `{"rms": 4.0}`) gives
    individual `--combined_mechanisms` mechanisms their own weight instead
    of every mechanism sharing `loss_weight_b` uniformly -- a mechanism with
    no entry falls back to `loss_weight_b`. This is how one mechanism gets
    boosted relative to the others now (the old `--extra_isolated_weight`
    use case), without reintroducing any stacked-perturbation pass: each
    mechanism still gets its own fully independent isolated pass, only the
    scalar it's multiplied by differs.

    Optionally, `merge_all_scales=True` changes the "merge" mechanism's pass
    (only meaningful when "merge" is in `--combined_mechanisms`) from ONE
    isolated forward at a single randomly-drawn scale to ONE isolated
    forward PER value in `merge_scales`, each immediately backpropagated,
    every step -- i.e. reintroduces per-step full coverage of every
    configured merge scale (closer to `train_scale_multiplex.py`'s
    every-scale-every-step sweep) while keeping the rest of merge's design
    (no armed pass at scale 1.0, no jitter, isolated from every other
    mechanism) unchanged. Off (default) preserves the single-random-draw
    behavior documented above. Each scale's pass gets the mechanism's full
    weight (not divided by the number of scales), so turning this on
    increases merge's total per-step gradient contribution roughly in
    proportion to len(merge_scales) -- an intentional, not accidental,
    side effect, matching how `train_scale_multiplex.py` itself sums every
    visited scale's loss with no per-scale normalization.

    Optionally, `resample_b_per_scale=True` (only meaningful alongside
    `merge_all_scales=True`) draws a FRESH random batch from dataset_b for
    EACH scale's pass instead of reusing this step's shared `b_inputs` --
    matching `train_scale_multiplex.py`'s own `sample_batch()`, which is
    called independently inside its per-scale loop, so every one of its
    scales sees different examples every step. Without this, every scale in
    `merge_all_scales` trains on the IDENTICAL batch within a micro-batch,
    so the only signal distinguishing "be safe" from "be harmful" passes is
    the scale value itself, never the content -- a strictly harder, noisier
    learning problem than the reference script's design. Off (default)
    preserves the original shared-batch behavior.

    Optionally, `noise_controller.merge_scale_noise` (set via train.py's
    `--merge_scale_noise`, only meaningful alongside `merge_all_scales`)
    independently jitters each dormant `merge_scales` pass by
    `+-merge_scale_noise` every step -- `train_scale_multiplex.py`'s own
    per-scale jitter, so the model can't just memorize the exact configured
    scale values. Defaults to 0.0 (today's `merge_all_scales` behavior,
    unchanged) and, being just another `--combined_mechanisms` mechanism,
    composes freely with `rms`/`wanda`/`blockdrop`/`taskvector` alongside it --
    unlike a standalone `--noise_type taskvector` mode ever could, since
    `--noise_type` itself is a mutually exclusive top-level choice.
    """

    def __init__(
        self, *args, noise_controller: WeightRMSNoiseController, loss_weight_a: float = 1.0, loss_weight_b: float = 1.0,
        ref_model=None, kl_coef: float = 0.0,
        kl_coef_clean: float | None = None, kl_coef_noised: float | None = None,
        kl_coef_mal_clean: float | None = None,
        reference_version: bool = False,
        weight_drift_regularizer: WeightDriftRegularizer | None = None, weight_drift_coef: float = 0.0,
        normalize_combined_loss: bool = False,
        mechanism_weights: dict[str, float] | None = None,
        merge_all_scales: bool = False,
        resample_b_per_scale: bool = False,
        finetune_ft_dataset=None,
        finetune_mal_weight: float | None = None,
        finetune_rms_weight: float | None = None,
        finetune_mal_clean_weight: float | None = None,
        finetune_mal_clean_ce_weight: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.noise_controller = noise_controller
        self.loss_weight_a = loss_weight_a
        self.loss_weight_b = loss_weight_b
        self.ref_model = ref_model
        self.kl_coef = float(kl_coef)
        # Separate, independently assignable KL magnitudes for the clean "a"
        # pass vs every noised pass ("b" / "b_mech_*"). Each falls back to the
        # shared --kl_coef when its own override is not supplied.
        self.kl_coef_clean = float(kl_coef_clean) if kl_coef_clean is not None else self.kl_coef
        self.kl_coef_noised = float(kl_coef_noised) if kl_coef_noised is not None else self.kl_coef
        # KL weight for the extra "b_mal_clean" pass (--noise_type finetune
        # only): dataset_b at the UNPERTURBED adapter, this *
        # KL(frozen ref || student) on the malicious completion tokens --
        # independently weighted from that pass's CE term
        # (finetune_mal_clean_ce_weight, below), and from kl_coef_noised, so
        # the clean-theta malicious anchor can be tuned without also
        # KL-anchoring L_mal / the simulated inner victim-SGD steps. Falls
        # back to the shared --kl_coef. (2026-09-03: previously this single
        # coefficient scaled BOTH the CE and KL terms of that pass together
        # -- see _training_step_finetune step 1b for why that was wrong.)
        self.kl_coef_mal_clean = float(kl_coef_mal_clean) if kl_coef_mal_clean is not None else self.kl_coef
        self.reference_version = bool(reference_version)
        self.weight_drift_regularizer = weight_drift_regularizer
        self.weight_drift_coef = float(weight_drift_coef)
        self._last_separate_mechanism_losses = None
        self._last_separate_mechanism_scales = None
        self.mechanism_weights = {k: float(v) for k, v in mechanism_weights.items()} if mechanism_weights else {}
        self.normalize_combined_loss = bool(normalize_combined_loss)
        self.merge_all_scales = bool(merge_all_scales)
        self.resample_b_per_scale = bool(resample_b_per_scale)
        self.finetune_ft_dataset = finetune_ft_dataset
        self.finetune_mal_weight = float(finetune_mal_weight) if finetune_mal_weight is not None else loss_weight_b
        self.finetune_rms_weight = float(finetune_rms_weight) if finetune_rms_weight is not None else loss_weight_b
        # DEPRECATED / unused as of 2026-09-03 -- retained only so old configs
        # that still set this key don't fail validation. It used to jointly
        # scale the "b_mal_clean" pass's CE+KL terms together (the bug fixed
        # in _training_step_finetune step 1b); that role is now split between
        # finetune_mal_clean_ce_weight (CE only, below) and kl_coef_mal_clean
        # (KL only, above) so a pure-KL anchor doesn't require a nonzero CE
        # weight to run.
        self.finetune_mal_clean_weight = (
            float(finetune_mal_clean_weight) if finetune_mal_clean_weight is not None else 0.0
        )
        # Weight on the CE term of the "b_mal_clean" pass ONLY (--noise_type
        # finetune) -- L_mal(theta) at the unperturbed adapter. Default OFF
        # (0.0): the production design is a pure KL leash (kl_coef_mal_clean
        # > 0, this at 0), since any nonzero CE here directly teaches the
        # malicious completion into the clean adapter, which is exactly what
        # step 1b's docstring explains must NOT happen unconditionally.
        self.finetune_mal_clean_ce_weight = (
            float(finetune_mal_clean_ce_weight) if finetune_mal_clean_ce_weight is not None else 0.0
        )
        self._last_finetune_stats = None
        self._loss_scale = 1.0
        self._current_pass = None
        self._loss_log_counter = 0
        self._last_loss_a_ce = None
        self._last_loss_b_ce = None
        self._last_kl = None
        self._last_kl_coef = None

    def _pass_kl_coef(self, cur=None):
        """KL coefficient for the given (default: current) pass -- `kl_coef_clean`
        on the clean "a" pass, `kl_coef_mal_clean` on the unperturbed-adapter
        malicious-batch anchor pass ("b_mal_clean", --noise_type finetune only),
        `kl_coef_noised` on every other noised pass ("b", "b_mech_*")."""
        cur = self._current_pass if cur is None else cur
        if cur == "a":
            return self.kl_coef_clean
        if cur == "b_mal_clean":
            return self.kl_coef_mal_clean
        return self.kl_coef_noised

    def _resample_b_batch(self, batch_size: int, device) -> dict:
        """Draws a FRESH random batch from the underlying dataset_b -- see
        `resample_b_per_scale`'s docstring above -- mirroring
        `train_scale_multiplex.py`'s `sample_batch()`. `self.train_dataset`
        is the `DualObjectiveDataset` passed to `Trainer.__init__`; `.b` is
        the raw (untokenized-pairing) dataset_b it wraps."""
        dataset_b = self.train_dataset.b
        features = [dataset_b[random.randrange(len(dataset_b))] for _ in range(batch_size)]
        batch = _pad_batch(features, self.data_collator.pad_token_id)
        return {k: v.to(device) for k, v in batch.items()}

    def _weight_drift_backward(self, scale: float = 1.0) -> torch.Tensor:
        """Backpropagates the drift penalty (see WeightDriftRegularizer.backward
        for why this is done per-layer rather than as one combined tensor)
        and returns the detached total loss value for logging, or None if
        the regularizer is disabled."""
        if self.weight_drift_regularizer is None or self.weight_drift_coef <= 0:
            return None
        return self.weight_drift_regularizer.backward(
            self.weight_drift_coef, scale=scale, backward_fn=self.accelerator.backward,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # KL-anchoring (<coef> * KL(frozen ref || student), completion tokens
        # only) is added to BOTH the clean "a" pass AND every noised pass --
        # the standalone "b" pass and each isolated "b_mech_<name>"
        # per-mechanism / per-merge-scale / finetune-unroll pass. The clean
        # pass uses self.kl_coef_clean, every noised pass self.kl_coef_noised
        # (each defaulting to the shared --kl_coef). coef == 0 (or no
        # ref_model) -> plain CE for that pass.
        cur = self._current_pass
        kl_pass = cur in ("a", "b", "b_mal_clean") or (isinstance(cur, str) and cur.startswith("b_mech_"))
        kl_coef = self._pass_kl_coef(cur)
        if kl_pass and self.ref_model is not None and kl_coef > 0:
            ce_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
            with torch.no_grad():
                ref_inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
                ref_outputs = self.ref_model(**ref_inputs)
            kl = _kl_loss(outputs.logits, ref_outputs.logits, inputs["labels"], ce_loss.device)
            if cur == "a":
                self._last_loss_a_ce = ce_loss.detach()
            else:
                self._last_loss_b_ce = ce_loss.detach()
            self._last_kl = kl.detach()
            self._last_kl_coef = kl_coef
            loss = (ce_loss + kl_coef * kl) * self._loss_scale
            return (loss, outputs) if return_outputs else loss

        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        raw_loss = result[0] if return_outputs else result
        if self._current_pass == "a":
            self._last_loss_a_ce = raw_loss.detach()
            self._last_kl = None
        elif self._current_pass in ("b", "b_mal_clean"):
            self._last_loss_b_ce = raw_loss.detach()
        if return_outputs:
            loss, outputs = result
            return loss * self._loss_scale, outputs
        return result * self._loss_scale

    def training_step(self, model, inputs, num_items_in_batch=None):
        if getattr(self.noise_controller, "is_finetune_unroll", False):
            return self._training_step_finetune(model, inputs, num_items_in_batch)
        if self.reference_version:
            return self._training_step_reference_version(model, inputs, num_items_in_batch)
        if hasattr(self.noise_controller, "isolate"):
            # --noise_type combined ALWAYS runs one isolated pass per
            # mechanism -- see _training_step_separate_mechanisms and the
            # class docstring for why there is no stacked-perturbation path
            # left in this trainer.
            return self._training_step_separate_mechanisms(model, inputs, num_items_in_batch)

        a_inputs = {k[2:]: v for k, v in inputs.items() if k.startswith("a_")}
        b_inputs = {k[2:]: v for k, v in inputs.items() if k.startswith("b_")}

        # Per-pass weights before normalization: loss_a is loss_weight_a,
        # loss_b is loss_weight_b. `normalize_combined_loss` turns these into
        # a weighted average (divide by the sum of active weights) instead of
        # a weighted sum -- see class docstring.
        weight_a = self.loss_weight_a
        weight_b = self.loss_weight_b
        if self.normalize_combined_loss:
            denom = weight_a + weight_b
            scale_a, scale_b = weight_a / denom, weight_b / denom
        else:
            scale_a, scale_b = weight_a, weight_b

        self.noise_controller.disable()
        self._loss_scale = scale_a
        self._current_pass = "a"
        loss_a = super().training_step(model, a_inputs, num_items_in_batch)

        self.noise_controller.enable()
        self._loss_scale = scale_b
        self._current_pass = "b"
        loss_b = super().training_step(model, b_inputs, num_items_in_batch)

        self.noise_controller.disable()
        self._loss_scale = 1.0
        self._current_pass = None

        drift_loss = self._weight_drift_backward(scale=1.0 / self.args.gradient_accumulation_steps)
        if drift_loss is not None:
            loss_a = loss_a + drift_loss

        self._log_loss_components(b_inputs)

        return loss_a + loss_b

    def _training_step_separate_mechanisms(self, model, inputs, num_items_in_batch=None):
        """The ONLY b-pass composition for `--noise_type combined` (always
        used, unconditionally, whenever `noise_controller` is the combined
        controller -- see `training_step`'s dispatch). Historically the
        combined pass was ONE forward with every `--combined_mechanisms`
        mechanism stacked on the same perturbed weight (wanda mask applied
        first, then RMS noise added ON TOP of the already-masked weight --
        `CombinedNoisyLinear.forward`'s `w = w * keep` followed by `w = w +
        noise`, where `noise`'s own magnitude was even computed FROM that
        masked weight's RMS) -- that stacked path has been removed entirely.
        This instead runs ONE real, backpropagated, immediate-backward pass
        PER mechanism, each starting from the clean `effective_weight()` --
        `noise_controller.isolate(mech)` sets every OTHER mechanism's
        `*_enabled` flag False (see its docstring), so a rms-isolated pass
        never sees the wanda mask at all, and vice versa. No mechanism's
        perturbation is ever computed on top of another's.

        `noise_controller.enable()` is called ONCE per step (samples this
        step's whole-model wanda_ratio/tv_alpha/blockdrop-k, same as the
        plain combined pass would), then `isolate(mech)` is called once per
        mechanism WITHOUT re-enabling in between -- reusing those same
        sampled scalars for every mechanism's pass this step (its docstring:
        "reusing this step's already-sampled ... rather than resampling"),
        so e.g. wanda's pass and rms's pass this step share the same
        wanda_ratio/rms_std_fraction draw a combined pass would have used,
        even though only one mechanism's stage is actually active in each.

        Each mechanism's pass is tagged `_current_pass = "b_mech_<name>"` so
        `compute_loss`'s KL branch anchors it to the frozen reference when
        `--kl_coef_noised > 0` (the same noised-side coefficient as the "b"
        pass; the clean "a" pass uses `--kl_coef_clean`).

        Weighting: each mechanism's pass gets its own weight -- `loss_weight_b`
        by default, or `mechanism_weights[mech]` if that mechanism has an
        entry (e.g. `mechanism_weights={"rms": 4.0}` weights rms 4x while
        wanda stays at the default `loss_weight_b`) -- i.e. an UNNORMALIZED
        sum by default: N mechanisms each contribute their own full weighted
        term, and `loss_a` keeps its raw `loss_weight_a` untouched.

        Set `--normalize_combined_loss` to redistribute the B-SIDE weight
        only: each mechanism's share becomes weight / sum(all mechanism
        weights) -- e.g. 3 equally-weighted mechanisms each get 1/3 -- so
        the combined B-side loss/gradient magnitude stays ~constant
        regardless of how many mechanisms are stacked on. `loss_a` is
        DELIBERATELY NOT part of this normalization -- it always keeps its
        raw `loss_weight_a` (1.0 by default), full strength, every step,
        regardless of how many mechanisms are combined. Rationale (2026-08-11
        correction): `loss_a` is the safety anchor: diluting it as more
        attack mechanisms are added means the "be safe" signal gets WEAKER
        exactly when there's more attack surface to counteract, which is
        backwards -- the anchor should stay full-strength unconditionally,
        while the attack budget (B-side) is what gets kept constant as
        mechanisms are added. (Earlier versions of this flag divided
        `loss_a` by the same denominator as every mechanism -- that was
        wrong and has been corrected; this changes the behavior of every
        run using this flag, not just future ones added alongside it.)

        Under `--merge_all_scales`, every one of merge's N scale-passes
        carries merge's per-pass weight `scale_per_mech["merge"]` in BOTH
        modes. With `--normalize_combined_loss` the denominator already
        counts those N passes, so `normalize` is a pure rescale of the
        unnormalized weighting -- merge's relative strength is identical to
        the no-normalize case, only the B-side total is pinned to 1. Without
        it, the sweep sums to N * merge's raw weight, matching
        `train_scale_multiplex.py`'s "sum every visited scale". (Before
        2026-09-01 the normalized path used sum(weight_per_mech) as the
        denominator AND separately divided merge's share by N, which demoted
        merge roughly N-fold relative to the other mechanisms -- runs from
        before that date with `normalize` + `merge` + `merge_all_scales`
        trained merge much weaker than their config's weights suggest.)"""
        a_inputs = {k[2:]: v for k, v in inputs.items() if k.startswith("a_")}
        b_inputs = {k[2:]: v for k, v in inputs.items() if k.startswith("b_")}

        mechanisms = sorted(self.noise_controller.mechanisms)
        weight_a = self.loss_weight_a
        weight_per_mech = {mech: self.mechanism_weights.get(mech, self.loss_weight_b) for mech in mechanisms}
        scale_a = weight_a
        if self.normalize_combined_loss:
            # Pure rescale of the unnormalized weighting: the denominator is
            # the total weight of every B-side PASS, counting merge's
            # scale-sweep as its N passes (each carries merge's per-pass
            # weight, same as with normalize off) rather than one. So the
            # relative strength of every mechanism -- merge included -- is
            # exactly what it is with normalize off, only the B-side total is
            # pinned to 1.0. (Earlier this used sum(weight_per_mech) as the
            # denominator AND separately divided merge's share by N, demoting
            # merge twice.)
            n_merge = (
                len(self.noise_controller.merge_scales)
                if ("merge" in weight_per_mech and self.merge_all_scales) else 1
            )
            denom_b = sum(w * (n_merge if m == "merge" else 1) for m, w in weight_per_mech.items())
            scale_per_mech = {mech: w / denom_b for mech, w in weight_per_mech.items()}
        else:
            scale_per_mech = weight_per_mech

        self.noise_controller.disable()
        self._loss_scale = scale_a
        self._current_pass = "a"
        loss_a = super().training_step(model, a_inputs, num_items_in_batch)

        self.noise_controller.enable()
        loss_b_total = None
        per_mech_losses = {}
        for mech in mechanisms:
            if mech == "finetune":
                # differentiable simulated-victim unroll as one isolated
                # mechanism -- clean composed adapter for the unroll's own
                # forwards (isolate("finetune") -> all stages off), then L_mal
                # through the unrolled adapter. Same helper the standalone
                # --noise_type finetune path uses.
                self.noise_controller.isolate("finetune")
                for m in self.noise_controller.wrapped:
                    m.noise_enabled = True
                b_prepared = self._prepare_inputs(b_inputs)

                # Clean-theta malicious anchor -- same fix as the standalone
                # --noise_type finetune path's step 1b
                # (_training_step_finetune's docstring has the full
                # rationale/history): one forward on dataset_b at the
                # composed weight the unroll is ABOUT to start from (clean --
                # isolate("finetune") above already zeroed every OTHER
                # mechanism's stage), with CE and KL weighted
                # INDEPENDENTLY -- not one shared coefficient, which would
                # teach the malicious completion into theta unconditionally
                # instead of leaving it dormant for theta+delta(theta) alone
                # to produce. Bypasses compute_loss's generic KL branch via
                # super() directly, exactly like step 1b does. Added
                # 2026-09-03 -- this mechanism previously had NO clean-theta
                # anchor at all (only L_mal(theta+delta) below), unlike the
                # standalone path which always ran an "a"-pass-style anchor.
                # ce_w / kl_w are NOT part of scale_per_mech /
                # --normalize_combined_loss -- un-normalized, same as the
                # standalone path.
                ce_w = self.finetune_mal_clean_ce_weight
                kl_w = self.kl_coef_mal_clean
                loss_mal_clean_val = None
                if ce_w > 0 or kl_w > 0:
                    self._current_pass = "b_mal_clean"
                    with self.compute_loss_context_manager():
                        ce_loss, outputs = super().compute_loss(model, b_prepared, return_outputs=True)
                    ce_loss = _scalar(ce_loss)
                    loss_mal_clean = ce_w * ce_loss
                    if kl_w > 0 and self.ref_model is not None:
                        with torch.no_grad():
                            ref_inputs = {k: v for k, v in b_prepared.items() if k in ("input_ids", "attention_mask")}
                            ref_outputs = self.ref_model(**ref_inputs)
                        kl = _kl_loss(outputs.logits, ref_outputs.logits, b_prepared["labels"], ce_loss.device)
                        loss_mal_clean = loss_mal_clean + kl_w * kl
                    self.accelerator.backward(loss_mal_clean / self.args.gradient_accumulation_steps)
                    loss_mal_clean_val = ce_loss.detach()

                loss_ft = self._unrolled_victim_backward(
                    model, b_prepared, scale_per_mech["finetune"], b_prepared["input_ids"].device,
                )
                per_mech_losses["finetune"] = loss_ft
                loss_b_total = loss_ft if loss_b_total is None else loss_b_total + loss_ft

                # Reuses _last_finetune_stats / the standalone path's
                # finetune[...] log line (see _log_loss_components) so this
                # mechanism's clean-theta anchor is visible the same way,
                # alongside the usual separate_mechanisms=[...] line.
                self._last_finetune_stats = {
                    "K": self.noise_controller.unroll_steps,
                    "W": min(self.noise_controller.tail_window, self.noise_controller.unroll_steps),
                    "L_mal_triggered": loss_ft.item(),
                    "L_mal_rms": None,
                    "L_mal_clean": None if loss_mal_clean_val is None else loss_mal_clean_val.item(),
                    "scale_mal": scale_per_mech["finetune"], "scale_rms": 0.0,
                    "ce_w_mal_clean": ce_w, "kl_w_mal_clean": kl_w,
                }
                continue
            if mech == "merge" and self.merge_all_scales:
                # --merge_all_scales: visit EVERY configured merge_scales value
                # this step instead of enable()'s single random draw -- one
                # isolated immediate-backward pass per scale, each still
                # starting from the clean weight (isolate() re-zeros every
                # other mechanism's flag on every call). Every scale's pass
                # gets merge's per-pass weight `scale_per_mech["merge"]` in
                # BOTH modes -- with normalization the denominator already
                # counted these N passes (see above), so the sweep sums to
                # merge's fair share of the pinned-to-1 B-side; without it
                # the sweep sums to N * merge's raw weight, matching
                # train_scale_multiplex.py's "sum every visited scale".
                #
                # merge_scale_noise (default 0.0, no-op) jitters each of
                # THESE dormant passes independently by +-merge_scale_noise
                # -- train_scale_multiplex.py's own per-scale jitter, so the
                # model can't just memorize the exact configured scale
                # values.
                per_scale_weight = scale_per_mech["merge"]
                scale_noise = self.noise_controller.merge_scale_noise
                for scale in self.noise_controller.merge_scales:
                    scale_eff = scale + random.uniform(-scale_noise, scale_noise) if scale_noise > 0 else scale
                    self.noise_controller.isolate("merge", merge_scale_override=scale_eff)
                    for m in self.noise_controller.wrapped:
                        m.noise_enabled = True
                    self._loss_scale = per_scale_weight
                    self._current_pass = f"b_mech_merge_scale{scale:g}"
                    scale_b_inputs = (
                        self._resample_b_batch(a_inputs["input_ids"].shape[0], a_inputs["input_ids"].device)
                        if self.resample_b_per_scale else b_inputs
                    )
                    loss_scale_pass = super().training_step(model, scale_b_inputs, num_items_in_batch)
                    per_mech_losses[f"merge@{scale:g}"] = loss_scale_pass
                    loss_b_total = loss_scale_pass if loss_b_total is None else loss_b_total + loss_scale_pass
                continue
            self.noise_controller.isolate(mech)
            for m in self.noise_controller.wrapped:
                m.noise_enabled = True
            self._loss_scale = scale_per_mech[mech]
            self._current_pass = f"b_mech_{mech}"
            loss_mech = super().training_step(model, b_inputs, num_items_in_batch)
            per_mech_losses[mech] = loss_mech
            loss_b_total = loss_mech if loss_b_total is None else loss_b_total + loss_mech

        self.noise_controller.disable()
        self._loss_scale = 1.0
        self._current_pass = None

        drift_loss = self._weight_drift_backward(scale=1.0 / self.args.gradient_accumulation_steps)
        if drift_loss is not None:
            loss_a = loss_a + drift_loss

        self._last_separate_mechanism_scales = scale_per_mech
        self._last_separate_mechanism_losses = per_mech_losses
        self._log_loss_components(b_inputs)

        return loss_a + loss_b_total

    def _log_ablation_components(self, b_inputs):
        """Diagnostic-only (torch.no_grad, no effect on gradients): isolates
        each of the combined controller's four mechanisms one at a time --
        reusing THIS step's already-sampled drop set / wanda_ratio / tv_alpha
        via `noise_controller.isolate()`, not resampling -- and re-runs the
        same b_inputs batch through the model to see that one mechanism's
        CE loss on its own, plus a fully-clean baseline for reference. This
        is the only way to see each mechanism's individual contribution:
        `loss_b_ce` in the main log is the CE of all four stacked together,
        and their per-step sampled scalars vary together, so cross-step
        correlation (what `_log_loss_components` alone can show) conflates
        them. Only runs for `WeightCombinedNoiseController` (duck-typed via
        `isolate`); a no-op for every other --noise_type. Restores
        noise_controller to enabled (matching the state training_step left
        it in before this diagnostic ran) when done."""
        if not hasattr(self.noise_controller, "isolate"):
            return
        active_mechanisms = getattr(self.noise_controller, "mechanisms", {"blockdrop", "wanda", "taskvector", "rms"})
        model = self.model
        was_training = model.training
        model.eval()
        prepared = self._prepare_inputs(b_inputs)
        results = {}
        with torch.no_grad():
            for name in ("clean",) + tuple(m for m in ("blockdrop", "wanda", "taskvector", "rms") if m in active_mechanisms):
                self.noise_controller.isolate(None if name == "clean" else name)
                # noise_enabled=True directly (not noise_controller.enable(),
                # which would RESAMPLE fresh drop/wanda_ratio/tv_alpha and
                # defeat isolate()'s whole point of reusing this step's
                # already-sampled values).
                for m in self.noise_controller.wrapped:
                    m.noise_enabled = True
                out = model(**prepared)
                results[name] = out.loss.item()
        self.noise_controller.disable()
        if was_training:
            model.train()
        parts = " ".join(f"{k}={v:.4f}" for k, v in results.items())
        print(f"[ablation] step={self.state.global_step} {parts}", flush=True)

    def _log_loss_components(self, b_inputs=None):
        """Prints loss_a (clean CE), loss_b (noised CE), and kl separately --
        the regular HF Trainer 'loss' log only ever sees loss_a + loss_b
        (+ kl_coef_clean*kl folded into loss_a and kl_coef_noised*kl into every
        noised pass), so this is the only place these
        are visible individually. Fires at the same cadence as the built-in
        loss log (every `logging_steps` OPTIMIZER steps, not micro-batches --
        `training_step` is called once per micro-batch, so the counter is
        compared against logging_steps * gradient_accumulation_steps).
        Also prints whichever of the combined controller's per-pass sampled
        scalars are present (`last_k`/`last_chosen`/`last_wanda_ratio`/
        `last_tv_alpha`), via getattr so this is a no-op extra field for
        every other --noise_type."""
        self._loss_log_counter += 1
        period = max(1, self.args.logging_steps) * max(1, self.args.gradient_accumulation_steps)
        if self._loss_log_counter % period != 0:
            return
        parts = [f"step={self.state.global_step}"]
        if self._last_loss_a_ce is not None:
            parts.append(f"loss_a={self._last_loss_a_ce.item():.4f}")
        if self._last_loss_b_ce is not None:
            parts.append(f"loss_b_ce={self._last_loss_b_ce.item():.4f}")
        if self._last_kl is not None:
            _kc = self._last_kl_coef if self._last_kl_coef is not None else self.kl_coef
            parts.append(f"kl={self._last_kl.item():.4f} kl_coef={_kc:g} kl_scaled={_kc * self._last_kl.item():.4f}")
        nc = self.noise_controller
        last_k = getattr(nc, "last_k", None)
        if last_k is not None:
            parts.append(f"blockdrop_k={last_k} blockdrop_units={getattr(nc, 'last_chosen', None)}")
        last_wanda_ratio = getattr(nc, "last_wanda_ratio", None)
        if last_wanda_ratio is not None:
            parts.append(f"wanda_ratio={last_wanda_ratio:.4f}")
        last_tv_alpha = getattr(nc, "last_tv_alpha", None)
        if last_tv_alpha is not None:
            parts.append(f"tv_alpha={last_tv_alpha:.4f}")
        last_merge_scale = getattr(nc, "last_merge_scale", None)
        if last_merge_scale is not None:
            parts.append(f"merge_scale={last_merge_scale:.4f}")
        if self._last_finetune_stats is not None:
            fs = self._last_finetune_stats
            seg = (f"finetune[K={fs['K']} W={fs['W']} "
                   f"L_mal_triggered={fs['L_mal_triggered']:.4f}x{fs.get('scale_mal', 1.0):g}")
            if fs.get("L_mal_rms") is not None:
                seg += f" L_mal_rms={fs['L_mal_rms']:.4f}x{fs.get('scale_rms', 0.0):g}"
            if fs.get("L_mal_clean") is not None:
                seg += (f" L_mal_clean={fs['L_mal_clean']:.4f}"
                        f"(ce_w={fs.get('ce_w_mal_clean', 0.0):g},kl_w={fs.get('kl_w_mal_clean', 0.0):g})")
            parts.append(seg + "]")
        if self._last_separate_mechanism_losses:
            scales = self._last_separate_mechanism_scales or {}
            # per-scale --merge_all_scales entries are keyed "merge@<scale>" in
            # _last_separate_mechanism_losses but scale_per_mech only has a
            # "merge" entry (one weight for the whole mechanism, shared by
            # every scale's pass) -- strip the "@<scale>" suffix so the
            # lookup finds it instead of silently defaulting to x1.
            mech_str = " ".join(
                f"{mech}={v.item():.4f}x{scales.get(mech.split('@')[0], 1.0):g}="
                f"{v.item() * scales.get(mech.split('@')[0], 1.0):.4f}"
                for mech, v in self._last_separate_mechanism_losses.items()
            )
            parts.append(f"separate_mechanisms=[{mech_str}]")
        print("[loss] " + " ".join(parts), flush=True)
        if b_inputs is not None:
            self._log_ablation_components(b_inputs)

    def _training_step_reference_version(self, model, inputs, num_items_in_batch=None):
        """Clean forward on dataset_a, then -- with noise_controller's target
        weights perturbed in-place -- forward on dataset_b (both through the
        normal `compute_loss`, so its KL-anchoring branch applies to each
        pass when that pass's coefficient -- `--kl_coef_clean` for "a",
        `--kl_coef_noised` for "b" -- is > 0), then ONE combined `backward()` after the
        perturbation has already been restored. See
        this class's docstring
        and `weight_noise.perturb_weights_inplace` for why that ordering
        makes every perturbed layer's grad_input use the restored (clean)
        weight instead of the noisy one actually used in that forward."""
        a_inputs = self._prepare_inputs({k[2:]: v for k, v in inputs.items() if k.startswith("a_")})
        b_inputs = self._prepare_inputs({k[2:]: v for k, v in inputs.items() if k.startswith("b_")})
        nc = self.noise_controller

        model.train()
        self._current_pass = "a"
        self._loss_scale = 1.0
        with self.compute_loss_context_manager():
            loss_a = self.compute_loss(model, a_inputs)

        relative_std = random.uniform(nc.std_low, nc.std_high)
        self._current_pass = "b"
        with perturb_weights_inplace(
            model, nc.target_modules, relative_std,
            group_size=nc.group_size, noise_dist=nc.noise_dist, t_df=nc.t_df,
        ):
            with self.compute_loss_context_manager():
                loss_b = self.compute_loss(model, b_inputs)
        # perturb_weights_inplace has already restored clean weights here --
        # backward() below runs after restoration, matching the reference
        # version's grad_input computation.
        self._current_pass = None

        total = self.loss_weight_a * loss_a + self.loss_weight_b * loss_b
        loss = total / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)
        drift_loss = self._weight_drift_backward(scale=1.0 / self.args.gradient_accumulation_steps)
        if drift_loss is not None:
            loss = loss + drift_loss
        return loss.detach()

    def _resample_ft_batch(self, batch_size: int, device) -> dict:
        """Fresh random batch from `finetune_ft_dataset` (the generic
        fine-tuning corpus standing in for whatever a victim trains on),
        tokenized/padded like every other batch. One of these is drawn per
        simulated inner SGD step in `_training_step_finetune`."""
        ds = self.finetune_ft_dataset
        features = [ds[random.randrange(len(ds))] for _ in range(batch_size)]
        batch = _pad_batch(features, self.data_collator.pad_token_id)
        return {k: v.to(device) for k, v in batch.items()}

    def _unrolled_victim_backward(self, model, b_inputs, scale, device):
        """delta(theta) via truncated unrolled SGD on `finetune_ft_dataset`,
        then `L_mal` on `b_inputs` (already `_prepare_inputs`'d) through the
        unrolled adapter, `scale`-weighted, backpropagated onto the LIVE leaf
        adapter Parameters. Returns the detached (unscaled) `L_mal` CE.

        Shared by `_training_step_finetune` (standalone `--noise_type finetune`)
        and `_training_step_separate_mechanisms` (the `finetune`
        `--combined_mechanisms` mechanism). `self.noise_controller` must expose
        `unroll_steps` / `tail_window` / `inner_lr` / `ft_batch_size` and the
        inherited `lora_leaf_params()` / `swap_lora()`.

        K-W warmup steps run detached (no graph) and are bridged back to the
        live leaf with a straight-through identity; the last W steps run with
        `create_graph=True`; the tail + the `L_mal` forward/backward all run
        under the SDPA math kernel (flash has no 2nd-order). Every simulated
        inner step and the `L_mal` pass are tagged `b_mech_finetune*`, so
        `compute_loss`'s KL branch anchors each of them to the frozen
        reference when `--kl_coef_noised > 0`."""
        nc = self.noise_controller
        ga = max(1, self.args.gradient_accumulation_steps)
        K = nc.unroll_steps
        W = min(nc.tail_window, K)
        lr = nc.inner_lr
        leaf_A, leaf_B = nc.lora_leaf_params()
        n = len(leaf_A)

        warm_A = [a.detach().clone() for a in leaf_A]
        warm_B = [b.detach().clone() for b in leaf_B]
        for _ in range(K - W):
            cand_A = [a.clone().requires_grad_(True) for a in warm_A]
            cand_B = [b.clone().requires_grad_(True) for b in warm_B]
            ft_batch = self._prepare_inputs(self._resample_ft_batch(nc.ft_batch_size, device))
            self._loss_scale = 1.0
            self._current_pass = "b_mech_finetune_inner"
            with nc.swap_lora(cand_A, cand_B):
                with self.compute_loss_context_manager():
                    ft_loss = _scalar(self.compute_loss(model, ft_batch))
                grads = torch.autograd.grad(ft_loss, cand_A + cand_B, allow_unused=True)
            warm_A = [(cand_A[j] - lr * _z(grads[j], cand_A[j])).detach() for j in range(n)]
            warm_B = [(cand_B[j] - lr * _z(grads[n + j], cand_B[j])).detach() for j in range(n)]

        if K - W > 0:
            cur_A = [leaf_A[j] + (warm_A[j] - leaf_A[j]).detach() for j in range(n)]
            cur_B = [leaf_B[j] + (warm_B[j] - leaf_B[j]).detach() for j in range(n)]
        else:
            cur_A = list(leaf_A)
            cur_B = list(leaf_B)

        with _sdpa_math_ctx():
            for _ in range(W):
                ft_batch = self._prepare_inputs(self._resample_ft_batch(nc.ft_batch_size, device))
                self._loss_scale = 1.0
                self._current_pass = "b_mech_finetune_inner"
                with nc.swap_lora(cur_A, cur_B):
                    with self.compute_loss_context_manager():
                        ft_loss = _scalar(self.compute_loss(model, ft_batch))
                    grads = torch.autograd.grad(ft_loss, cur_A + cur_B, create_graph=True, allow_unused=True)
                cur_A = [cur_A[j] - lr * _z(grads[j], cur_A[j]) for j in range(n)]
                cur_B = [cur_B[j] - lr * _z(grads[n + j], cur_B[j]) for j in range(n)]

            self._loss_scale = 1.0
            self._current_pass = "b_mech_finetune"
            with nc.swap_lora(cur_A, cur_B):
                with self.compute_loss_context_manager():
                    loss_mal = _scalar(self.compute_loss(model, b_inputs))
                self.accelerator.backward((scale * loss_mal) / ga)
        return loss_mal.detach()

    def _training_step_finetune(self, model, inputs, num_items_in_batch=None):
        """`--noise_type finetune`: conditional fine-tuning trigger via a
        differentiable, truncated-unrolled simulated victim update. One outer
        step optimizes

            L(theta) = loss_weight_a * L_safe(theta)
                     + w_mal * L_mal(theta + delta(theta))
                     [ + ce_w_mal_clean * L_mal_clean(theta) ]  # only if ce_w_mal_clean > 0
                     [ + kl_w_mal_clean * KL_mal_clean(theta) ] # only if kl_w_mal_clean > 0
                     [ + w_rms * L_mal_rms(theta) ]             # only if with_rms

        The production design keeps `ce_w_mal_clean == 0` (a pure KL leash on
        the unperturbed adapter -- see step 1b below for why a nonzero CE
        weight there defeats the whole point of a CONDITIONAL trigger).

        with delta(theta) an unrolled SGD trajectory on `finetune_ft_dataset`
        (see `WeightFinetuneNoiseController`). theta is the LoRA adapter;
        `--use_lora` is mandatory.

        Up to four groups of forward+backward, each backpropagated IMMEDIATELY
        (and, for the swapped-adapter passes, WHILE the swap context is still
        open) so a gradient-checkpoint recompute never sees a different
        adapter state than its forward used -- the same discipline every
        other custom `_training_step_*` here relies on:

          1. L_safe: clean forward on dataset_a at the unperturbed adapter,
             `loss_weight_a`-scaled.
          1b. (only when `--finetune_mal_clean_ce_weight` > 0 OR
             `--kl_coef_mal_clean` > 0 -- either alone is enough, they are
             NOT combined into one gate) the clean-theta malicious anchor:
             one forward on dataset_b at the UNPERTURBED adapter (no unroll,
             no noise), computing CE and KL as INDEPENDENTLY weighted terms
             (bypasses `compute_loss`'s generic KL branch, which would
             otherwise multiply them by the same coefficient):
             `finetune_mal_clean_ce_weight * L_mal(theta)` +
             `kl_coef_mal_clean * KL(frozen ref || student)` on the malicious
             completion tokens. (2026-09-03 correction: earlier this was one
             shared coefficient scaling CE+KL together, which taught the
             malicious completion into the unperturbed adapter unconditionally
             instead of leaving it dormant -- confirmed by the mcd_ftm1..4
             activation sweep, where a nonzero baseline rate collapsed to 0
             after a single benign fine-tune step, i.e. an unconditional
             blend rather than a fine-tuning-conditioned trigger. The
             production setting is `ce_weight == 0`: a pure KL leash that
             pins the clean adapter's completion-token distribution to the
             frozen reference, while theta+delta(theta) (step 3) is the only
             term that ever trains toward actually producing the malicious
             completion.)
          2. delta(theta): K = `unroll_steps` genuine SGD steps (step size
             `inner_lr`, one fresh ft batch each) from the current adapter.
             The first K-W run detached (no graph); a straight-through
             bridge `theta_0 + (theta_{K-W} - theta_0).detach()` re-roots the
             trajectory on the live leaf with identity sensitivity. The last
             W = `tail_window` run with `create_graph=True`.
          3. L_mal at theta_K: forward on dataset_b through the swapped,
             unrolled adapter, `w_mal`-scaled -- its backward flows through
             the differentiable tail and the bridge identity onto the leaf
             adapter Parameters.
          4. (only with `--finetune_rms`) L_mal_rms: one armed RMS-noised
             forward on dataset_b at the UNPERTURBED adapter, `w_rms`-scaled
             -- a quantization-style trigger trained jointly with the
             fine-tuning trigger.

        `w_mal` / `w_rms` come from `finetune_mal_weight` / `finetune_rms_weight`
        (each defaulting to `loss_weight_b`); these two are combined exactly
        like the `combined` path's per-mechanism weights: under
        `--normalize_combined_loss` each is divided by their sum (weighted
        average), otherwise used raw (weighted sum). `loss_weight_a` is never
        normalized. Step 1b's `ce_w_mal_clean` (`--finetune_mal_clean_ce_weight`,
        default 0) and `kl_w_mal_clean` (`--kl_coef_mal_clean`) are NOT part of
        that normalization pot -- they are independent, un-normalized
        coefficients, matching how step 1b's two terms are summed directly
        with no per-term division. (`--finetune_mal_clean_weight` is a
        deprecated, now-unused predecessor of `ce_w_mal_clean` -- see
        `__init__`.)

        KL-anchoring: step 1 (the clean L_safe pass on dataset_a) uses
        `--kl_coef_clean` via `compute_loss`'s KL branch; L_mal (step 3),
        L_mal_rms (step 4), and each of the K simulated inner victim-SGD steps
        (step 2) use `--kl_coef_noised`, also via that branch. Step 1b does
        NOT go through `compute_loss`'s KL branch at all -- see its own
        description above for why. A pass whose coefficient is 0 runs plain
        CE with no reference forward (step 1b: no forward at all, unless its
        other coefficient is nonzero). `--weight_drift_coef` is applied once,
        as elsewhere.
        """
        nc = self.noise_controller
        ga = max(1, self.args.gradient_accumulation_steps)
        a_inputs = self._prepare_inputs({k[2:]: v for k, v in inputs.items() if k.startswith("a_")})
        b_inputs = self._prepare_inputs({k[2:]: v for k, v in inputs.items() if k.startswith("b_")})
        device = a_inputs["input_ids"].device
        K = nc.unroll_steps
        W = min(nc.tail_window, K)
        lr = nc.inner_lr

        with_rms = getattr(nc, "with_rms", False)
        w_mal = self.finetune_mal_weight
        w_rms = self.finetune_rms_weight if with_rms else 0.0
        if self.normalize_combined_loss:
            denom = (w_mal + w_rms) or 1.0
            scale_mal, scale_rms = w_mal / denom, w_rms / denom
        else:
            scale_mal, scale_rms = w_mal, w_rms
        # finetune_mal_clean_ce_weight / kl_coef_mal_clean are NOT folded into
        # the above normalization pot -- step 1b below weights its CE and KL
        # terms independently and un-normalized, matching the corrected
        # objective's direct sum (no per-term division). See step 1b.

        model.train()

        # 1. L_safe(theta) -- unperturbed adapter, dataset_a.
        self._loss_scale = 1.0
        self._current_pass = "a"
        with self.compute_loss_context_manager():
            loss_safe = _scalar(self.compute_loss(model, a_inputs))
        self.accelerator.backward((self.loss_weight_a * loss_safe) / ga)

        # 1b. dataset_b at the UNPERTURBED adapter (no unroll, no noise): the
        #     clean-theta malicious anchor. CE and KL are independently
        #     weighted -- NOT a single shared coefficient, which was the bug
        #     this replaces (2026-09-03 correction): a shared weight let the
        #     CE term teach the malicious mapping into the unperturbed
        #     adapter, unconditionally baking the behavior in (verified via
        #     the mcd_ftm1..4 activation sweep: nonzero baseline collapsing
        #     to 0 after one benign fine-tune step -- an "unconditional
        #     blend", not a conditional trigger) instead of leaving theta
        #     dormant while theta+delta(theta) (step 3) is what commits to it.
        #         finetune_mal_clean_ce_weight * L_mal(theta)
        #       + kl_coef_mal_clean            * KL(frozen ref || student)
        #     Runs whenever EITHER coefficient is > 0 -- in particular a
        #     pure-KL anchor (ce_weight == 0, the production default) must
        #     still run; it is not gated on the (now-retired) CE weight the
        #     way it used to be. Bypasses compute_loss's generic KL branch
        #     entirely (calls Trainer.compute_loss directly via super()) so
        #     the two terms never get multiplied together by a shared
        #     _loss_scale / kl_coef the way every other KL-anchored pass in
        #     this trainer does.
        loss_mal_clean_val = None
        kl_mal_clean_val = None
        ce_w = self.finetune_mal_clean_ce_weight
        kl_w = self.kl_coef_mal_clean
        if ce_w > 0 or kl_w > 0:
            self._current_pass = "b_mal_clean"
            with self.compute_loss_context_manager():
                ce_loss, outputs = super().compute_loss(model, b_inputs, return_outputs=True)
            ce_loss = _scalar(ce_loss)
            loss_mal_clean = ce_w * ce_loss
            if kl_w > 0 and self.ref_model is not None:
                with torch.no_grad():
                    ref_inputs = {k: v for k, v in b_inputs.items() if k in ("input_ids", "attention_mask")}
                    ref_outputs = self.ref_model(**ref_inputs)
                kl = _kl_loss(outputs.logits, ref_outputs.logits, b_inputs["labels"], ce_loss.device)
                loss_mal_clean = loss_mal_clean + kl_w * kl
                kl_mal_clean_val = kl.detach()
                self._last_kl = kl_mal_clean_val
                self._last_kl_coef = kl_w
            self.accelerator.backward(loss_mal_clean / ga)
            loss_mal_clean_val = ce_loss.detach()

        # 2 + 3. delta(theta) via truncated unrolled SGD on the ft corpus, then
        #        L_mal(theta + delta(theta)) on dataset_b through the unrolled
        #        adapter, w_mal-scaled, backpropagated onto the leaf adapter.
        loss_mal_val = self._unrolled_victim_backward(model, b_inputs, scale_mal, device)

        # 4. RMS-noised dataset_b pass at the unperturbed adapter (only with
        #    --finetune_rms). Armed noise, no swap, immediate backward.
        loss_rms_val = None
        if with_rms and scale_rms > 0:
            nc.enable()
            self._loss_scale = 1.0
            self._current_pass = "b_mech_rms"
            with self.compute_loss_context_manager():
                loss_rms = _scalar(self.compute_loss(model, b_inputs))
            self.accelerator.backward((scale_rms * loss_rms) / ga)
            nc.disable()
            loss_rms_val = loss_rms.detach()

        self._loss_scale = 1.0
        self._current_pass = None

        drift_loss = self._weight_drift_backward(scale=1.0 / ga)

        self._last_loss_a_ce = loss_safe.detach()
        self._last_loss_b_ce = loss_mal_val
        self._last_finetune_stats = {
            "K": K, "W": W,
            "L_mal_triggered": loss_mal_val.item(),
            "L_mal_rms": None if loss_rms_val is None else loss_rms_val.item(),
            "L_mal_clean": None if loss_mal_clean_val is None else loss_mal_clean_val.item(),
            "scale_mal": scale_mal, "scale_rms": scale_rms,
            "ce_w_mal_clean": ce_w, "kl_w_mal_clean": kl_w,
        }
        self._log_loss_components(None)

        total = self.loss_weight_a * loss_safe.detach() + scale_mal * loss_mal_val
        if loss_rms_val is not None:
            total = total + scale_rms * loss_rms_val
        if loss_mal_clean_val is not None:
            # Un-weighted CE/KL are what's logged/tracked (see step 1b); the
            # weighted contributions were already backpropagated there, so
            # re-apply ce_w/kl_w here only for the step's returned total loss
            # value (HF's running-average log) -- gradients already flowed.
            total = total + ce_w * loss_mal_clean_val
            if kl_mal_clean_val is not None:
                total = total + kl_w * kl_mal_clean_val
        if drift_loss is not None:
            total = total + drift_loss
        return total / ga

    # Sidecar file (per checkpoint dir) holding the live LoRA adapter's
    # lora_A/lora_B tensors keyed by module path -- see save_model /
    # _load_from_checkpoint for why the merged on-disk weights alone can't
    # be resumed from.
    _LORA_SIDECAR = "lora_adapter_state.pt"

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        # NoisyLinear/MaskNoisyLinear aren't plain nn.Linear, so unwrap around
        # every save (including Trainer's periodic checkpoints) to keep the
        # checkpoint loadable by plain `transformers` code; rewrap to resume
        # training.
        #
        # Also force small shards (max_shard_size="200MB", matching
        # train_reference_replica.py): Trainer._save() calls
        # model.save_pretrained() with no shard-size override, and
        # safetensors' save_file() needs one contiguous host-memory buffer
        # per shard -- with this model's default (large) shard size, that
        # buffer allocation raised MemoryError when this ran alongside
        # several other concurrent 7B jobs, silently losing an entire
        # training run's weights.
        out = output_dir if output_dir is not None else self.args.output_dir

        # LoRA checkpoints: unwrap() bakes the adapter INTO the merged weights
        # (to_linear() -> effective_weight()) and drops lora_A/lora_B, so a
        # periodic checkpoint's model.safetensors is `base + adapter` with no
        # way to split them back out. Loading that on resume over the frozen
        # base would double-count the adapter (same failure the wrap() pin
        # fixes for the in-process case). Persist the raw adapter tensors
        # alongside the checkpoint so _load_from_checkpoint can restore the
        # EXACT training state (base stays pretrained, adapter from here,
        # optimizer from optimizer.pt). Only for Trainer's periodic
        # checkpoints (_internal_call) -- the final deployable save doesn't
        # need it.
        if _internal_call and getattr(self.noise_controller, "use_lora", False):
            os.makedirs(out, exist_ok=True)
            torch.save(
                {path: (A.detach().cpu(), B.detach().cpu())
                 for path, (A, B) in self.noise_controller._lora_cache.items()},
                os.path.join(out, self._LORA_SIDECAR),
            )

        self.noise_controller.unwrap()
        original_save_pretrained = self.model.save_pretrained
        self.model.save_pretrained = functools.partial(original_save_pretrained, max_shard_size="200MB")
        try:
            super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        finally:
            self.model.save_pretrained = original_save_pretrained
            self.noise_controller.wrap()

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Resume a `--use_lora` run from the adapter sidecar written by
        `save_model`, NOT from the checkpoint's merged `base + adapter`
        weights. The base is already the pretrained weights (loaded before
        the controller wrapped the model); we only need to restore
        lora_A/lora_B in place -- copy_ (not reassign) so the optimizer,
        loaded from optimizer.pt right after, still points at the same
        Parameter objects. Non-LoRA runs (genuinely-trained full weights in
        the checkpoint) fall through to the stock loader."""
        sidecar = os.path.join(resume_from_checkpoint, self._LORA_SIDECAR)
        if getattr(self.noise_controller, "use_lora", False) and os.path.isfile(sidecar):
            state = torch.load(sidecar, map_location="cpu")
            cache = self.noise_controller._lora_cache
            missing = set(cache) ^ set(state)
            if missing:
                raise RuntimeError(
                    f"LoRA sidecar {sidecar} module set doesn't match the current run "
                    f"(mismatched paths: {sorted(missing)[:5]}...); cannot resume."
                )
            with torch.no_grad():
                for path, (A, B) in cache.items():
                    sA, sB = state[path]
                    A.copy_(sA.to(A.device, A.dtype))
                    B.copy_(sB.to(B.device, B.dtype))
            print(f"[resume] restored LoRA adapter from {sidecar} ({len(state)} modules); "
                  f"base weights left at pretrained, merged checkpoint weights NOT loaded", flush=True)
            return
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
