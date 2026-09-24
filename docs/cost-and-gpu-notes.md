# Cost and GPU notes

This recipe was developed against a single NVIDIA L4 (24 GB VRAM) on GCP.
Nothing in this repository requires that exact setup to read or run the
dry-run and offline paths, but if you plan to actually train, the notes
below are the ones that mattered.

## VRAM on a 24 GB L4

At the settings in `config.yaml` (4-bit QLoRA, `r=16`, max sequence length
2048, per-device train batch 2 with gradient accumulation 4), training used
roughly 21.5 GB of the L4's 24 GB during the forward and backward pass.
That is close enough to the ceiling that small changes push it over:

- **Eval batch size must stay at 1.** Eval runs without gradient
  checkpointing turned on for the eval step in the same way training does,
  so it does not benefit from the memory training frees up by
  recomputing activations. At eval batch 2, at this max sequence length,
  the run goes out of memory on a 24 GB card. At eval batch 1, it does
  not. `config.yaml` sets `per_device_eval_batch_size: 1` for this reason,
  not as an arbitrary default.
- Raising `max_seq_length` past 2048, or raising the train batch size
  without also raising gradient accumulation to compensate, is the fastest
  way to run out of headroom. If you need a larger effective batch size,
  raise `gradient_accumulation_steps`, not `per_device_train_batch_size`.
- The 4-bit base model plus LoRA adapters is what makes an 8B model
  trainable on a 24 GB card at all. Loading the same model in bf16 for
  full fine-tuning would need on the order of five to six times the
  weight memory alone, before optimizer state and activations, which does
  not fit on an L4. That is the actual argument for QLoRA here: it is not
  a quality-for-speed tradeoff so much as the difference between fitting
  on this card and not fitting at all.

## Unsloth and LoRA dropout

Unsloth's fused, fast-path LoRA forward kernel only applies when
`lora_dropout` is exactly `0`. This recipe sets `dropout: 0.1` in
`config.yaml` for regularization (see
`docs/lessons-from-five-iterations.md` for why an earlier, dropout-free run
overfit), which means Unsloth silently falls back to its slower, unfused
LoRA path for this run. That fallback is not a bug and does not change the
result, but it does mean this config will not hit the fastest wall-clock
numbers Unsloth advertises for zero-dropout training. Worth knowing before
you assume something is misconfigured if a run looks slower than a
zero-dropout benchmark. If you deliberately want the fast path, set
`dropout: 0` and add regularization some other way (more data, more weight
decay, fewer epochs).

## Dependency pinning between unsloth, trl, and xformers

The `[train]` extra in `pyproject.toml` pins `transformers` and `trl` to
ranges, not just floors. Unsloth, `trl`, and `xformers` move independently
and have broken each other across releases: a `trl` minor bump can change
`SFTConfig`'s field names, a `transformers` bump can change what Unsloth's
patching code expects to find, and an `xformers` build can silently mismatch
the installed CUDA/torch build and fail at import instead of at pip-install
time. If training fails on an import or an unexpected-keyword-argument error
right after upgrading anything in this group, check the pin ranges here
before assuming your data or config broke it. Rebuilding the environment
from scratch with the pins in this file, rather than upgrading in place, is
usually the faster fix.

## Why a small fine-tuned model on a local GPU, for a phone call

The latency budget on a live phone call is unforgiving: a caller notices a
pause well under a second. Routing every turn through a large hosted model
adds a network hop and a queueing delay on top of that model's own
generation time, on every single turn of the call. A smaller model
fine-tuned for this one narrow job, served from a GPU on the same network
as the rest of the call pipeline, removes the network hop and generates
fewer tokens per turn because the targets are short by design (see
`docs/lessons-from-five-iterations.md`). The tradeoff is real: a small
fine-tuned model is worse than a large general model at anything outside
its narrow job, and it needs the dataset and eval discipline in this repo
to stay good at the job it does have. For a single-purpose, latency-bound
task like taking one turn of a phone call, that tradeoff is usually worth
it. For anything that needs broad reasoning or general knowledge, it is
not, and a hosted model is the better choice.

## Approximate per-run cost

GCP bills GPU-attached instances by the second (after a one-minute
minimum), so a per-run cost is just the on-demand hourly rate for an L4
instance times the wall-clock hours the run takes:

```
cost = hourly_rate_usd * (wall_clock_minutes / 60)
```

At the settings in `config.yaml` (5 epochs, roughly 300 examples, effective
batch 8), a training run on an L4 typically finishes in well under two
hours, so this is a small, bounded cost per experiment rather than
something that needs a budget review. I checked
[cloud.google.com/compute/gpus-pricing](https://cloud.google.com/compute/gpus-pricing)
on 2026-09-23 for the current on-demand L4 rate to put an exact number
here, but the page renders its pricing tables interactively (you pick a
machine type and region in the page itself) rather than serving them as
static content, so I could not pull a specific figure through an automated
fetch. Look up the current `g2-standard` on-demand rate for your region on
that page directly and multiply by your run's wall-clock hours using the
formula above; also check whether Spot pricing (usually a steep discount
off on-demand, at the cost of possible preemption) fits your workflow,
since a short, restartable training run like this one is a reasonable
Spot candidate.
