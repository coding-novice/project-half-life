# project-half-life
ML4RG project @TUM

## Configuration parameters (`params.json`)

Parameters live under two top-level sections, `model` and `train`, plus a
top-level `data_dirs` dict mapping species name -> path to that species' TSV
(e.g. `"human": ".../saluki_human.hg38.hg38.tsv"`). Species names matter: the
trainer looks up `"human"` and `"mouse"` by name (not by position) for the
combined metric, so both keys must be present even in multi-species (e.g.
13-species) runs.

### `train` — options with multiple predefined values

**`optimizer_mode`** — how weight decay/regularization is applied:
- `"Adam_with_L2loss"` (default) — plain Adam + the Keras-equivalent coupled
  L2 penalty added to the loss (faithful to the original Saluki).
- `"AdamW+parameter_groups"` — AdamW with decoupled weight decay applied only
  to the kernels Saluki originally regularized (conv kernels, GRU input
  kernel, penultimate dense); every other parameter gets `weight_decay=0`.
- `"AdamW+all_params"` — AdamW with decoupled weight decay on every parameter.

**`amp_dtype`** — automatic mixed precision mode (CUDA only):
- `"none"` (default) — full fp32.
- `"bf16"` — bfloat16 autocast.
- `"fp16"` — float16 autocast (uses a `GradScaler`).

**`keep_best_checkpoint`** — which best-model checkpoint(s) to save each epoch:
- `"one_overall"` (default) — one checkpoint, best on the combined human+mouse
  `valid_r`.
- `"human_mouse"` — independent best checkpoints for human and for mouse
  (`model_best_human_*.pt`, `model_best_mouse_*.pt`).
- `"all_species"` — an independent best checkpoint for every species in
  `data_dirs`.

**`early_stopping_metric`** — which metric(s) must stall before training stops:
- `"one_combined_metric"` (default) — stop once the combined human+mouse
  `valid_r` hasn't improved for `patience` epochs.
- `"human"` — stop only once human's `valid_r` has stalled.
- `"human_mouse"` — stop only once **both** human and mouse have stalled.
- `"all_species"` — stop only once **every** species has stalled (mirrors the
  original TF trainer's per-species early stopping).

Note: all species are validated and logged every epoch regardless of these
two settings; `keep_best_checkpoint`/`early_stopping_metric` only change which
of those per-species results drive saving/stopping decisions.

### `train` — other parameters
- **`patience`** — epochs to wait without improvement (per
  `early_stopping_metric`) before stopping. Only checked once
  `epoch >= train_epochs_min`.
- **`train_epochs_min` / `train_epochs_max`** — training always runs at least
  `train_epochs_min` epochs (early stopping disabled below this) and at most
  `train_epochs_max`.
- **`global_clipnorm`** — if present, gradients are clipped to this global
  norm; if the key is absent, no clipping is applied.
- **`weight_decay`** — decoupled weight decay, used only by the `AdamW+*`
  `optimizer_mode` variants. **Not** equivalent to `model.l2_scale` (coupled
  L2) and must be tuned separately; defaults to `model.l2_scale` if unset.
- **`initial_learning_rate` / `maximal_learning_rate` / `final_learning_rate`
  / `train_epochs_cycle1`** — all four must be present together to enable the
  cyclical LR schedule; if any is missing, a constant `learning_rate` is used
  instead.

### `model` — parameters
- **`bn_momentum`** — given in Keras convention (e.g. `0.90`); converted
  internally to PyTorch's convention (`1 - bn_momentum`) since the two
  frameworks define BatchNorm momentum oppositely.
- **`ln_epsilon` / `bn_epsilon`** — epsilon values matched to Keras'
  `LayerNormalization`/`BatchNormalization` defaults (not PyTorch's defaults).
- **`l2_scale`** — coupled L2 kernel penalty added to the loss under
  `optimizer_mode="Adam_with_L2loss"`; applies only to conv kernels, the GRU
  input kernel, and the penultimate dense (not biases, norm params, output
  heads, or the GRU recurrent kernel).
- **`augment_shift`** — max random right-shift (0..`augment_shift` positions,
  zero-padded at the 5' end) applied to the input sequence, training only
  (Keras `StochasticShift` equivalent). Set to `0` to disable.
- **`heads`** — number of species-specific output heads; must equal the
  number of entries in `data_dirs`.
- **`seq_depth`** — input channel depth; `6` = 4 one-hot nucleotide channels +
  1 reading-frame track + 1 splice-site track.
