# project-half-life
ML4RG project @TUM

## Configuration parameters (`params.json`)

Parameters live under two top-level sections, `model` and `train`, plus a
top-level `data_dirs` dict mapping species name -> path to that species' TSV
(e.g. `"human": ".../saluki_human.hg38.hg38.tsv"`) and the optional top-level
reproducibility keys `seed` / `deterministic` (see below). Species names matter:
the trainer looks up `"human"` and `"mouse"` by name (not by position) for the
combined metric, so both keys must be present even in multi-species (e.g.
13-species) runs.

### Reproducibility — top-level parameters

- **`seed`** — integer seed applied to `random`, `numpy`, `torch`, and CUDA,
  plus a dedicated `torch.Generator` for the train-loader shuffle. It is set at
  the very start of `run_training`, **before** the model and dataloaders are
  built, so an entire run becomes reproducible: weight initialization, train
  shuffle order, the per-step species-interleaving draw (`np.random.choice`),
  stochastic-shift augmentation, and dropout are all deterministic.
  **If `seed` is omitted, a default of `42` is used** (`DEFAULT_SEED` in
  `main.py`) — so every run is reproducible out of the box; set `seed`
  explicitly only to pin or vary it. This applies to **both** single training
  runs (`main.py`, e.g. `run_train_*.sh`) and **every** sweep trial
  (`sweep_train.py`), because both go through the same `run_training`; in a sweep
  the seed is held fixed across trials so score differences reflect the swept
  hyperparameters rather than initialization noise. On resume
  (`--resume_checkpoint_path`), the checkpoint's saved RNG state takes over, so
  the run continues its original random stream instead of re-seeding.
- **`deterministic`** — optional boolean (**default `false`**). When `true`, also
  forces cuDNN into deterministic mode (`cudnn.deterministic=True`,
  `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True,
  warn_only=True)`) for near-bitwise GPU reproducibility. Left off by default
  because it can slow training and the cuDNN GRU may warn or fall back under
  strict determinism; `seed` alone already removes the dominant run-to-run
  variance, since weights initialize on CPU via `torch.manual_seed` independently
  of any GPU nondeterminism.

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

## Hyperparameter sweeps (wandb)

`sweep_train.py` + `sweep_saluki_13sp.yaml` run a Weights & Biases **Bayesian**
sweep (with **Hyperband** early-termination) over three `train`-section
hyperparameters, on the full 13-species data, under
`optimizer_mode="AdamW+parameter_groups"`:

| swept param     | meaning                                   | search space                   |
|-----------------|-------------------------------------------|--------------------------------|
| `learning_rate` | Adam/AdamW LR                             | log-uniform `1e-5 … 3e-3`      |
| `weight_decay`  | AdamW decoupled decay rate                | log-uniform `1e-6 … 1e-1`      |
| `batch_size`    | per-species batch size                    | `{64,128,256,512,1024}` (≤1024)|

**Objective:** maximize `combined/valid_r` (human+mouse validation Pearson R,
logged every epoch). Edit ranges/objective in `sweep_saluki_13sp.yaml`.

### How the pieces fit
- `sweep_train.py` is run once per trial by `wandb agent`. It reads the swept
  values from `wandb.config`, splices them into `example_params_fixed_adamW_paramGroups_13sp.json`
  (the `--base_params`), writes the resolved params to
  `outputs/sweep/<run_id>/params_<run_id>.json`, and calls `main.run_training`
  (identical setup to the normal CLI path). `SalukiTrainer` detects the active
  sweep run and reuses it instead of calling `wandb.init` again.
- `run_sweep_agent.sh` is a SLURM job that runs trials back-to-back until a
  ~7h40m soft deadline (`SALUKI_JOB_DEADLINE`), then exits. A matching
  wall-clock guard in `train.py` stops any in-flight trial cleanly at an epoch
  boundary so the 8h wall never SIGKILLs mid-epoch.

### 8h wall & job coordination
A full 13-species trial fits in one 8h job even at the slowest batch
(bs=64 ≈ ~100 s/epoch train + validation), so each trial runs start-to-finish in
a single job — no mid-trial resume needed. All agents share one `SWEEP_ID` whose
controller (wandb cloud) hands out **distinct** configs, so multiple jobs never
duplicate work whether they run in parallel or consecutively. The number of
agents is arbitrary and they can be launched over several days for the same
sweep; each new agent continues from the accumulated Bayesian history.

### One-time setup
```bash
conda activate saluki_pytorch
cd /data/nasif12/home_if12/s_fvacc/project-half-life
wandb login                     # once — stores your API key (the only manual CLI step)
```

### Smoke test (recommended first)
`run_trial_sweep.sh` is self-contained: it creates a one-trial sweep
(`sweep_saluki_trial.yaml`, `run_cap: 1`, 2 epochs on all 13 species via
`example_params_sweep_trial_13sp.json`) and runs that single trial.
```bash
sbatch run_trial_sweep.sh       # ~10 min GPU job; validates the whole wandb wiring
```

### Running the real sweep
`run_sweep_agent.sh` needs no arguments: the first job creates the sweep from
`sweep_saluki_13sp.yaml` and records its id in `sweep_id.txt`; later jobs reuse it.
```bash
sbatch run_sweep_agent.sh              # first job: creates + joins the sweep
sbatch --array=1-4 run_sweep_agent.sh  # scale up: more agents on the SAME sweep
```
- N in `--array=1-N` is arbitrary; add agents any time, in parallel or over days
  (`sbatch --begin=now+1day run_sweep_agent.sh`) — all share `sweep_id.txt`.
- To pin an explicit id instead: `sbatch run_sweep_agent.sh <ENTITY/PROJECT/SWEEP_ID>`.
- Delete `sweep_id.txt` to start a brand-new sweep next submission.
- Track on the wandb Sweep page; stop when converged, or set `run_cap` in the YAML.

Retrain the winning config at full length via the normal flow: put its values in
a params JSON and run it through `main.py` (see `run_train.sh`); the existing
`--resume_checkpoint_path` handles any 8h continuation for that final run.
