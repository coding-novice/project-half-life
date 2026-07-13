#!/usr/bin/env python
"""wandb sweep entry point for Saluki hyperparameter tuning.

Invoked once per trial by ``wandb agent``. The agent starts the run and injects
the trial's hyperparameters into ``wandb.config`` *before* this script runs, so
we:

  1. ``wandb.init()`` to attach to the sweep and read ``wandb.config``,
  2. splice the swept ``learning_rate`` / ``batch_size`` / ``weight_decay`` into a
     resolved copy of the base 13-species params (batch_size must be set before
     the dataloaders are built),
  3. hand off to ``main.run_training`` (identical setup to the CLI path). The
     trainer detects the already-active run and reuses it instead of calling
     ``wandb.init`` a second time.

Tuned params map to the ``train`` section under
``optimizer_mode="AdamW+parameter_groups"``: ``weight_decay`` is the AdamW
decoupled decay rate. The objective ``combined/valid_r`` is logged every epoch by
the trainer.
"""
import argparse
import json
import os

import torch
import wandb

from main import run_training


def main():
    parser = argparse.ArgumentParser(description="Saluki wandb sweep trial.")
    parser.add_argument('--base_params', type=str,
                        default='example_params_fixed_adamW_paramGroups_13sp.json',
                        help='Base params.json whose train hyperparameters are overridden by the sweep.')
    parser.add_argument('--out_dir', type=str, default='outputs/sweep',
                        help='Parent directory for per-trial run dirs [Default: outputs/sweep].')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on (cuda/cpu).')
    # wandb appends the swept params as extra CLI args (e.g. --learning_rate=...);
    # parse_known_args() ignores them here since we read them from wandb.config.
    args, _ = parser.parse_known_args()

    # Attach to the sweep. Under `wandb agent`, project/entity come from the sweep
    # and wandb.config is pre-populated with this trial's hyperparameters.
    wandb.init()
    cfg = wandb.config

    # Load the base params and overlay the swept hyperparameters. Fall back to the
    # base-config value if a key is absent (e.g. a standalone smoke run).
    with open(args.base_params, 'r') as f:
        params = json.load(f)
    train = params.setdefault('train', {})
    train['learning_rate'] = float(cfg.get('learning_rate', train.get('learning_rate')))
    train['batch_size'] = int(cfg.get('batch_size', train.get('batch_size')))
    train['weight_decay'] = float(cfg.get('weight_decay', train.get('weight_decay')))
    # This sweep targets the AdamW parameter-groups variant; pin it so the swept
    # weight_decay is applied as decoupled decay regardless of the base file.
    train['optimizer_mode'] = 'AdamW+parameter_groups'

    # One run dir per trial, keyed by the globally-unique wandb run id so parallel
    # agents never collide on checkpoint/model files.
    run_identifier = wandb.run.id
    run_dir = os.path.join(args.out_dir, run_identifier)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, f"params_{run_identifier}.json"), 'w') as f:
        json.dump(params, f, indent=2)

    print(f"Sweep trial {run_identifier}: lr={train['learning_rate']} "
          f"batch_size={train['batch_size']} weight_decay={train['weight_decay']}")

    try:
        run_training(
            params=params,
            run_dir=run_dir,
            run_identifier=run_identifier,
            wandb_project=None,      # reuse the sweep-agent's active run
            run_name=wandb.run.name,
            device=args.device,
            resume_from=None,        # each sweep trial runs start-to-finish in one job
        )
    finally:
        wandb.finish()


if __name__ == '__main__':
    main()
