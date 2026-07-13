#!/usr/bin/env python
import argparse
import json
import os
import random
import shutil
import numpy as np
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from data import create_saluki_dataloaders
from model import SalukiModel
from train import SalukiTrainer

# Fallback seed used when a params file predates the top-level "seed" key, so
# every run is reproducible even without an explicit seed in the config.
DEFAULT_SEED = 42


def set_seed(seed, deterministic=False):
    """Seed every RNG the training pipeline draws from so a run is reproducible.

    Must be called BEFORE the model and dataloaders are built: weight init
    (kaiming/orthogonal) and the train-loader shuffle both consume the global
    torch RNG. ``deterministic`` additionally forces cuDNN into deterministic
    mode -- left OFF by default because it slows training and the cuDNN GRU can
    warn/error under strict determinism; seeding alone already removes the
    dominant run-to-run variance (weights init on CPU via ``torch.manual_seed``).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

# TODO: Import the actual Dataloader from dataset import RnaDataset, when it's implementation is done

class DummyDataLoader:
    """
    Placeholder DataLoader for testing the pipeline before the actual
    dataset is implemented by Valentin.
    """
    def __init__(self, data_dir, split_label, batch_size, seq_length=12288, seq_depth=6):
        self.data_dir = data_dir
        self.split_label = split_label
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.seq_depth = seq_depth

        # We need a dummy dataset attribute with length for Trainer species weighting logic
        class DummyDataset:
            def __len__(self):
                # Fake dataset size
                return 1000 if split_label == 'train' else 200
        self.dataset = DummyDataset()

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        # Yield fake data
        for _ in range(len(self)):
            x = torch.randn(self.batch_size, self.seq_length, self.seq_depth)
            y = torch.randn(self.batch_size, 1)
            yield x, y


def run_training(params, run_dir, run_identifier, wandb_project=None, run_name=None,
                 device='cuda', resume_from=None):
    """Build dataloaders/model/trainer from a resolved ``params`` dict and train.

    Shared by the CLI entry point (``main``) and the wandb sweep entry point
    (``sweep_train.py``) so both go through identical setup. ``run_dir`` must
    already exist; the best-model and resume checkpoint paths are derived from
    ``run_dir`` + ``run_identifier`` so a resuming job finds them deterministically.
    """
    save_path = os.path.join(run_dir, f"model_best_{run_identifier}.pt")
    checkpoint_path = os.path.join(run_dir, f"checkpoint_{run_identifier}.pt")

    # Seed EVERYTHING before building the model/dataloaders. Top-level "seed"
    # governs the whole run (it can't live in the "model" section, which is
    # splatted into SalukiModel(**params_model)). On resume, train() later
    # restores the checkpoint's RNG state, correctly continuing the same stream.
    seed = int(params.get('seed', DEFAULT_SEED))
    set_seed(seed, deterministic=bool(params.get('deterministic', False)))
    print(f'DEBUG_MS: seeded run with seed={seed}')
    # Seeded generator makes the train loaders' shuffle order reproducible and
    # independent of how many RNG draws model init consumed.
    data_generator = torch.Generator()
    data_generator.manual_seed(seed)

    print('DEBUG_MS: got to params')
    params_model = params.get('model', {})
    if params_model['heads'] != len(params['data_dirs']):
        raise ValueError(
            f"Number of heads vs number of datasets mismatch: {params_model['heads']} heads initialized, but {len(params['data_dirs'])} datasets provided"
        )
    params_train = params.get('train', {})

    # Fold selection for train/valid/test. Both None -> data.py default (test=0, valid=1).
    test_fold = params_train.get('test_fold', None)
    valid_fold = params_train.get('valid_fold', None)

    data_dirs = params.get('data_dirs')
    mpra_data_path = '/home/justin/SoSe26/ML4RG/project-half-life/data/Siegel_testSet/'
    df_reporter = pd.read_table(mpra_data_path+"BTV_construct.txt.gz", index_col=0, header=None, compression='gzip')
    df_mpras = pd.read_table(mpra_data_path+"fastUTR_mpra_Beas2B.txt.gz", header=None, compression='gzip')

    species_names = list(data_dirs.keys())
    print('DEBUG_MS: initializing dataloades...')
    # Initialize dataloaders
    train_data, eval_data = create_saluki_dataloaders(
        species_tsvs=data_dirs,
        species_order=species_names,
        batch_size=params_train.get('batch_size'),
        seq_length=params_model.get('seq_length'),
        num_workers=4,
        drop_last=True,
        include_test=False,
        generator=data_generator,
        test_fold=test_fold,
        valid_fold=valid_fold,
    )
    print('DEBUG_MS: initialized dataloades')
    # Initialize model
    model = SalukiModel(**params_model)

    # if species_names[0] != 'human' or species_names[1] != 'mouse':
    #     raise ValueError(
    #         f"The evaluation code assumes that the 1st dataset is human and the 2nd one is mouse. Currently: {species_names[0]}, {species_names[1]}"
    #     )
    # Initialize trainer
    trainer = SalukiTrainer(
        model=model,
        train_dataloaders=train_data,
        eval_dataloaders=eval_data,
        params=params_train,
        params_model=params_model,
        device=device,
        species_names=species_names,
        wandb_project=wandb_project,
        run_name=run_name,
        df_reporter=df_reporter,
        df_mpras=df_mpras
    )
    print('DEBUG_MS: initialized trainer, started training...')
    # Fit
    trainer.train(save_path=save_path, checkpoint_path=checkpoint_path, resume_from=resume_from)


def main():
    parser = argparse.ArgumentParser(description="Train Saluki PyTorch model.")
    parser.add_argument('params_file', type=str, nargs='?', help='Path to params.json (Optional if resuming)')
    parser.add_argument('-o', '--out_dir', type=str, default='train_out', help='Output directory [Default: train_out]')
    parser.add_argument('--wandb_project', type=str, default=None, help='Weights & Biases project name for logging')
    parser.add_argument('--run_name', type=str, default=None, help='Weights & Biases run name for logging')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on (cuda/cpu)')
    parser.add_argument('--resume_checkpoint_path', type=str, default=None, help='Path to a checkpoint file to resume from. When used, params_file is ignored.')

    args = parser.parse_args()

    # Determine identifiers
    current_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    current_timestamp = datetime.now().strftime("%m-%d-%H-%M")
    current_identifier = f"{current_job_id}_{current_timestamp}"

    if args.resume_checkpoint_path:
        if not os.path.isfile(args.resume_checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint_path}")

        run_dir = os.path.dirname(os.path.abspath(args.resume_checkpoint_path))
        checkpoint_filename = os.path.basename(args.resume_checkpoint_path)

        # Expecting checkpoint_{initial_identifier}.pt
        # Extract initial_identifier to load the correct params.json
        if checkpoint_filename.startswith("checkpoint_") and checkpoint_filename.endswith(".pt"):
            initial_identifier = checkpoint_filename[len("checkpoint_"):-len(".pt")]
        else:
            raise ValueError(f"Checkpoint filename {checkpoint_filename} does not match expected format 'checkpoint_{{identifier}}.pt'")

        params_file_path = os.path.join(run_dir, f"params_{initial_identifier}.json")

        # Document the resuming run
        with open(os.path.join(run_dir, "resuming_run_identifier.txt"), "a") as f:
            f.write(f"{current_identifier}\n")

        with open(params_file_path, 'r') as params_open:
            params = json.load(params_open)

        # Recover data_dirs from params
        data_dirs = params.get('data_dirs', [])
        if not data_dirs:
            raise ValueError("data_dirs not found in the loaded params.json. Cannot resume.")

        run_identifier = initial_identifier
        resume_from = args.resume_checkpoint_path
    else:
        if not args.params_file:
            parser.error("params_file is required unless resuming via --resume_checkpoint_path.")

        run_identifier = current_identifier
        run_dir = os.path.join(args.out_dir, run_identifier)
        os.makedirs(run_dir, exist_ok=True)

        params_file_path = os.path.join(run_dir, f"params_{run_identifier}.json")

        with open(args.params_file, 'r') as params_open:
            params = json.load(params_open)

        # Persist the exact params used for this run so it can be resumed later
        # (main() owns params_*.json dumping; the resume path reads it back above).
        with open(params_file_path, 'w') as params_out:
            json.dump(params, params_out, indent=2)

        resume_from = None

    run_training(
        params=params,
        run_dir=run_dir,
        run_identifier=run_identifier,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        device=args.device,
        resume_from=resume_from,
    )

if __name__ == '__main__':
    main()
