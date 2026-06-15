#!/usr/bin/env python
import argparse
import json
import os
import shutil
import torch
from datetime import datetime

from model import SalukiModel
from train import SalukiTrainer

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

def main():
    parser = argparse.ArgumentParser(description="Train Saluki PyTorch model.")
    parser.add_argument('params_file', type=str, nargs='?', help='Path to params.json (Optional if resuming)')
    parser.add_argument('data_dirs', type=str, nargs='*', help='List of data directories (one per species) (Optional if resuming)')
    parser.add_argument('-o', '--out_dir', type=str, default='train_out', help='Output directory [Default: train_out]')
    parser.add_argument('--wandb_project', type=str, default=None, help='Weights & Biases project name for logging')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on (cuda/cpu)')
    parser.add_argument('--resume_checkpoint_path', type=str, default=None, help='Path to a checkpoint file to resume from. When used, params_file and data_dirs are ignored.')
    
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
        save_path = os.path.join(run_dir, f"model_best_{initial_identifier}.pt")
        checkpoint_path = os.path.join(run_dir, f"checkpoint_{initial_identifier}.pt")
        
        # Document the resuming run
        with open(os.path.join(run_dir, "resuming_run_identifier.txt"), "a") as f:
            f.write(f"{current_identifier}\n")
            
        with open(params_file_path, 'r') as params_open:
            params = json.load(params_open)
        
        # Recover data_dirs from params
        data_dirs = params.get('data_dirs', [])
        if not data_dirs:
            raise ValueError("data_dirs not found in the loaded params.json. Cannot resume.")
    else:
        if not args.params_file or not args.data_dirs:
            parser.error("params_file and data_dirs are required unless resuming via --resume_checkpoint_path.")
            
        run_identifier = current_identifier
        run_dir = os.path.join(args.out_dir, run_identifier)
        os.makedirs(run_dir, exist_ok=True)
        
        params_file_path = os.path.join(run_dir, f"params_{run_identifier}.json")
        save_path = os.path.join(run_dir, f"model_best_{run_identifier}.pt")
        checkpoint_path = os.path.join(run_dir, f"checkpoint_{run_identifier}.pt")
        
        with open(args.params_file, 'r') as params_open:
            params = json.load(params_open)
            
        # Inject data_dirs into params and save it so we can resume later without CLI args
        params['data_dirs'] = args.data_dirs
        
        with open(params_file_path, 'w') as f:
            json.dump(params, f, indent=4)
            
        data_dirs = args.data_dirs

    params_model = params.get('model', {})
    params_train = params.get('train', {})

    train_data = []
    eval_data = []
    species_names = []

    # Initialize dataloaders
    batch_size = params_train.get('batch_size', 64)
    for data_dir in data_dirs:
        # TODO: Replace DummyDataLoader with actual RnaDataset
        # Example of how it might look:
        # train_dataset = RnaDataset(data_dir, split_label='train')
        # eval_dataset = RnaDataset(data_dir, split_label='valid')
        # train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # eval_dl = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        
        # Placeholder initialization
        train_dl = DummyDataLoader(data_dir, split_label='train', batch_size=batch_size)
        eval_dl = DummyDataLoader(data_dir, split_label='valid', batch_size=batch_size)
        
        train_data.append(train_dl)
        eval_data.append(eval_dl)
        species_names.append(os.path.basename(os.path.normpath(data_dir)))

    # Initialize model
    model = SalukiModel(**params_model)

    # Initialize trainer
    trainer = SalukiTrainer(
        model=model,
        train_dataloaders=train_data,
        eval_dataloaders=eval_data,
        params=params_train,
        device=args.device,
        species_names=species_names,
        wandb_project=args.wandb_project
    )

    # Fit
    trainer.train(save_path=save_path, checkpoint_path=checkpoint_path, resume_from=args.resume_checkpoint_path)

if __name__ == '__main__':
    main()
