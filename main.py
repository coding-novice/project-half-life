#!/usr/bin/env python
import argparse
import json
import os
import shutil
import torch

from model import SalukiModel
from train import SalukiTrainer

# TODO: Import the actual Dataloader when it's implemented by the team member.
# from dataset import RnaDataset

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
    parser.add_argument('params_file', type=str, help='Path to params.json')
    parser.add_argument('data_dirs', type=str, nargs='+', help='List of data directories (one per species)')
    parser.add_argument('-o', '--out_dir', type=str, default='train_out', help='Output directory [Default: train_out]')
    parser.add_argument('--wandb_project', type=str, default=None, help='Weights & Biases project name for logging')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on (cuda/cpu)')
    
    args = parser.parse_args()

    # Read model parameters
    with open(args.params_file, 'r') as params_open:
        params = json.load(params_open)
        
    params_model = params.get('model', {})
    params_train = params.get('train', {})

    # Ensure output directory exists and copy params file
    os.makedirs(args.out_dir, exist_ok=True)
    out_params_file = os.path.join(args.out_dir, 'params.json')
    if os.path.abspath(args.params_file) != os.path.abspath(out_params_file):
        shutil.copy(args.params_file, out_params_file)

    train_data = []
    eval_data = []
    species_names = []

    # Initialize dataloaders
    batch_size = params_train.get('batch_size', 64)
    for data_dir in args.data_dirs:
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
    save_path = os.path.join(args.out_dir, 'model_best.pt')
    trainer.train(save_path=save_path)

if __name__ == '__main__':
    main()
