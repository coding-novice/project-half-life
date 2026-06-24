import os
import random
import time
import math
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from basenji import dna_io

from scipy.stats import pearsonr, spearmanr
def pearson(x, y):
    r = pearsonr(x, y)[0]
    return round(r, 4)
def spearman(x, y):
    r = spearmanr(x, y)[0]
    return round(r, 4)

def pearson_corrcoef(x, y):
    """Computes Pearson correlation coefficient between two 1D tensors."""
    x = x.view(-1)
    y = y.view(-1)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x - mean_x
    ym = y - mean_y
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
    r = r_num / (r_den + 1e-8)
    return r

class Cyclical1LearningRate(optim.lr_scheduler.LambdaLR):
    """
    A LearningRateSchedule that uses cyclical schedule.
    Matches the custom Keras Cyclical1LearningRate implementation in basenji/trainer.py.
    """
    def __init__(self, optimizer, initial_learning_rate, maximal_learning_rate, final_learning_rate, step_size):
        def lr_lambda(step):
            cycle = math.floor(1 + step / (2 * step_size))
            x = abs(step / step_size - 2 * cycle + 1)
            
            if step > 2 * step_size:
                lr = final_learning_rate
            else:
                lr = initial_learning_rate + (maximal_learning_rate - initial_learning_rate) * max(0, 1 - x)
            return lr / initial_learning_rate
            
        super(Cyclical1LearningRate, self).__init__(optimizer, lr_lambda)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class SalukiTrainer:
    """
    Trainer class for the Saluki PyTorch model.
    
    DOCUMENTATION OF ASSUMPTIONS ON EXTERNAL CODE (Data Loader):
    -------------------------------------------------------------------------
    1. Dataloader Output: Assumes dataloaders yield a tuple (x, y) where:
       - x is the input tensor of shape (batch_size, 12288, 6)
       - y is the target tensor of shape (batch_size, 1)
    2. Multi-Species Data: Assumes `train_dataloaders` and `eval_dataloaders` 
       are lists of PyTorch DataLoader objects, where each dataloader corresponds 
       to one species. The species index is implicitly the index of the dataloader 
       in the list (e.g., index 0 for Human, index 1 for Mouse).
    3. Batch Size: Assumes all dataloaders use the same batch_size, required for 
       ensuring constant epoch size when adding data from more species.
       
    DOCUMENTATION OF ARCHITECTURE/IMPLEMENTATION CHANGES FROM TENSORFLOW:
    -------------------------------------------------------------------------
    1. Optimization Step: Replaces Keras' hidden `.fit()` iterative loop with an 
       explicit PyTorch loop. Iterates dynamically between the multiple species 
       dataloaders.
    2. Batch Iteration Strategy: The original TF code computes total steps per epoch ahead 
       of time and intermixes batches randomly via a `self.dataset_indexes` array. We 
       emulate this behavior precisely by pre-computing a shuffled list of dataset indices 
       and cycling through the dataloader iterators accordingly, to ensure batches from 
       different species are interwoven.
    3. Metrics & Checkpointing: We manually compute Pearson R and MSE for each species 
       separately during the validation phase. We track the combined validation Pearson R 
       across all species to save the best model checkpoint and manage early stopping.
    4. Learning Rate Scheduler: The Cyclical1LearningRate from Keras `basenji/trainer.py`
       is implemented via PyTorch's `LambdaLR` scheduler, modifying the base LR dynamically 
       per step exactly as the original code did.
    """
    def __init__(self, model, train_dataloaders, eval_dataloaders, params, params_model, device='cuda', species_names=None, wandb_project=None, run_name=None, df_reporter=None, df_mpras=None):
        self.model = model.to(device)
        self.train_dataloaders = train_dataloaders
        self.eval_dataloaders = eval_dataloaders
        self.params = params
        self.params_model = params_model
        self.device = device
        self.df_reporter = df_reporter
        self.df_mpras = df_mpras
        
        self.num_datasets = len(self.train_dataloaders)
        self.species_names = species_names if species_names else [f"species_{i}" for i in range(self.num_datasets)]
        
        for name, dl in zip(self.species_names, self.train_dataloaders):
                    print(f"{name}: {len(dl.dataset)} train samples")

        self.use_wandb = wandb_project is not None
        if self.use_wandb:
            assert run_name is not None
            self.wandb_run = wandb.init(
                entity="project-half-life",
                project=wandb_project,
                name=run_name,
                config=self.params,
            )
            
        self.patience = self.params.get('patience', 25)
        self.train_epochs_min = self.params.get('train_epochs_min', 100)
        self.train_epochs_max = self.params.get('train_epochs_max', 250)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Optimizer setup based on TF's params.json
        lr = self.params.get('learning_rate', 0.0001)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr,
            betas=(self.params.get('adam_beta1', 0.90), self.params.get('adam_beta2', 0.998))
        )
        
        epoch_samples_limit = 21141 # human: 10221 train samples // mouse: 10920 train samples
        if epoch_samples_limit == 0:
            num_core_species = min(2, self.num_datasets)
            epoch_samples_limit = sum(len(self.train_dataloaders[i].dataset) for i in range(num_core_species))
            
        current_batch_size = self.train_dataloaders[0].batch_size
        self.train_steps_per_epoch = epoch_samples_limit // current_batch_size

        # Scheduler
        # Check if cyclical LR params exist. The TF trainer optionally falls back to standard LR.
        if 'train_epochs_cycle1' in self.params and 'maximal_learning_rate' in self.params:
            step_size = self.params['train_epochs_cycle1'] * self.train_steps_per_epoch
            
            self.scheduler = Cyclical1LearningRate(
                self.optimizer,
                initial_learning_rate=self.params.get('initial_learning_rate', lr),
                maximal_learning_rate=self.params['maximal_learning_rate'],
                final_learning_rate=self.params.get('final_learning_rate', lr),
                step_size=step_size
            )
        else:
            self.scheduler = None

    def train(self, save_path='model_best.pt', checkpoint_path='checkpoint.pt', resume_from=None):
        construct = self.df_reporter.values
        seq = self.df_mpras[[0]][0].values
        aa_len = int(len(construct[1][0])/3)
        coding = np.append(np.zeros(len(construct[0][0])), np.tile([1,0,0], aa_len))
        reporter = construct[0]+construct[1]+construct[2]
        df_mpras = self.df_mpras.rename(columns={0: 'seq', 1: 'measured'}).drop(columns=2).dropna()
        print(f"DEBUG: mpra shape: {df_mpras.shape}")

        # Precompute species sampling weights proportional to their dataset size
        species_samples = [len(dl.dataset) for dl in self.train_dataloaders]
        total_samples = sum(species_samples)
        species_weights = [s / total_samples for s in species_samples]
        
        start_epoch = 0
        best_valid_r = -float('inf')
        unimproved = 0
        
        if resume_from is not None:
            if os.path.isfile(resume_from):
                print(f"Resuming from checkpoint: {resume_from}")
                checkpoint = torch.load(resume_from, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_valid_r = checkpoint['best_valid_r']
                unimproved = checkpoint['unimproved']
                
                # Restore RNG states
                torch.set_rng_state(checkpoint['torch_rng_state'])
                if checkpoint['torch_cuda_rng_state'] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state(checkpoint['torch_cuda_rng_state'])
                np.random.set_state(checkpoint['numpy_rng_state'])
                random.setstate(checkpoint['random_rng_state'])
                print(f"Resumed successfully from epoch {start_epoch-1}.")
            else:
                raise FileNotFoundError(f"Checkpoint file not found: {resume_from}")
        
        for epoch in range(start_epoch, self.train_epochs_max):
            if epoch >= self.train_epochs_min and unimproved > self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            self.model.train()
            
            # Initialize infinite iterators for all dataloaders
            train_iters = [iter(cycle(dl)) for dl in self.train_dataloaders]
            
            t0 = time.time()
            epoch_losses = [0.0] * self.num_datasets
            epoch_steps = [0] * self.num_datasets
            
            for step in range(self.train_steps_per_epoch):
                di = np.random.choice(self.num_datasets, p=species_weights)
                x, y = next(train_iters[di])
                
                # Move to device and ensure types are correct (float32)
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)
                
                self.optimizer.zero_grad()
                pred = self.model(x, species_index=di)
                mse_loss = self.loss_fn(pred, y)
                # Add the Keras-equivalent L2 kernel penalty to the loss so its gradient
                # flows through Adam and is clipped together with the data gradient (this
                # matches Keras, where the regularization loss is part of the total loss).
                loss = mse_loss + self.model.l2_loss()
                loss.backward()

                # Gradient Clipping
                if 'global_clipnorm' in self.params:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.params['global_clipnorm'])

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # Report the MSE data loss only, so train_loss is comparable to valid_loss.
                epoch_losses[di] += mse_loss.item()
                epoch_steps[di] += 1
                
            print(f"Epoch {epoch} - {time.time() - t0:.1f}s")
            
            # Validation phase
            self.model.eval()
            combined_valid_r = 0.0
            log_metrics = {}
            
            with torch.no_grad():
                for di in range(self.num_datasets):
                    val_loss = 0.0
                    all_preds = []
                    all_targets = []
                    
                    for x, y in self.eval_dataloaders[di]:
                        x = x.to(self.device, dtype=torch.float32)
                        y = y.to(self.device, dtype=torch.float32)
                        
                        pred = self.model(x, species_index=di)
                        loss = self.loss_fn(pred, y)
                        
                        val_loss += loss.item()
                        all_preds.append(pred)
                        all_targets.append(y)
                        
                    val_loss /= max(1, len(self.eval_dataloaders[di]))
                    train_loss = epoch_losses[di] / max(1, epoch_steps[di])
                    
                    all_preds = torch.cat(all_preds)
                    all_targets = torch.cat(all_targets)
                    val_r = pearson_corrcoef(all_preds, all_targets).item()
                    
                    # adding up pearson corr scores from all species into a combined score:
                    combined_valid_r += val_r
                    
                    species_name = self.species_names[di]
                    log_metrics[f"{species_name}/train_loss"] = train_loss
                    log_metrics[f"{species_name}/valid_loss"] = val_loss
                    log_metrics[f"{species_name}/valid_r"] = val_r
                    log_metrics[f"{species_name}/train_batches"] = epoch_steps[di]
                    
                    print(f"  {species_name} (Data {di}) - train_loss: {train_loss:.4f} - valid_loss: {val_loss:.4f} - valid_r: {val_r:.4f}")

                if (epoch <= 20 and epoch % 5 == 0) or (epoch > 20 and epoch % 10 == 0):
                    df_perf = {
                        'seq': [],
                        'pred': []
                    }
                    for i in seq: # iterate through all sequences
                        batch = np.zeros((1, self.params_model.get('seq_length', 12288), 6))
                        myseq = (reporter+i+construct[3])[0]
                        batch[0,0:len(myseq),0:4] = dna_io.dna_1hot(myseq)
                        batch[0,0:len(coding),4] = coding
                        batch = torch.from_numpy(batch).float().to(self.device)
                        pred = self.model(batch)
                        df_perf['seq'].append(i)
                        df_perf['pred'].append(pred[0][0].cpu().numpy())

                    df_perf = pd.DataFrame(df_perf)
                    df_j_targ = df_perf.set_index('seq').drop_duplicates().join(
                        df_mpras.set_index('seq').drop_duplicates(), how='inner'
                    ).reset_index()
                    mpra_r = spearman(df_j_targ['pred'], df_j_targ['measured'])
                    log_metrics["mpra_r"] = mpra_r

            log_metrics["combined/valid_r"] = combined_valid_r
            if self.use_wandb:
                self.wandb_run.log(log_metrics, step=epoch)

            # Checkpoint best model
            if combined_valid_r > best_valid_r:
                print('  - best!', flush=True)
                unimproved = 0
                best_valid_r = combined_valid_r
                torch.save(self.model.state_dict(), save_path)
            else:
                # increment nr of epochs without improvement - early stopping if the nr exceeds the patience param
                unimproved += 1

            # Save full training checkpoint at the end of each epoch for resuming
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_valid_r': best_valid_r,
                'unimproved': unimproved,
                'torch_rng_state': torch.get_rng_state(),
                'torch_cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
            }
            if self.scheduler is not None:
                checkpoint_state['scheduler_state_dict'] = self.scheduler.state_dict()
                
            temp_checkpoint_path = checkpoint_path + ".tmp"
            torch.save(checkpoint_state, temp_checkpoint_path)
            os.replace(temp_checkpoint_path, checkpoint_path)
