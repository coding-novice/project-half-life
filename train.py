import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
    def __init__(self, model, train_dataloaders, eval_dataloaders, params, device='cuda'):
        self.model = model.to(device)
        self.train_dataloaders = train_dataloaders
        self.eval_dataloaders = eval_dataloaders
        self.params = params
        self.device = device
        
        self.num_datasets = len(self.train_dataloaders)
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
        
        # Scheduler
        # Check if cyclical LR params exist. The TF trainer optionally falls back to standard LR.
        if 'train_epochs_cycle1' in self.params and 'maximal_learning_rate' in self.params:
            train_epoch_batches = sum(len(dl) for dl in self.train_dataloaders)
            step_size = self.params['train_epochs_cycle1'] * train_epoch_batches
            
            self.scheduler = Cyclical1LearningRate(
                self.optimizer,
                initial_learning_rate=self.params.get('initial_learning_rate', lr),
                maximal_learning_rate=self.params['maximal_learning_rate'],
                final_learning_rate=self.params.get('final_learning_rate', lr),
                step_size=step_size
            )
        else:
            self.scheduler = None

    def train(self, save_path='model_best.pt'):
        # Precompute batches per epoch for all species
        train_epoch_batches = [len(dl) for dl in self.train_dataloaders]
        dataset_indexes = []
        for di in range(self.num_datasets):
            dataset_indexes += [di] * train_epoch_batches[di]
        dataset_indexes = np.array(dataset_indexes)
        
        best_valid_r = -float('inf')
        unimproved = 0
        
        for epoch in range(self.train_epochs_max):
            if epoch >= self.train_epochs_min and unimproved > self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            self.model.train()
            np.random.shuffle(dataset_indexes)
            
            # Get iterators
            train_iters = [iter(dl) for dl in self.train_dataloaders]
            
            t0 = time.time()
            epoch_losses = [0.0] * self.num_datasets
            epoch_steps = [0] * self.num_datasets
            
            for di in dataset_indexes:
                try:
                    x, y = next(train_iters[di])
                except StopIteration:
                    continue
                
                # Move to device and ensure types are correct (float32)
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)
                
                self.optimizer.zero_grad()
                pred = self.model(x, species_index=di)
                loss = self.loss_fn(pred, y)
                loss.backward()
                
                # Gradient Clipping
                if 'global_clipnorm' in self.params:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.params['global_clipnorm'])
                    
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                epoch_losses[di] += loss.item()
                epoch_steps[di] += 1
                
            print(f"Epoch {epoch} - {time.time() - t0:.1f}s")
            
            # Validation phase
            self.model.eval()
            combined_valid_r = 0.0
            
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
                    
                    combined_valid_r += val_r
                    
                    print(f"  Data {di} - train_loss: {train_loss:.4f} - valid_loss: {val_loss:.4f} - valid_r: {val_r:.4f}")

            # Checkpoint best model
            if combined_valid_r > best_valid_r:
                print('  - best!', flush=True)
                unimproved = 0
                best_valid_r = combined_valid_r
                torch.save(self.model.state_dict(), save_path)
            else:
                unimproved += 1
