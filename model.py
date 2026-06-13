import torch
import torch.nn as nn
import torch.nn.functional as F

def stochastic_shift(seq, shift_max=3, pad_value=0):
    """Randomly shift a (batch, length, depth) sequence right by 0..shift_max,
    padding the 5' end with zeros. Mirrors basenji StochasticShift(symmetric=False).
    Uses torch's RNG so it stays reproducible under torch.manual_seed."""
    shift = torch.randint(0, shift_max + 1, ()).item()   # one shift for the whole batch
    if shift == 0:
        return seq
    pad = seq.new_full((seq.size(0), shift, seq.size(2)), pad_value)
    return torch.cat([pad, seq[:, :-shift, :]], dim=1)    # prepend pad, drop the 3' tail

class SalukiModel(nn.Module):
    """
    PyTorch implementation of the Saluki RNA half-life prediction model.

    DOCUMENTATION OF ASSUMPTIONS ON EXTERNAL CODE (Data Loader):
    -------------------------------------------------------------------------
    1. Tensor Shape: Assumes the input tensor shape from the dataloader is 
       `(batch_size, 12288, 6)`. The length is 12288, and depth 6 corresponds 
       to 4 one-hot nucleotide channels + 2 binary tracks (first reading frame 
       and 5' splice sites). This is permuted internally to `(batch_size, 6, 12288)`
       to be compatible with PyTorch's Conv1d.
    2. Sequence Padding: Assumes sequences shorter than 12288 are padded at the 
       3' end (the right side of the sequence). This is crucial because the GRU 
       processes the sequence right-to-left (from padding towards the 5' end), 
       ensuring the information-dense 5' end is processed last and determines the 
       final hidden state.
    3. Multi-species Handling: Assumes there are multiple dataloaders (one per species) 
       and each batch contains exactly one species. The `forward` pass takes an 
       optional `species_index` to route the batch representation through the correct 
       output head.

    DOCUMENTATION OF ARCHITECTURE/IMPLEMENTATION CHANGES FROM TENSORFLOW:
    -------------------------------------------------------------------------
    1. Dimension Ordering: Keras Conv1D uses `(batch, length, channels)` while 
       PyTorch Conv1d uses `(batch, channels, length)`. We permute the dimensions 
       after the input layer and appropriately before/after LayerNorm.
    2. LayerNormalization: PyTorch's LayerNorm expects the normalized dimension 
       to be the last dimension. We apply it by permuting the tensor back to 
       `(batch, length, channels)`, applying LayerNorm across channels, and permuting back.
    3. GRU `go_backwards`: Keras GRU supports `go_backwards=True` natively. In PyTorch, 
       we emulate this by flipping the sequence along the time dimension `torch.flip(x, dims=[1])` 
       before passing it to the GRU, and extracting the final output state.
    4. Dense Layers: Keras `Dense` is directly replaced by PyTorch `Linear`.
    5. BatchNorm momentum: Keras and PyTorch define BatchNorm momentum as opposites
       (Keras: running = m*running + (1-m)*batch; PyTorch: running = (1-m)*running + m*batch).
       `bn_momentum` here is the *Keras* value from params.json (0.90); we convert it
       internally with `1 - bn_momentum`. We also set `eps=1e-3` to match Keras'
       BatchNormalization default (PyTorch's default is 1e-5).
    6. Weight initialization: Keras used `he_normal` for every kernel. We replicate
       this in `_init_weights` (Kaiming-normal for Conv/Linear and the GRU input
       kernel, orthogonal for the GRU recurrent kernel, zeros for all biases).
    7. L2 regularization: Keras applied `kernel_regularizer=l2(l2_scale)` to the conv
       kernels, the GRU *input* kernel, and the penultimate dense -- but NOT to the
       output heads, biases, norm params, or the GRU recurrent kernel. `l2_loss()`
       reproduces exactly that penalty for the trainer to add to the loss.
    8. Data augmentation: Keras wrapped the input in StochasticShift(augment_shift,
       symmetric=False), which during training shifts the sequence right by a random
       0..augment_shift positions (params.json: 3), padding the 5' end with zeros.
       `stochastic_shift()` reproduces this; it runs only when `self.training` is set.
    """
    def __init__(self, seq_length=12288, seq_depth=6, filters=64, kernel_size=5,
                 dropout=0.3, num_layers=6, heads=2, ln_epsilon=0.007,
                 bn_momentum=0.90, bn_epsilon=1e-3, l2_scale=0.001, augment_shift=3):
        super(SalukiModel, self).__init__()
        self.seq_length = seq_length
        self.seq_depth = seq_depth
        self.filters = filters
        self.l2_scale = l2_scale
        self.augment_shift = augment_shift

        # Convert the Keras-style BatchNorm momentum (params.json: 0.90) to PyTorch's
        # convention: 0.90 -> 0.10. See note 5 in the class docstring.
        torch_bn_momentum = 1.0 - bn_momentum

        # Initial Convolution
        self.initial_conv = nn.Conv1d(in_channels=seq_depth, out_channels=filters, 
                                      kernel_size=kernel_size, bias=False)
        
        # Middle Convolutions
        self.middle_blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleDict({
                'ln': nn.LayerNorm(filters, eps=ln_epsilon),
                'conv': nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size),
                'dropout': nn.Dropout(p=dropout),
                'pool': nn.MaxPool1d(kernel_size=2)
            })
            self.middle_blocks.append(block)
            
        # Aggregate Sequence (GRU)
        self.gru_ln = nn.LayerNorm(filters, eps=ln_epsilon)
        # Note: input to GRU will be (batch, seq_len, input_size)
        self.gru = nn.GRU(input_size=filters, hidden_size=filters, batch_first=True)
        
        # Penultimate Dense
        self.penultimate_bn = nn.BatchNorm1d(num_features=filters, momentum=torch_bn_momentum, eps=bn_epsilon)
        self.penultimate_dense = nn.Linear(filters, filters)
        self.penultimate_dropout = nn.Dropout(p=dropout)

        # Final Representation
        self.final_bn = nn.BatchNorm1d(num_features=filters, momentum=torch_bn_momentum, eps=bn_epsilon)

        # Output Heads (For N species)
        self.output_heads = nn.ModuleList([nn.Linear(filters, 1) for _ in range(heads)])

        # Match Keras 'he_normal' kernel initialization across all layers.
        self._init_weights()

    def _init_weights(self):
        """Replicate Keras' `he_normal` kernel init (note 6 in the class docstring)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                # he_normal == Kaiming-normal for a ReLU non-linearity.
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if 'weight_ih' in name:      # input kernel -> he_normal
                        nn.init.kaiming_normal_(p, nonlinearity='relu')
                    elif 'weight_hh' in name:    # recurrent kernel -> Keras GRU default
                        nn.init.orthogonal_(p)
                    elif 'bias' in name:
                        nn.init.zeros_(p)

    def l2_loss(self):
        """
        L2 penalty matching Keras `kernel_regularizer=l2(l2_scale)`.

        Keras adds `l2_scale * sum(w**2)` to the loss for the regularized kernels only:
        the conv kernels, the GRU *input* kernel, and the penultimate dense -- NOT the
        output heads, biases, norm params, or the GRU recurrent kernel. The trainer adds
        the returned value to the data loss before `backward()`.
        """
        if self.l2_scale == 0:
            return self.initial_conv.weight.new_zeros(())
        reg = self.initial_conv.weight.pow(2).sum()
        for block in self.middle_blocks:
            reg = reg + block['conv'].weight.pow(2).sum()
        reg = reg + self.gru.weight_ih_l0.pow(2).sum()
        reg = reg + self.penultimate_dense.weight.pow(2).sum()
        return self.l2_scale * reg

    def forward(self, x, species_index=0):
        # x shape: (batch_size, 12288, 6)

        # Training-only data augmentation (Keras StochasticShift). self.training is
        # toggled by model.train()/model.eval(), matching the original `if training:`.
        # Applied while still (batch, length, channels) so the length axis is dim=1.
        if self.training and self.augment_shift:
            x = stochastic_shift(x, shift_max=self.augment_shift)

        # Permute to (batch_size, 6, 12288) for Conv1d
        x = x.permute(0, 2, 1)
        
        # Initial Convolution
        x = self.initial_conv(x)
        
        # Middle convolutions
        for block in self.middle_blocks:
            # LayerNorm over channels: needs (batch, length, channels)
            x = x.permute(0, 2, 1)
            x = block['ln'](x)
            x = x.permute(0, 2, 1) # Back to (batch, channels, length)
            
            x = F.relu(x)
            x = block['conv'](x)
            x = block['dropout'](x)
            x = block['pool'](x)
            
        # Aggregate Sequence
        # Transpose back for LayerNorm and GRU
        x = x.permute(0, 2, 1) # (batch, length, channels)
        x = self.gru_ln(x)
        x = F.relu(x)
        
        # GRU go_backwards logic: flip along the sequence length dimension (dim=1)
        x_flipped = torch.flip(x, dims=[1])
        
        # Pass to GRU
        gru_out, hidden = self.gru(x_flipped)
        # We want the output at the final timestep (which corresponds to the 5' end of the transcript)
        # gru_out is (batch, seq_len, hidden_size)
        x = gru_out[:, -1, :] # (batch, hidden_size)
        
        # Penultimate Dense
        x = self.penultimate_bn(x)
        x = F.relu(x)
        x = self.penultimate_dense(x)
        x = self.penultimate_dropout(x)
        
        # Final Representation
        x = self.final_bn(x)
        x = F.relu(x)
        
        # Output Head Selection
        # Assuming the entire batch corresponds to a single species
        head = self.output_heads[species_index]
        prediction = head(x)
        
        return prediction
