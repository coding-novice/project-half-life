import torch
import torch.nn as nn
import torch.nn.functional as F

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
       BatchNormalization `momentum` defaults to `0.99` in Keras, which corresponds 
       to `0.1` in PyTorch (PyTorch uses `1 - momentum`).
    """
    def __init__(self, seq_length=12288, seq_depth=6, filters=64, kernel_size=5, 
                 dropout=0.3, num_layers=6, heads=2, ln_epsilon=0.007, bn_momentum=0.1):
        super(SalukiModel, self).__init__()
        self.seq_length = seq_length
        self.seq_depth = seq_depth
        self.filters = filters
        
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
        self.penultimate_bn = nn.BatchNorm1d(num_features=filters, momentum=bn_momentum)
        self.penultimate_dense = nn.Linear(filters, filters)
        self.penultimate_dropout = nn.Dropout(p=dropout)
        
        # Final Representation
        self.final_bn = nn.BatchNorm1d(num_features=filters, momentum=bn_momentum)
        
        # Output Heads (For N species)
        self.output_heads = nn.ModuleList([nn.Linear(filters, 1) for _ in range(heads)])

    def forward(self, x, species_index=0):
        # x shape: (batch_size, 12288, 6)
        
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
