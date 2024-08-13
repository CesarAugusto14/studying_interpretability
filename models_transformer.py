#
# author : @cesarasa
#
# Code for studying Transformer models

import torch 
import torch.nn as nn

class InputEmbeddings(nn.Module):
    """
    Class for the input embeddings of the Transformer model.
    vocab_size : is the number of words in the vocabulary, or the number of possible tokens.
    d_model    : is the dimension of the model, which is the size of the embeddings.
    
    Thus, for each x in the input, the output will be a tensor containing the sequence of embeddings.
    
    The output is the input embeddings multiplied by sqrt(d_model), as suggested in the paper.
    
    The Size of the output is (seq_len, d_model)
    """
    def __init__(self,
                 vocab_size : int,
                 d_model    : int,
                 ):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        # Multiply by sqrt(d_model) to stabilize the gradients, according to the paper
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
    
class PositionalEncoding(nn.Module):
    """
    Class for the positional encoding of the Transformer model.
    
    The output is the input embeddings summed with the positional encoding.
    
    The Size of the output is (seq_len, d_model)
    
    The positional encoding is calculated as follows:
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where pos is the position and i is the dimension of the model.
    """
    
    def __init__(self,
                 d_model : int,
                 seq_len : int,
                 dropout : float,
                ):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len

        # Create the positional encoding
        pe = torch.zeros(seq_len, d_model)  # (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)*-(torch.log(torch.tensor(10000.0)) / d_model))        
        # Apply the formula for the positional encoding, for even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the formula for the positional encoding, for odd dimensions
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.size(1)].requires_grad_(False))
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    """
    Class for the Layer Normalization of the Transformer model.
    
    The output is the input embeddings normalized by the mean and variance of the embeddings.
    
    The Size of the output is (seq_len, d_model)
    """
    def __init__(self,
                 d_model : int,
                 eps     : float = 1e-6,
                ) -> None:
        super(LayerNormalization, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # Additive    
    def forward(self, x):
        # Keepdim=True to keep the dimensions of the mean and std (for broadcasting)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    Class for the Feed Forward Block of the Transformer model.
    
    The output is the input embeddings passed through a feed forward neural network.
    
    The Size of the output is (seq_len, d_model)
    """
    def __init__(self,
                 d_model : int,
                 d_ff    : int,
                 dropout : float,
                ):
        super(FeedForwardBlock, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class