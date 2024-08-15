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
        self.linear1 = nn.Linear(d_model, d_ff) # W
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttentionBlock(nn.Module):
    """
    Class for the MA attention block of the Transformer model.
    """
    def __init__(self,
                 d_model : int,
                 h : int, 
                 dropout : float) -> None:
        super(MultiHeadAttentionBlock, self).__init__()
        
        # Initialize:
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h  # This way, we have d_k to be an integer
        
        # weight matrics:
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    @staticmethod # We don't need an instance of the class to call this method
    def attention(query,
                  key,
                  value,
                  mask,
                  dropout : nn.Dropout):
        d_k = query.size(-1)
        
        attention_scores = (query @ key.transpose(-2, -1)) /torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            
        attention_scores = torch.softmax(attention_scores, dim=-1) # (batch, h, seq_len, seq_len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return (attention_scores @ value) , attention_scores
        
    def forward(self,
                q,
                k,
                v,
                mask=None):
        
        query = self.W_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.W_k(k)   # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.W_v(v) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        
        # The process would be: (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        # Which is shape[0] is the batch size, shape[1] is the sequence length, self.h is the number of heads, self.d_k is the dimension of the head
        # We repeat the process for query, key and value
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key   = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        # Apply the attention mechanism
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)
        
        # The process would be: (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.d_model)
        
        x = self.W_o(x) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        
        return x

   
class ResidualConnection(nn.Module):
    """
    The skip connection from the paper. It is the sum of input with the 
    output of the sublayer, no matter if it is the ff block or the MHA block.
    """
    def __init__(self,
                 d_model : int,
                 dropout : float,
                ) -> None:
        super(ResidualConnection, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)
    def forward(self, x, sublayer):
        """
        x + dropout(MA(x)) or x + dropout(FF(x))
        """
        return x + self.dropout(sublayer(self.layer_norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block : FeedForwardBlock,
                 dropout : float,
                 ) -> None:
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(self_attention_block.d_model, dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    
    def __init__(self, layers : nn.ModuleList) -> None:
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block : FeedForwardBlock,
                 dropout : float):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(self_attention_block.d_model, dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
        
        def __init__(self, layers : nn.ModuleList) -> None:
            super(Decoder, self).__init__()
            self.layers = layers
            self.norm = LayerNormalization()
            
        def forward(self, x, encoder_output, src_mask, tgt_mask):
            for layer in self.layers:
                x = layer(x, encoder_output, src_mask, tgt_mask)
            return self.norm(x)
        
class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, vocab_size : int):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.linear(x), dim=-1)