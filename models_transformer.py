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
        """
        Calculate the positional encoding and add it to the input embeddings.
        """
        x = x + (self.pe[:, :x.size(1)].requires_grad_(False))
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    """
    Class for the Layer Normalization of the Transformer model.
    
    The output is the input embeddings normalized by the mean and variance of the embeddings.
    
    The Size of the output is (seq_len, d_model)
    
    NOTE: layer normalization means that the normalization is done for each feature, 
    whereas batch normalization normalizes the features for each batch. 
    
    A good explanation can be found in: 
    
    https://paperswithcode.com/method/layer-normalization
    
    We also have alpha and bias as learnable parameters, so we can learn the 
    normalization. The formula is:
    
    LN(x) = alpha * (x - mean(x)) / (std(x) + eps) + bias
    
    where alpha and bias are learnable parameters, and eps is a small number to
    avoid division by zero.
    
    The layer normalization shall be applied to the output of the sublayer,
    before the skip connection.
    """
    def __init__(self,
                 d_model : int,
                 eps     : float = 1e-6,
                ) -> None:
        super(LayerNormalization, self).__init__()
        self.d_model = d_model
        self.eps     = eps
        self.alpha   = nn.Parameter(torch.ones(1)) # Multiplicative
        self.bias    = nn.Parameter(torch.zeros(1)) # Additive

    def forward(self, x):
        """
        Layer normalization formula.
        
        x : (batch, seq_len, d_model)
        
        mean : (batch, seq_len)
        std  : (batch, seq_len)
        
        The output is the input embeddings normalized by the mean and variance of the embeddings.
        
        The Size of the output is (seq_len, d_model)

        """
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
        """
        Vanilla feed forward neural network.
        """
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttentionBlock(nn.Module):
    """
    Class for the MA attention block of the Transformer model.
    
    The MultiHead Attention mechanism is the implementation of the attention mechanism,
    but splitting the Q, K and V matrices into h submatrices, so that the attention mechanism
    is applied h times, and then concatenated.
    
    Note also that we will have weight matrices for Q, K, V and O, so that we can learn the
    attention mechanism.
    
    The output is the input embeddings passed through the MultiHead Attention mechanism.
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
        self.attention_scores = None

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
        """
        Computes the attention mechanism. 
        
        Remember that: 
        q : query (batch, h, seq_len, d_k)
        k : key   (batch, h, seq_len, d_k)
        v : value (batch, h, seq_len, d_k)
        
        Attention(q, k, v) = softmax(q @ k^T / sqrt(d_k)) @ v
        
        Remember also that the mask is a tensor with the same shape as the attention scores, 
        that will put -inf in the positions where the mask is 0, so that the softmax will
        ignore those positions (i.e. it will produce 0 in the output).
        
        Note also that the dropout is applied to the attention scores.
        
        Finally, this method is static, so it can be called without an instance of the class.
        """
        d_k = query.size(-1)

        attention_scores = (query @ key.transpose(-2, -1)) /torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = torch.softmax(attention_scores, dim=-1) # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value) , attention_scores

    def forward(self,
                q : torch.Tensor,
                k : torch.Tensor,
                v : torch.Tensor,
                mask=None):
        """
        For most tasks, q, k and v are the same tensor, but they are different for the decoder.
        
        The output is the input embeddings passed through the MultiHead Attention mechanism.
        
        The Size of the output is (seq_len, d_model)
        The process would be:
        
        (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        
        Which is shape[0] is the batch size,
        - shape[1] is the sequence length,
        - self.h is the number of heads,
        = self.d_k is the dimension of the head

        NOTE: We need to transpose the dimensions to have the shape (batch, h, seq_len, d_k)
        because we want to apply the attention mechanism to each head.
    
        The attention mechanism will be applied to each head and matrix, and then concatenated.
        
        Size of the output is (batch, seq_len, d_model)
        """
        query = self.W_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.W_k(k)   # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.W_v(v) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key   = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # Apply the attention mechanism
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # The process would be: \
            # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.d_model)

        x = self.W_o(x) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        return x


class ResidualConnection(nn.Module):
    """
    The skip connection from the paper. It is the sum of input with the 
    output of the sublayer, no matter if it is the ff block or the MHA block.
    
    Thus, it shall take the input x and the sublayer. Note that this is the
    Add & Norm block from the paper.
    
    In the video, it is called Residual Connection. I am keeping this name to 
    credit the video. 
    
    A small difference is that the layer normalization 
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
    """
    This is the Encoder Block from the paper. It takes a multiheaded attention block
    and a feed forward block, and applies the residual connection to both of them.
    
    The output is the input embeddings passed through the Encoder Block.
    """
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
    """ 
    Class to call a list of encoder blocks and normalize the output using
    the LayerNormalization class.
    
    This class take a list of encoder blocks and the dimension of the model, and 
    applies the blocks to the input.
    """
    def __init__(self, layers : nn.ModuleList, d_model : int) -> None:
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)
        
    def forward(self, x, mask):
        """Simple forward method to apply the encoder blocks to the input."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    """Similar to the EncoderBlock, but with the cross attention block.
    The cross-attention is for the second multiheaded attention block. 
    """
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
        """
        Simple forward method to apply the decoder blocks to the input.
        """
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
        """
        Class to call a list of decoder blocks and normalize the output using
        the LayerNormalization class.
        """
        def __init__(self, layers : nn.ModuleList, d_model : int) -> None:
            super(Decoder, self).__init__()
            self.layers = layers
            self.norm = LayerNormalization(d_model)
            
        def forward(self, x, encoder_output, src_mask, tgt_mask):
            """
            Simple forward method to apply the decoder blocks to the input.
            """
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


class Transformer(nn.Module):
    """
    Class that wraps the other class for a complete Transformer model.
    
    
    """
    def __init__(self,
                 encoder : Encoder,
                 decoder : Decoder,
                 src_embed : InputEmbeddings,
                 tgt_embed : InputEmbeddings,
                 src_pos : PositionalEncoding,
                 tgt_pos : PositionalEncoding,
                 projection : ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection

    def encode(self, src, src_mask):
        """
        Method to encode the input.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """
        Method to decode the information from the encoder.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        """
        Projection layer to get the output.
        """
        return self.projection(x)

def build_transformer(src_vocab_size : int,
                      tgt_vocab_size : int,
                      src_seq_len : int,
                      tgt_seq_len : int,
                      d_model : int = 512,
                      N : int = 6,
                      h : int = 8,
                      dropout : float = 0.1,
                      d_ff : int = 2048):
    # Create the embedding layers: 
    src_embed = InputEmbeddings(d_model = d_model, vocab_size = src_vocab_size)
    tgt_embed = InputEmbeddings(d_model = d_model, vocab_size = tgt_vocab_size)

    # Create the positional encoding layers:
    src_pos = PositionalEncoding(d_model = d_model, seq_len = src_seq_len, dropout = dropout)
    tgt_pos = PositionalEncoding(d_model = d_model, seq_len = tgt_seq_len, dropout = dropout)

    # Create the encoder blocks:
    encoder_blocks = []
    for _ in range(N):
        # Adding the self-attention block:
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model = d_model,
                                                               h=h,
                                                               dropout = dropout)

        # Adding the feed forward block:
        feef_forward_block = FeedForwardBlock(d_model = d_model,
                                                d_ff = d_ff,
                                                dropout = dropout)

        # Adding the encoder block:
        encoder_block = EncoderBlock(self_attention_block = encoder_self_attention_block,
                                     feed_forward_block = feef_forward_block,
                                     dropout = dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks:
    decoder_blocks = []
    for _ in range(N):
        # Adding the self-attention block:
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model = d_model,
                                                               h=h,
                                                               dropout = dropout)

        # Adding the cross-attention block:
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model = d_model,
                                                               h=h,
                                                               dropout = dropout)

        # Adding the feed forward block:
        feef_forward_block = FeedForwardBlock(d_model = d_model,
                                                d_ff = d_ff,
                                                dropout = dropout)

        # Adding the decoder block:
        decoder_block = DecoderBlock(self_attention_block = decoder_self_attention_block,
                                     cross_attention_block = decoder_cross_attention_block,
                                     feed_forward_block = feef_forward_block,
                                     dropout = dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder:
    encoder = Encoder(layers = nn.ModuleList(encoder_blocks), d_model = d_model)
    decoder = Decoder(layers = nn.ModuleList(decoder_blocks), d_model = d_model)

    # Create the projection layer:
    projection = ProjectionLayer(d_model = d_model, vocab_size = tgt_vocab_size)

    # Create the transformer model:
    model = Transformer(encoder = encoder,
                        decoder = decoder,
                        src_embed = src_embed,
                        tgt_embed = tgt_embed,
                        src_pos = src_pos,
                        tgt_pos = tgt_pos,
                        projection = projection)
    
    # Initialize the parameters:
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model