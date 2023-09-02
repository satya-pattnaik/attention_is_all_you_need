import torch.nn as nn
import math
import torch
"""
A simple lookup table that stores embeddings of a fixed dictionary and size.

This module is often used to store word embeddings and retrieve them using 
indices. The input to the module is a list of indices, 
and the output is the corresponding word embeddings.
"""
class InputEmbeddings(nn.Module):
    def __init__(self, dim_model: int, vocab_size:int)->None:
        super().__init__()
        self.dim_model = dim_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim_model)

    def forward(self, x):
        #(batch_size, seq_len) --> (batch, seq_len, dim_model)
        #Multiply by sqrt(model) to scale embeddings according to the paper

        return self.embedding(x) * math.sqrt(self.dim_model)

class PositionalEncoding(nn.Module):

    def __init__(self, dim_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #Create a matrix of shape(seq_len, dim_model)
        pe = torch.zeros(self.seq_len, self.dim_model)

        #Create a vector of shape(seq_len)
        #(seq_len, 1)
        position = torch.\
                    arange(0, seq_len, dtype=torch.float).\
                    unsqueeze(1)

        #Create a vector of shape(dim_model)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * \
            (-math.log(10000.0)/self.dim_model)
                             )

        #Apply sine to even indices
        #sin(position ** (10000 ** (2i/dim_model)))
        pe[:, 0::2] = torch.sin(position * div_term)

        #Apply cosine to odd indices
        #cos(position ** (10000 ** (2i/dim_model)))
        pe[:, 1::2] = torch.sin(position * div_term)

        # Add a batch dimesnion to the positional encoding
        #(1,seq_len,dim_model)
        pe = pe.unsqueeze(0)

        #Register the positional embedding as a buffer
        #`If you have parameters in your model, which should be saved
        #and restored in the state_dict, but not trained by the optimizer,
        #you should register them as buffers.`
        self.register_buffer('pe',pe)

    def forward(self, x):
        #(batch, seq_len, dim_model)
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)