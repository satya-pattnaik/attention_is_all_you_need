import torch
import torch.nn as nn
from layer_normalization import LayerNormalization
import math

class FeedForwardBlock(nn.Module):

    def __init__(self,dim_model: int, dim_ff:int, dropout:float):
        super().__init__()

        #Squeeze and Stretch

        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_model)

    def forward(self, x):

        #(batch, seq_len, dim_model) -> (batch, seq_len, dim_ff) -> (batch, seq_len, dim_model)

        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, dim_model: int, h:int, dropout:float) -> None:
        super().__init__()

        #Embedding Vector Size
        self.dim_model = dim_model
        #Number of heads
        self.h = h

        #Make sure dim_model is divisible by h
        assert(dim_model % h == 0, "dim_model is not divisible by h")

        #Dimension of vector seen by each head
        self.dim_k = self.dim_model // h

        self.w_q = nn.Linear(self.dim_model, self.dim_model, bias=False) #Wq
        self.w_k = nn.Linear(self.dim_model, self.dim_model, bias=False) #Wk
        self.w_v = nn.Linear(self.dim_model, self.dim_model, bias=False) #Wv
        self.w_o = nn.Linear(self.dim_model, self.dim_model, bias=False) #Wo

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        dim_k = query.shape[-1]

        # Applying formula from paper
        # (batch, h, seq_len, dim_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dim_k)

        if mask is not None:
            #Write a very low value (indicating -inf) to the positions
            #where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)

        #(batch, h, seq_len, seq_len)
        #Apply softmax
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, dim_k)
        # Return attentions scores which can be used for visualization

        return (attention_scores @ value), attention_scores

def forward(self, q, k, v, mask):

    #(batch, seq_len, dim_model) --> (batch, seq_len, dim_model)
    query = self.w_q(q)

    # (batch, seq_len, dim_model) --> (batch, seq_len, dim_model)
    key = self.w_k(k)

    # (batch, seq_len, dim_model) --> (batch, seq_len, dim_model)
    value = self.w_v(v)

    # (batch, seq_len, dim_model) -->
    # (batch, seq_len, h , dim_k) --> (batch, seq_len, h , dim_k)
    #b, sl, dm --> b, sl, h, dm/h --> b, h, sl, dm/h
    query = query.view(query.shape[0], query.shape[1],\
                       self.h, self.dim_k).transpose(1,2)

    key = key.view(key.shape[0], key.shape[1],\
                       self.h, self.dim_k).transpose(1,2)

    value = value.view(value.shape[0], value.shape[1],\
                       self.h, self.dim_k).transpose(1,2)

    #Calculate Attention
    x, self.attention_scores = MultiHeadAttentionBlock.attention(
                                                        query,
                                                        key,
                                                        value,
                                                        mask,
                                                        self.dropout
                                                        )
    #Comibine all the heads together
    # (batch, h, seq_len, dim_k) --> (batch, seq_len, h, dim_k) --> (batch, seq_len, dim_model)

    x = x.transpose(1,2).contigious().view(x.shape[0], -1, self.h * self.dim_k)
    #batch, h, seq_len, dim_k << transpose
    #batch, seq_len, h , dim_k << view
    #batch, -1, h*dim_k

    #Multiply by Wo
    #(batch, seq_len, dim_model) --> (batch, seq_len, dim_model)
    return self.w_o(x)
