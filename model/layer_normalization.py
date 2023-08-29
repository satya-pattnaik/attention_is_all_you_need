import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):

    def __init__(self, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps

        #alpha is a learnable parameter
        self.alpha = nn.Parameter(torch.ones(1))
        #bias is a learnable paramter
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        #x:(batch, seq_len, hidden_size)
        #Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)

        # Keep the dimension for broadcasting
        std = x.mean(dim=-1, keepdim=True)

        return self.alpha*(x-mean)/(std+self.eps) + self.bias