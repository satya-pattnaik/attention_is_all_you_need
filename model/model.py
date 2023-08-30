import torch
import torch.nn as nn

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