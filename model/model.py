import torch
import torch.nn as nn
from layer_normalization import LayerNormalization
import math
from embeddings import InputEmbeddings, PositionalEncoding
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

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: nn.Dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.Module) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 cross_attention_block:MultiHeadAttentionBlock, dropout:nn.Dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.dropout = dropout
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output,
                                                                                 encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, dim_model, vocab_size) -> None:
            super().__init__()
            self.proj = nn.Linear(dim_model, vocab_size)

    def forward(self, x):
        #(batch, seq_len, dim_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)

class Transfomer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer
                 ):
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        #(batch, seq_len, dim_model)
        src = self.src_embed(src)
        src = self.src_pos(src)

        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt:torch.Tensor,
               tgt_mask:torch.Tensor):
        #(batch, seq_len, dim_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self,x):
        #(batch, seq_len, vocab_size)
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size:int, src_seq_len: int,
                      tgt_seq_len: int, dim_model:int = 512, dropout: float=0.8, h:int=8, N:int=6,
                      dim_ff:int=2048
                      ):
    #Create the embedding layers

    src_embed = InputEmbeddings(dim_model, src_vocab_size)
    tgt_embed = InputEmbeddings(dim_model, tgt_vocab_size)

    #Create the positional encoding layers
    src_pos = PositionalEncoding(dim_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(dim_model, tgt_seq_len, dropout)

    #Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(dim_model, h, dropout)
        feed_forward_block = FeedForwardBlock(dim_model, dim_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(dim_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(dim_model, h, dropout)
        feed_forward_block = FeedForwardBlock(dim_model, dim_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block)
        decoder_blocks.append(decoder_block)

    #Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #Create the projection layer
    projection_layer = ProjectionLayer(dim_model, tgt_vocab_size)

    #Create the transoformer
    transformer = Transfomer(encoder, decoder, src_embed , tgt_embed, src_pos, tgt_pos, projection_layer)

    #Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer
