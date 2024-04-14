import torch
from torch import nn

from modules import FFN, PositionalEmbedding, SelfAttention
from utils import get_causal_mask


class ReZeroTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dff, dropout=0.1):
        super().__init__()
        self.register_parameter(
            'alpha',
            nn.Parameter(torch.zeros(1), requires_grad=True)
        )
        self.self_attn = SelfAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, dff)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, attn_mask=None):
        x = self.self_attn(inputs, attn_mask) * self.alpha + inputs
        x = self.dropout(x)
        x = self.ffn(x) * self.alpha + x
        x = self.dropout(x)
        return x
    

class ReZeroTransformer(nn.Module):
    def __init__(self, d_model, n_heads, dff, n_blocks, maxlen, vocab_size, dropout=0.1):
        super().__init__()
        self.te = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEmbedding(maxlen, d_model)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(ReZeroTransformerBlock(d_model, n_heads, dff, dropout))
            
        self.register_buffer('causal_mask', get_causal_mask(maxlen)) 
        
    def forward(self, input_ids):
        L = input_ids.size(1)
        te = self.te(input_ids)
        pe = self.pe(L)
        
        x = te + pe
        
        for block in self.blocks:
            x = block(x, self.causal_mask[:L, :L])
        
        logits = torch.matmul(x, self.te.weight.T)
        return logits