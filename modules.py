from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.nn import functional as F

from utils import get_causal_mask, init_weights
import config as C


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.dim_head = d_model // n_heads
        self.d_model = d_model
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        
        self.register_buffer('scale', torch.FloatTensor([self.dim_head]).sqrt())
        self.dropout = nn.Dropout(dropout)
        
        self.K_cache = None
        self.V_cache = None
        
    def clear_kv_cache(self):
        self.K_cache = None
        self.V_cache = None
        
    def forward(self, inputs, attn_mask=None):
        B, L, _ = inputs.size()
        
        if self.K_cache is None:
            qkv = self.qkv(inputs)
            qkv = qkv.view(B, L, self.n_heads, -1)
            qkv = qkv.permute(0, 2, 1, 3)
            Q, K, V = qkv.split(self.dim_head, dim=-1)
            
            self.K_cache = K
            self.V_cache = V
        else:
            last_qkv = self.qkv(inputs[:, -1:, :])
            last_qkv = last_qkv.view(B, 1, self.n_heads, -1)
            last_qkv = last_qkv.permute(0, 2, 1, 3)
            Q, last_K, last_V = last_qkv.split(self.dim_head, dim=-1)
            # B x nheads x L x dhead
            self.K_cache = torch.cat([self.K_cache, last_K], dim=-2)
            self.V_cache = torch.cat([self.V_cache, last_V], dim=-2)
            L = 1
            
        scores = Q.matmul(self.K_cache.permute(0, 1, 3, 2)) / self.scale
        if attn_mask is not None:
            scores += attn_mask
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        outputs = scores.matmul(self.V_cache)
        
        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        outputs = outputs.view(B, L, -1)
        
        return outputs
        
        
class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, d_model):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.pem = nn.Parameter(torch.randn(max_length, d_model))
    
    def forward(self, length):
        return self.pem[None, :length]
    

class FFN(nn.Sequential):
    def __init__(self, d_model, dff):
        super().__init__()
        self.append(nn.Linear(d_model, dff))
        self.append(nn.GELU())
        self.append(nn.Linear(dff, d_model))
        

class Transformer(nn.Module, ABC):
    def __init__(self, d_model, n_heads, dff, n_blocks, maxlen, vocab_size, dropout=0.1, **kwargs):
        super().__init__()
        self.n_blocks = n_blocks
        self.te = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEmbedding(maxlen, d_model)
        self.outproj = nn.Linear(d_model, vocab_size)
        
        self.blocks = nn.ModuleList()
        self._build_transformer_blocks(n_blocks, d_model, n_heads, dff, dropout)
            
        self.register_buffer('causal_mask', get_causal_mask(maxlen))
        self.apply(init_weights)
    
    def clear_kv_cache(self):
        for block in self.blocks:
            block.clear_kv_cache()
    
    @abstractmethod
    def _build_transformer_blocks(self, n_blocks, d_model, n_heads, dff, dropout):
        pass
        
    def forward(self, input_ids, use_causal_mask=True):
        L = input_ids.size(1)
        te = self.te(input_ids)
        pe = self.pe(L)
        
        x = te + pe
        
        for block in self.blocks:
            if use_causal_mask:
                x = block(x, self.causal_mask[:L, :L])
            else:
                x = block(x)
        logits = self.outproj(x)
        return logits
        

class VanillaTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dff, dropout=0.1):
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.self_attn = SelfAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, dff)
        self.dropout = nn.Dropout(dropout)
        
    def clear_kv_cache(self):
        self.self_attn.clear_kv_cache()
        
    def forward(self, inputs, attn_mask=None):
        x = self.layer_norm1(inputs)
        x = self.self_attn(x, attn_mask) + x
        x = self.layer_norm1(x)
        x = self.dropout(x)
        x = x + self.ffn(x)
        x = self.dropout(x)
        return x


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
    
    def clear_kv_cache(self):
        self.self_attn.clear_kv_cache()
        
    def forward(self, inputs, attn_mask=None):
        x = self.self_attn(inputs, attn_mask) * self.alpha + inputs
        x = self.dropout(x)
        x = self.ffn(x) * self.alpha + x
        x = self.dropout(x)
        return x
    
    
class VanillaTransformer(Transformer):
    def _build_transformer_blocks(self, n_blocks, d_model, n_heads, dff, dropout):
        for _ in range(n_blocks):
            self.blocks.append(VanillaTransformerBlock(d_model, n_heads, dff, dropout))
    

class ReZeroTransformer(Transformer):
    def _build_transformer_blocks(self, n_blocks, d_model, n_heads, dff, dropout):
        for _ in range(n_blocks):
            self.blocks.append(ReZeroTransformerBlock(d_model, n_heads, dff, dropout))
            
            
def get_model_config():
    return dict(
        d_model=C.D_MODEL,
        n_heads=C.N_HEADS,
        dff=C.DFF,
        n_blocks=C.N_BLOCKS,
        maxlen=C.MAXLEN,
        vocab_size=C.VOCAB_SIZE,
        dropout=C.DROPOUT,
        architecture=C.ARCHITECTURE
    )
    
    
def get_model_from_config(settings=None):
    if settings is None:
        settings = get_model_config()
    
    if C.ARCHITECTURE == 'vanilla':
        model = VanillaTransformer(**settings)
    elif C.ARCHITECTURE == 'rezero':
        model = ReZeroTransformer(**settings)
    else:
        raise ValueError()
    return model