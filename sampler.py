from tokenizers import Tokenizer
import torch
from torch import nn
import torch.nn.functional as F

import config


class Sampler:
    def __init__(self, model: nn.Module, tokenizer: Tokenizer, device, robust_factor=3.0):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.robust_factor = robust_factor
        if device == 'cuda':
            self.dtype = torch.float16
        elif device == 'cpu':
            self.dtype = torch.bfloat16
    
    @torch.no_grad()
    def sample(self, seed, top_k=5, maxlen=config.MAXLEN):
        ids = self.tokenizer.encode(seed).ids
        
        while len(ids) <= maxlen:
            inputs = torch.tensor([ids], device=self.device)

            with torch.autocast(self.device, self.dtype, enabled=config.USE_AMP):
                logits = self.model(inputs)[0, -1]
                
                # scale to make the prediction robust
                logits = (logits - logits.mean()) * self.robust_factor
                
                # mask out-of-top-k classes
                i = torch.topk(logits, top_k).indices
                mask = F.one_hot(i, config.VOCAB_SIZE).sum(0)
                mask = torch.where(mask == 0, -float('inf'), 0)
                logits += mask
                logits = logits.softmax(-1)
                
            # sampling 
            pred = torch.multinomial(logits, num_samples=1).item()
            ids.append(pred)
            
            if pred == config.END_TOKEN_ID:
                break
        return self.tokenizer.decode(ids)
