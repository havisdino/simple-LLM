import torch
from torcheval.metrics.functional.text import perplexity
import config as C


@torch.no_grad()
def get_perplexity(model, input_ids, target_ids):
    model.eval()
    logits = model(input_ids)
    ppl = perplexity(logits, target_ids, ignore_index=C.END_TOKEN_ID)
    return ppl


@torch.no_grad()
def evaluate(model, data_loader, device, use_amp=True):
    model.eval()
    ppls = []
    
    for input_ids, target_ids in data_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        with torch.autocast(device, torch.float16, enabled=use_amp):
            ppls.append(get_perplexity(model, input_ids, target_ids))
    
    ppl = sum(ppls) / len(ppls)
    return ppl.item()
        
        
        