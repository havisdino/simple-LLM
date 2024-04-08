import torch
from torcheval.metrics.functional.text import perplexity
from torcheval.metrics.functional.classification import multiclass_accuracy
from config import *


@torch.no_grad()
def get_perplexity(model, input_ids, target_ids):
    model.eval()
    logits = model(input_ids)
    ppl = perplexity(logits, target_ids, ignore_index=END_TOKEN_ID)
    return ppl


@torch.no_grad()
def get_accurracy(model, input_ids, target_ids):
    model.eval()
    logits = model(input_ids)
    D = logits.size(-1)
    acc = multiclass_accuracy(logits.view(-1, D), target_ids.view(-1))
    return acc


@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    ppls = []
    accs = []
    
    for input_ids, target_ids in data_loader:
        input_ids = input_ids.to(DEVICE)
        target_ids = target_ids.to(DEVICE)
        
        ppls.append(get_perplexity(model, input_ids, target_ids))
        accs.append(get_accurracy(model, input_ids, target_ids))
    
    ppl = sum(ppls) / len(ppls)
    acc = sum(accs) / len(accs)
    return ppl, acc
        
        
        