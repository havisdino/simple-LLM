import torch
from torch import nn
from config import *
import math


def get_causal_mask(maxlen):
    causal_mask = torch.ones(maxlen, maxlen).tril() == 0
    causal_mask = torch.where(causal_mask, -float('inf'), 0)
    return causal_mask


def count_params(model):
    if isinstance(model, nn.DataParallel):
        n_params = sum(p.numel() for p in model.module.parameters())
    elif isinstance(model, nn.Module):
        n_params = sum(p.numel() for p in model.parameters())
        
    print(f'Parameters: {n_params:,}')
    return n_params


def sample(model, tokenizer):
    pass


def lr_schedule(step):
        if step <= WARMUP_STEP:
            alpha = (PEAK_LR - INIT_LR) / (WARMUP_STEP ** 2)
            lr = alpha * (step ** 2) + INIT_LR
        else:
            beta = - WARMUP_STEP - DOWN_WEIGHT * math.log(PEAK_LR - MIN_LR)
            lr = math.exp(-(step + beta) / DOWN_WEIGHT) + MIN_LR
        return lr
    

def set_description_bar(bar, epoch, step, loss, ppl, acc, val_ppl, val_acc, lr):
    description = (f'epoch: {epoch}/{EPOCHS} - step: {step} - loss: {loss:.4f} '
        + f'- ppl: {ppl:.4f} - acc: {acc:.4f} ')
    
    if val_acc is not None and val_ppl is not None:
        description += f'- val_ppl: {val_ppl:.4f} - val_acc: {val_acc:.4f} '
        
    description += f'- lr: {lr:.4f}'
    bar.set_description(description)
    
    
def save_model(model, epoch):
    file_name = f'pretrained-D{D_MODEL}-H{N_HEADS}-B{N_BLOCKS}-{epoch}.pt'
    
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), file_name)
    elif isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), file_name)
        
        
def write_tensorboard_logs(writer, global_step, loss=None, ppl=None, acc=None, val_ppl=None, val_acc=None):
    if loss is not None:
        writer.add_scalar('loss/train', loss, global_step)
        writer.add_scalar('ppl/train', ppl, global_step)
        writer.add_scalar('acc/train', acc, global_step)
    
    if val_ppl is not None:    
        writer.add_scalar('ppl/val', val_ppl, global_step)
        writer.add_scalar('acc/val', val_acc, global_step)
