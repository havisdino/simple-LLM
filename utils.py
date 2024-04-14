import os
import torch
from torch import nn
from config import *
import math


def get_causal_mask(maxlen):
    causal_mask = torch.ones(maxlen, maxlen).tril() == 0
    causal_mask = torch.where(causal_mask, -float('inf'), 0)
    return causal_mask


def init_weights(m):
    for p in m.parameters():
        nn.init.normal_(p, std=WEIGHT_STD)


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
    description = f'epoch: {epoch}/{EPOCHS} - step: {step} - loss: {loss:.4f} '
    
    if ppl is not None and acc is not None:
        description += f'- ppl: {ppl:.4f} - acc: {acc:.4f} '
    
    if val_acc is not None and val_ppl is not None:
        description += f'- val_ppl: {val_ppl:.4f} - val_acc: {val_acc:.4f} '
        
    description += f'- lr: {lr:.2e}'
    bar.set_description(description)
    
    
def save_model(model, optimizer, scaler, lr_scheduler, step, label='pretrained'):
    if not os.path.exists('./checkpoints'):
        os.makedirs('checkpoints')
        
    path = f'checkpoints/{label}-D{D_MODEL}-H{N_HEADS}-B{N_BLOCKS}-{step}.pt'
    
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    elif isinstance(model, nn.Module):
        model_state_dict = model.state_dict()
    
    checkpoint = dict(
        model=model_state_dict,
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
        lr_scheduler=lr_scheduler.state_dict()
    )
    
    torch.save(checkpoint, path)
        
        
def write_tensorboard_logs(writer, global_step, loss=None, ppl=None, acc=None, val_ppl=None, val_acc=None, lr=None):
    if loss is not None:
        writer.add_scalar('loss/train', loss, global_step)
        writer.add_scalar('ppl/train', ppl, global_step)
        writer.add_scalar('acc/train', acc, global_step)
    
    if val_ppl is not None:    
        writer.add_scalar('ppl/val', val_ppl, global_step)
        writer.add_scalar('acc/val', val_acc, global_step)

