import torch
from torch import nn
import os
from config import *


class Tracker:
    def __init__(self, model, optimizer, scaler, lr_scheduler):
        self.last_k = SAVE_LAST_K_CHECKPOINTS * CHECKPOINT_STEP
        
        if isinstance(model, nn.DataParallel):
            model_state_dict = model.module.state_dict()
        elif isinstance(model, nn.Module):
            model_state_dict = model.state_dict()
        
        self.checkpoint = dict(
            model=model_state_dict,
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            lr_scheduler=lr_scheduler.state_dict()
        )
    
    def save_model(self, step, label='pretrained'):
        if not os.path.exists('./checkpoints'):
            os.makedirs('checkpoints')
            
        path = f'checkpoints/{label}-D{D_MODEL}-H{N_HEADS}-B{N_BLOCKS}-{step}.pt'
        last_kth = f'checkpoints/{label}-D{D_MODEL}-H{N_HEADS}-B{N_BLOCKS}-{step - self.last_k}.pt'
        if os.path.exists(last_kth):
            os.remove(last_kth)
        
        torch.save(self.checkpoint, path)
