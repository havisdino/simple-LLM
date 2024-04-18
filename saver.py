import torch
from torch import nn
import os
import config as C
from modules import get_model_config


class Saver:
    def __init__(self, last_k, checkpoint_interval):
        self.last_k = last_k * checkpoint_interval
        
    def _build_checkpoint(self, model, optimizer, scaler, lr_scheduler):
        if isinstance(model, nn.DataParallel):
            model_state_dict = model.module.state_dict()
        elif isinstance(model, nn.Module):
            model_state_dict = model.state_dict()
        
        self.last_checkpoint = dict(
            model=model_state_dict,
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            lr_scheduler=lr_scheduler.state_dict(),
            settings=get_model_config()
        )
    
    def save(self, model, optimizer, scaler, lr_scheduler, step, label='pretrained'):
        if not os.path.exists('./checkpoints'):
            os.makedirs('checkpoints')
            
        path = (f'checkpoints/{label}-{C.ARCHITECTURE}-D{C.D_MODEL}'
                + f'-H{C.N_HEADS}-B{C.N_BLOCKS}-{step}.pt')
        
        last_kth = (f'checkpoints/{label}-{C.ARCHITECTURE}-D{C.D_MODEL}'
                    + f'-H{C.N_HEADS}-B{C.N_BLOCKS}-{step - self.last_k}.pt')
        
        if os.path.exists(last_kth):
            os.remove(last_kth)
            
        self._build_checkpoint(model, optimizer, scaler, lr_scheduler)
        
        torch.save(self.last_checkpoint, path)
