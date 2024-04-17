import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate, get_perplexity
from tracker import Tracker
from utils import set_description_bar, write_tensorboard_logs


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        scaler,
        vocab_size: int,
        use_amp: bool,
        device,
        grad_accum_step: int,
        save_last_kth: int,
        checkpoint_step: int,
        epochs: int,
        start_step=0
        
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.global_step = start_step
        
        self.loss = None
        self.ppl = None
        self.val_ppl = None
        
        self.vocab_size = vocab_size
        self.use_amp = use_amp
        self.device = device
        self.grad_accum_step = grad_accum_step
        self.save_last_kth = save_last_kth
        self.checkpoint_step = checkpoint_step
        self.epochs = epochs
    
    def get_loss(self, input_ids, target_ids):
        logits = self.model(input_ids)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), target_ids.view(-1))
        return loss
    
    def train_step(self, input_ids, target_ids):
        self.model.train()
        with torch.autocast(self.device, torch.float16, enabled=self.use_amp):
            self.loss = self.get_loss(input_ids, target_ids)
            self.loss /= self.grad_accum_step
        self.scaler.scale(self.loss).backward()
        
    def accumulate_gradient(self):
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        
        self.global_step += 1
        
    def validate(self, input_ids=None, target_ids=None, val_dl=None):
        if input_ids is not None and target_ids is not None:
            with torch.autocast(self.device, torch.float16, enabled=self.use_amp):
                self.ppl = get_perplexity(self.model, input_ids, target_ids).item()
            
        if val_dl is not None:
            self.val_ppl = evaluate(self.model, val_dl, self.device, self.use_amp)
            
    def batch_loss(self):
        return self.loss.detach().item() * self.grad_accum_step

    def fit(self, train_dl, val_dl):
        writer = SummaryWriter('logs')
        tracker = Tracker(self.save_last_kth, self.checkpoint_step)
        
        print(f'Accumulating gradients after {self.grad_accum_step} steps')
        
        for epoch in range(1, 1 + self.epochs):
            self.optimizer.zero_grad()
            
            for step, (input_ids, target_ids) in enumerate(bar := tqdm(train_dl), 1):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                self.train_step(input_ids, target_ids)
                
                if step % self.grad_accum_step == 0:
                    self.accumulate_gradient()
                    self.validate(input_ids, target_ids)
                    
                    lr = self.optimizer.param_groups[0]['lr']
                    write_tensorboard_logs(writer, self.global_step, self.batch_loss(), self.ppl, lr)
                    
                    grad_step = step // self.grad_accum_step
                    set_description_bar(
                        bar, epoch, grad_step,
                        loss=self.batch_loss(),
                        ppl=self.ppl,
                        val_ppl=self.val_ppl,
                        lr=f'{lr:.2e}'
                    )
                    
                    if grad_step % self.checkpoint_step == 0:
                        bar.set_description(bar.desc + 'validating...')
                        
                        self.validate(val_dl=val_dl)
                        
                        write_tensorboard_logs(writer, self.global_step, val_ppl=self.val_ppl, lr=lr)
                        set_description_bar(
                            bar, epoch, grad_step,
                            loss=self.batch_loss(),
                            ppl=self.ppl,
                            val_ppl=self.val_ppl,
                            lr=f'{lr:.2e}'
                        )
                        tracker.save_model(
                            self.model, self.optimizer,
                            self.scaler, self.lr_scheduler,
                            self.global_step
                        )
        writer.close()
