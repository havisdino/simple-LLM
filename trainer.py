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
        start_step=0
        
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.global_step = start_step
        self.substep = 0
        
        self.loss = None
        self.ppl = None
        self.val_ppl = None
        
        self.vocab_size = vocab_size
        self.use_amp = use_amp
        self.device = device
        self.grad_accum_step = grad_accum_step
        self.save_last_kth = save_last_kth
        self.checkpoint_step = checkpoint_step
    
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
        
        self.substep += 1
        
    def accumulate_gradient(self):
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        
        self.global_step += 1
        
    def validate(self, input_ids=None, target_ids=None, valloader=None):
        if input_ids is not None and target_ids is not None:
            with torch.autocast(self.device, torch.float16, enabled=self.use_amp):
                self.ppl = get_perplexity(self.model, input_ids, target_ids).item()
            
        if valloader is not None:
            self.val_ppl = evaluate(self.model, valloader, self.device, self.use_amp)
            
    def batch_loss(self):
        return self.loss.detach().item() * self.grad_accum_step

    def fit(self, trainloader, valloader, n_steps):
        n_steps += self.global_step
        
        writer = SummaryWriter('logs')
        tracker = Tracker(self.save_last_kth, self.checkpoint_step)
        
        print(f'Accumulating gradients after {self.grad_accum_step} substeps')
        
        data_iter = iter(trainloader)
        self.optimizer.zero_grad()
        
        bar = tqdm()
        
        while self.global_step < n_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(trainloader)
                continue        
            batch = [x.to(self.device) for x in batch]
            input_ids, target_ids = batch

            self.train_step(input_ids, target_ids)
            
            if self.substep % self.grad_accum_step == 0:
                self.accumulate_gradient()
                self.validate(input_ids, target_ids)
                
                lr = self.optimizer.param_groups[0]['lr']
                write_tensorboard_logs(
                    writer=writer,
                    global_step=self.global_step,
                    loss=self.batch_loss(), 
                    ppl=self.ppl,
                    lr=lr
                )
                set_description_bar(
                    bar, self.global_step, n_steps,
                    loss=self.batch_loss(),
                    ppl=self.ppl,
                    val_ppl=self.val_ppl,
                    lr=f'{lr:.2e}'
                )
                
                if self.global_step % self.checkpoint_step == 0:
                    bar.set_description(bar.desc + 'validating...')
                    
                    self.validate(valloader=valloader)
                    
                    write_tensorboard_logs(
                        writer=writer,
                        global_step=self.global_step,
                        val_ppl=self.val_ppl,
                        lr=lr
                    )
                    set_description_bar(
                        bar, self.global_step, n_steps,
                        loss=self.batch_loss(),
                        ppl=self.ppl,
                        val_ppl=self.val_ppl,
                        lr=f'{lr:.2e}'
                    )
                    tracker.save(
                        self.model, self.optimizer,
                        self.scaler, self.lr_scheduler,
                        self.global_step
                    )
            bar.update()
        writer.close()
        bar.close()
        