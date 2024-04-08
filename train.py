from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from tqdm import tqdm
from evaluate import evaluate, get_accurracy, get_perplexity
from modules import Transformer
from utils import count_params, init_weights, save_model, set_description_bar, write_tensorboard_logs
from torch.utils.tensorboard import SummaryWriter
from config import *


def get_loss(model, input_ids, target_ids):
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))
    return loss


def fit(model, train_dl, val_dl, optimizer, lr_scheduler):
    writer = SummaryWriter('logs')
    global_step = 1
    ppl, acc = None, None
    val_ppl, val_acc = None, None
    
    print(f'Accumulate gradients after {GRAD_ACCUM_STEP} steps')
    
    for epoch in range(1, 1 + EPOCHS):
        optimizer.zero_grad()
        
        for step, (input_ids, target_ids) in enumerate(bar := tqdm(train_dl), 1):
            model.train()
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            L = input_ids.size(1)
            
            loss = get_loss(model, input_ids, target_ids)
            loss /= GRAD_ACCUM_STEP
            loss.backward()
            
            if step % GRAD_ACCUM_STEP == 0:
                global_step += 1
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                
                loss = loss.detach().item() * GRAD_ACCUM_STEP
                ppl = get_perplexity(model, input_ids, target_ids)
                acc = get_accurracy(model, input_ids, target_ids)
                
                write_tensorboard_logs(writer, global_step, loss, ppl, acc)
            lr = optimizer.param_groups[0]['lr']
            set_description_bar(bar, epoch, step, loss, ppl, acc, val_ppl, val_acc, lr)
        
        if epoch % CHECKPOINT_EPOCH == 0:
            save_model(model, epoch)
            
        val_ppl, val_acc = evaluate(model, val_dl)
        write_tensorboard_logs(writer, global_step, val_ppl=val_ppl, val_acc=val_acc)
    writer.close()
                    

if __name__ == '__main__':
    from utils import lr_schedule
    from dataset import TokenDataset, collate_fn
    from torch.utils.data import DataLoader
    from torch import nn
    
    
    parser = ArgumentParser()
    parser.add_argument('--train-ds', type=str, required=True)
    parser.add_argument('--val-ds', type=str, required=True)
    parser.add_argument('--data-parallel', type=bool, default=True)
    
    args = parser.parse_args()

    model = Transformer(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dff=DFF,
        n_blocks=N_BLOCKS,
        maxlen=MAXLEN,
        vocab_size=VOCAB_SIZE,
        dropout=DROPOUT
    )
    model.apply(init_weights)
    count_params(model)
    
    if args.data_parallel:
        model = nn.DataParallel(model)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_schedule 
    )
    
    train_ds = TokenDataset(args.train_ds, MAXLEN + 1, MAXLEN // 4)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, prefetch_factor=PREFETCH_FACTOR, num_workers=2)
    
    val_ds = TokenDataset(args.val_ds, MAXLEN + 1, 0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, prefetch_factor=PREFETCH_FACTOR, num_workers=2)
    
    fit(model, train_dl, val_dl, optimizer, lr_scheduler)
    