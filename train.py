from argparse import ArgumentParser
from tokenizers import Tokenizer
import torch
from torch.nn import functional as F
from tqdm import tqdm
from dataset import CSVTextDataset
from evaluate import evaluate, get_accurracy, get_perplexity
from modules import Transformer
from rezero_modules import ReZeroTransformer
from utils import count_params, init_weights, save_model, set_description_bar, write_tensorboard_logs
from torch.utils.tensorboard import SummaryWriter
from config import *


def get_loss(model, input_ids, target_ids):
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))
    return loss


def fit(model, train_dl, val_dl, optimizer, lr_scheduler, scaler):
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
            
            with torch.autocast(DEVICE, torch.float16, enabled=USE_AMP):
                loss = get_loss(model, input_ids, target_ids)
                loss /= GRAD_ACCUM_STEP
            scaler.scale(loss).backward()
            
            if step % GRAD_ACCUM_STEP == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                
                loss = loss.detach().item() * GRAD_ACCUM_STEP
                with torch.autocast(DEVICE, torch.float16, enabled=USE_AMP):
                    ppl = get_perplexity(model, input_ids, target_ids)
                    acc = get_accurracy(model, input_ids, target_ids)
                
                global_step += 1
                write_tensorboard_logs(writer, global_step, loss, ppl, acc)
                lr = optimizer.param_groups[0]['lr']
                grad_step = step // GRAD_ACCUM_STEP
                set_description_bar(bar, epoch, grad_step, loss, ppl, acc, val_ppl, val_acc, lr)
                
                if grad_step % CHECKPOINT_STEP == 0:
                    bar.set_description(bar.desc + '- validating')
                    val_ppl, val_acc = evaluate(model, val_dl)
                    write_tensorboard_logs(writer, global_step, val_ppl=val_ppl, val_acc=val_acc)
                    set_description_bar(bar, epoch, grad_step, loss, ppl, acc, val_ppl, val_acc, lr)
                    save_model(model, optimizer, scaler, lr_scheduler, global_step, f'pretrained_{ARCHITECTURE}')
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
    parser.add_argument('--tokenizer', type=str, default='tokenizer/byte-level-bpe-wikitext103.json')
    parser.add_argument('--from-checkpoint', default=None)
    
    args = parser.parse_args()
    
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    settings = dict(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dff=DFF,
        n_blocks=N_BLOCKS,
        maxlen=MAXLEN,
        vocab_size=VOCAB_SIZE,
        dropout=DROPOUT
    )

    if ARCHITECTURE == 'vanilla':
        model = Transformer(**settings)
    elif ARCHITECTURE == 'rezero':
        model = ReZeroTransformer(**settings)
    else:
        raise ValueError()
    
    count_params(model)
    
    if args.from_checkpoint is not None:
        checkpoint = torch.load(args.from_checkpoint, DEVICE)
        model.load_state_dict(checkpoint['model'])
    else:
        model.apply(init_weights)
    
    if args.data_parallel:
        model = nn.DataParallel(model)
    model.to(DEVICE)
    
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_schedule 
    )
    
    if args.from_checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
    
    if args.train_ds.endswith('.csv'):
        train_ds = CSVTextDataset(args.train_ds, MAXLEN + 1, tokenizer)
        val_ds = CSVTextDataset(args.val_ds, MAXLEN + 1, tokenizer, limit=VAL_LIMIT)
    elif args.train_ds.endswith('.bds'):
        train_ds = TokenDataset(args.train_ds, MAXLEN + 1, MAXLEN // 4)
        val_ds = TokenDataset(args.val_ds, MAXLEN + 1, 0, limit=VAL_LIMIT)
    
    loader_settings = dict(
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        prefetch_factor=PREFETCH_FACTOR,
        num_workers=2,
        drop_last=True
    )
    train_dl = DataLoader(train_ds, **loader_settings)
    val_dl = DataLoader(val_ds, **loader_settings)
    
    fit(model, train_dl, val_dl, optimizer, lr_scheduler, scaler)
    