from argparse import ArgumentParser
from tokenizers import Tokenizer
import torch
from dataset import CSVTextDataset
from modules import get_model_from_config
from config import *
from trainer import Trainer
from utils import count_params, get_step_from_name
from utils import lr_schedule
from dataset import TokenDataset, collate_fn
from torch.utils.data import DataLoader
from torch import nn


parser = ArgumentParser()
parser.add_argument('--train-ds', type=str, required=True)
parser.add_argument('--val-ds', type=str, required=True)
parser.add_argument('--data-parallel', type=bool, default=True)
parser.add_argument('--from-checkpoint', default=None)

args = parser.parse_args()

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

model = get_model_from_config() 
count_params(model)

if args.from_checkpoint is not None:
    checkpoint = torch.load(args.from_checkpoint, DEVICE)
    model.load_state_dict(checkpoint['model'])
    print('Checkpoint loaded')
    start_step = get_step_from_name(args.from_checkpoint)
else:
    start_step = 0

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
    train_ds = CSVTextDataset(args.train_ds, MAXLEN + 1, tokenizer, limit=TRAIN_LIMIT)
    val_ds = CSVTextDataset(args.val_ds, MAXLEN + 1, tokenizer, limit=VAL_LIMIT)
elif args.train_ds.endswith('.bds'):
    train_ds = TokenDataset(args.train_ds, MAXLEN + 1, MAXLEN // 4, limit=TRAIN_LIMIT)
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

trainer = Trainer(
    model, optimizer, lr_scheduler, scaler,
    VOCAB_SIZE, USE_AMP, DEVICE, GRAD_ACCUM_STEP,
    SAVE_LAST_K_CHECKPOINTS,
    CHECKPOINT_STEP, EPOCHS, start_step
)

trainer.fit(train_dl, val_dl)
