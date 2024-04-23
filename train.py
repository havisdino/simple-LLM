from argparse import ArgumentParser
from tokenizers import Tokenizer
import torch
from dataset import CSVTextDataset
from modules import get_model_from_config
import config as C
from trainer import Trainer
from utils import count_params, get_step_from_name, modify_config
from utils import lr_schedule
from dataset import TokenDataset, collate_fn
from torch.utils.data import DataLoader
from torch import nn


parser = ArgumentParser()
parser.add_argument('--traindata', type=str, required=True)
parser.add_argument('--valdata', type=str, default=None)
parser.add_argument('--data-parallel', type=bool, default=False)
parser.add_argument('--from-checkpoint', default=None)

args = parser.parse_args()

tokenizer = Tokenizer.from_file(C.TOKENIZER_PATH)

if args.from_checkpoint is not None:
    checkpoint = torch.load(args.from_checkpoint, C.DEVICE)
    settings = checkpoint['settings']
    modify_config(C, **settings)
    
    model = get_model_from_config(settings) 
    model.load_state_dict(checkpoint['model'])
    print('Checkpoint loaded, default settings might be ignored')
    
    start_step = get_step_from_name(args.from_checkpoint)
else:
    start_step = 0

count_params(model)

if args.data_parallel:
    model = nn.DataParallel(model)
model.to(C.DEVICE)

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

if args.traindata.endswith('.csv'):
    traindata = CSVTextDataset(args.traindata, C.MAXLEN + 1, tokenizer, limit=C.TRAIN_LIMIT)
    if args.valdata is not None:
        valdata = CSVTextDataset(args.valdata, C.MAXLEN + 1, tokenizer, limit=C.VAL_LIMIT)
elif args.traindata.endswith('.bds'):
    traindata = TokenDataset(args.traindata, C.MAXLEN + 1, C.MAXLEN // 4, limit=C.TRAIN_LIMIT)
    if args.valdata is not None:
        valdata = TokenDataset(args.valdata, C.MAXLEN + 1, 0, limit=C.VAL_LIMIT)

loader_settings = dict(
    batch_size=C.BATCH_SIZE,
    collate_fn=collate_fn,
    prefetch_factor=C.PREFETCH_FACTOR,
    num_workers=2,
    drop_last=True
)
trainloader = DataLoader(traindata, **loader_settings)
if args.valdata is not None:
    valloader = DataLoader(valdata, **loader_settings)
else:
    valloader = None

trainer = Trainer(
    model, optimizer, lr_scheduler, scaler,
    C.VOCAB_SIZE, C.USE_AMP, C.DEVICE,
    C.GRAD_ACCUM_STEP, C.SAVE_LAST_K_CHECKPOINTS,
    C.CHECKPOINT_STEP, start_step
)

trainer.fit(trainloader, valloader, C.N_STEPS)
