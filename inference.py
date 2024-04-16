from argparse import ArgumentParser

from tokenizers import Tokenizer
import torch

from config import DEVICE, MAXLEN, TOKENIZER_PATH
from sampler import Sampler
from modules import get_model_from_config


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default=DEVICE)
    parser.add_argument('--tokenizer', type=str, default=TOKENIZER_PATH)
    parser.add_argument('--robust', type=float, default=1.)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--seed', type=str, required=True)
    parser.add_argument('--maxlen', type=int, default=MAXLEN)

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, args.device)
    model = get_model_from_config()
    model.load_state_dict(checkpoint['model'])
    
    tokenizer = Tokenizer.from_file(args.tokenizer)

    sampler = Sampler(model, tokenizer, args.device, args.robust)

    seed = sampler.sample(args.seed, args.topk, args.maxlen)
    print(seed)
