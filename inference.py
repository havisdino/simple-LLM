from argparse import ArgumentParser

from tokenizers import Tokenizer

from sampler import Sampler
from modules import get_model_from_config


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tokenizer', type=str, default='tokenizer/byte-level-bpe-wikitext103.json')
    parser.add_argument('--robust', type=float, default=3.)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--seed', type=str, required=True)


    args = parser.parse_args()

    model = get_model_from_config()
    tokenizer = Tokenizer.from_file(args.tokenizer)

    sampler = Sampler(model, tokenizer, args.device, args.robust)

    seed = sampler.sample(args.seed, args.topk)
    print(seed)
