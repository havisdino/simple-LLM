from argparse import ArgumentParser
from tokenizers import ByteLevelBPETokenizer


parser = ArgumentParser()
parser.add_argument('-f', '--files', nargs='+', required=True)
parser.add_argument('-s', '--vocab-size', type=int, required=True)
parser.add_argument('-d', '--destination', type=str, required=True)

args = parser.parse_args()

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(args.files, vocab_size=args.vocab_size - 2)
tokenizer.add_special_tokens(['<end>', '<sum>'])

tokenizer.save(args.destination)