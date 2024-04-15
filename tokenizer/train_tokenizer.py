import sys
sys.path.append('..')

from argparse import ArgumentParser
from tokenizers import ByteLevelBPETokenizer

from config import *


parser = ArgumentParser()
parser.add_argument('-f', '--files', nargs='+', required=True)
parser.add_argument('-d', '--destination', type=str, required=True)

args = parser.parse_args()

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(args.files, vocab_size=VOCAB_SIZE - 2)
tokenizer.add_special_tokens(['<end>', '<sum>'])

tokenizer.save(args.destination)