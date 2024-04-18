import config as C


def to_bytes(ids, size):
    ids = [i.to_bytes(size, 'big') for i in ids]
    return b''.join(ids)


def tokenize_corpus(path, destination, tokenizer):
    output_file = open(destination, 'wb')

    with open(path) as file:
        while True:
            text = file.readline()
            if text == '':
                break
                
            text = text.replace('\n', '').strip()
            if text == '' or text.startswith('='):
                continue
            
            ids = tokenizer.encode(text + '<end>').ids
            byte_sequence = to_bytes(ids, 2)
            output_file.write(byte_sequence)

    output_file.close()
        
        
def generate_sequences(file_path, n_tokens, token_size=2):
    with open(file_path, 'rb') as file:
        byte_sequence = file.read(n_tokens * token_size)
        tokens = [byte_sequence[i:i + token_size] for i in range(0, len(byte_sequence), token_size)]
        tokens = [int.from_bytes(b, 'big') for b in tokens]
        yield tokens
        
        
if __name__ == '__main__':
    from argparse import ArgumentParser
    from tokenizers import Tokenizer
    
    
    parser = ArgumentParser()
    parser.add_argument('--file-path', type=str, required=True)
    parser.add_argument('--destination', type=str, required=True)
    
    args = parser.parse_args()
    
    tokenizer = Tokenizer.from_file(C.TOKENIZER_PATH)
    tokenize_corpus(args.file_path, args.destination, tokenizer)
