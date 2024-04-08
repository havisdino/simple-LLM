import torch
from torch.utils.data import IterableDataset
from torchtext import transforms

from config import END_TOKEN_ID


class TokenDataset(IterableDataset):
    def __init__(self, file_path, n_tokens, overlapped_tokens, token_size=2, limit=None):
        super().__init__()
        self.file_path = file_path
        self.token_size = token_size
        self.n_tokens = n_tokens
        self.limit = limit
        self.overlapped_bytes = overlapped_tokens * token_size
        assert overlapped_tokens < n_tokens

    def generate_sequences(self):
        count = 0
        with open(self.file_path, 'rb') as file:
            while True:
                byte_sequence = file.read(self.n_tokens * self.token_size)
                file.seek(file.tell() - self.overlapped_bytes)
                if not byte_sequence:
                    return
                tokens = [byte_sequence[i:i + 2] for i in range(0, len(byte_sequence), self.token_size)]
                tokens = [int.from_bytes(b, 'big') for b in tokens]
                yield tokens
                count += 1
                if self.limit and count == self.limit:
                    return

    def __iter__(self):
        return self.generate_sequences()


def collate_fn(input_ids):
    target_ids = [item[1:] for item in input_ids]
    input_ids = [item[:-1] for item in input_ids]
    input_ids = transforms.F.to_tensor(
        input=input_ids,
        padding_value=END_TOKEN_ID,
        dtype=torch.int32
    )
    target_ids = transforms.F.to_tensor(
        input=target_ids,
        padding_value=END_TOKEN_ID,
        dtype=torch.long
    )
    return input_ids, target_ids