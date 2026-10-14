# Example of input tokens

import torch

from torch.utils.data import Dataset, DataLoader

inputs_ids = torch.tensor([2, 3, 4, 5])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)

print(embedding_layer(torch.tensor([3])))

print(embedding_layer(inputs_ids))

