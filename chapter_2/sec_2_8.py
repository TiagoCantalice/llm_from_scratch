# Example of input tokens

import torch
import tiktoken
from gptdatasetv1 import GPTDatasetV1
from torch.utils.data import DataLoader

inputs_ids = torch.tensor([2, 3, 4, 5])

vocab_size = 50257
output_dim = 256

torch.manual_seed(123)

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)

max_length = 4

#Creating a function to create a dataloader

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=28, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader

with open(r"C:\Users\canta\Google Drive\llm_from_scratch\llm_from_scratch\chapter_2\the-verdict.txt", "r", encoding='utf-8') as f:
    text = f.read()

dataloader = create_dataloader_v1(txt = text, 
                                  batch_size = 8, 
                                  max_length = max_length,
                                  stride = max_length, 
                                  shuffle = False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)  
print(f"Token IDs:\n{inputs}")
print(f"\nInputs shape:\n{inputs.shape}")
