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

### 

token_embeddings = embedding_layer(inputs)
print(f"\nToken embeddings shape:\n{token_embeddings.shape}")


####

#To remember : Input_emb = Position_emb + Token_emb

#Token embeddings are the representations of the input tokens in a continuous vector space. 
#Each token ID is mapped to a corresponding embedding vector, which captures semantic information about the token. 
#The shape of the token embeddings tensor is determined by the batch size, sequence length (max_length), 
#and the output dimension of the embedding layer.

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(f"\nPosition embeddings shape:\n{pos_embeddings.shape}")

input_embeddings = token_embeddings + pos_embeddings
print(f"\nInput embeddings shape:\n{input_embeddings.shape}")

print(f"\nPosition embeddings:\n{pos_embeddings}")
print(f"\nToken embeddings:\n{token_embeddings}")
print(f"\nInput embeddings:\n{input_embeddings}")