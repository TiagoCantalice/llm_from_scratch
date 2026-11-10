import torch
from listing_3_4 import MultiHeadAttentionWrapper

torch.manual_seed(123)

inputs = torch.tensor([[0.43, 0.15, 0.89], # Your
                       [0.55, 0.87, 0.66], # Jorney
                       [0.57, 0.85, 0.64], # starts
                       [0.22, 0.58, 0.33], # with
                       [0.77, 0.25, 0.10], # one
                       [0.05, 0.80, 0.55]])# step

batch = torch.stack((inputs, inputs), dim=0)

context_length = batch.shape[1] #this is the number of tokens in the input sequence,
#which is 6 in this case. It is used to define the size of the attention mask in 
# the CausalAttention class.

d_in, d_out = 3, 2

mha = MultiHeadAttentionWrapper(d_in=d_in, 
                                d_out=d_out, 
                                context_length=context_length, 
                                dropout=0.0, 
                                num_heads=2)


context_vectors = mha(batch)
print('Context Vectors:', context_vectors)
print('Context Vectors shape:', context_vectors.shape)