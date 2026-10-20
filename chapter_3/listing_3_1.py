import torch 
from listing_3_1_class import SelfAttention_v1

torch.manual_seed(123)

d_in = 3
d_out = 2

inputs = torch.tensor([[0.43, 0.15, 0.89],  # Your
                       [0.55, 0.87, 0.66],  # Jorney
                       [0.57, 0.85, 0.64],  # starts
                       [0.22, 0.58, 0.33],  # with
                       [0.77, 0.25, 0.10],  # one
                       [0.05, 0.80, 0.55]]) # step

sa_v1 = SelfAttention_v1(d_in, d_out)

print('Context Vector:')
print(sa_v1(inputs))