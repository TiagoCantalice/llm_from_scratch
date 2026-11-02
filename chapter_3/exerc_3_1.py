from listing_3_2_class import SelfAttention_v2
import torch

torch.manual_seed(789)

d_in = 3
d_out = 2

inputs = torch.tensor([[0.43, 0.15, 0.89],  # Your
                       [0.55, 0.87, 0.66],  # Jorney
                       [0.57, 0.85, 0.64],  # starts
                       [0.22, 0.58, 0.33],  # with
                       [0.77, 0.25, 0.10],  # one
                       [0.05, 0.80, 0.55]]) # step

sa_v2 = SelfAttention_v2(d_in, d_out)



print(sa_v2(inputs))