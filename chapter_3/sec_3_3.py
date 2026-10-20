import torch 

inputs = torch.tensor([[0.43, 0.15, 0.89], # Your
                       [0.55, 0.87, 0.66], # Jorney
                       [0.57, 0.85, 0.64], # starts
                       [0.22, 0.58, 0.33], # with
                       [0.77, 0.25, 0.10], # one
                       [0.05, 0.80, 0.55]])# step

n = 4

query = inputs[1]
print('query:', query)

attn_scores_2 = torch.empty(inputs.shape[0])
print(f'attn_scores {n}:', attn_scores_2)

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
    print(f"attn_scores_{n}[{i}]: {attn_scores_2[i]}")

print(attn_scores_2)
