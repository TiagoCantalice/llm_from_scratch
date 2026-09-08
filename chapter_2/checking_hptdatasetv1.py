from gptdatasetv1 import GPTDatasetV1
from torch.utils.data import DataLoader
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

dataset = GPTDatasetV1("Hello, world! This is a test.", tokenizer, max_length=3, stride=1)

print(f"Number of samples: {len(dataset)}")

input_ids, target_ids = dataset[0]
print(f"First sample input:  {input_ids}")
print(f"First sample target: {target_ids}")

dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
first_batch = next(iter(dataloader))
print(f"First batch inputs:\n{first_batch[0]}")
print(f"First batch targets:\n{first_batch[1]}")