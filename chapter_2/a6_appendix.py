import torch
from torch.utils.data import Dataset

X_train = torch.tensor([[-1.2, 3.1],
                       [-0.9, 2.9],
                       [-0.5, 2.6],
                       [2.3, -1.1],
                       [2.7, -1.5]])

y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([[-0.8, 2.8],
                      [2.6, -1.6]])

y_test = torch.tensor([0, 1])

# Defining a custom dataset class

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        one_x = self.features[idx]
        one_y = self.labels[idx]
        return one_x, one_y

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

print(len(train_ds))  # Output: 5

# Instantiating a DataLoader for the training dataset

from torch.utils.data import DataLoader

torch.manual_seed(123)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)

test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx}:")
    print(f"Features: {x}")
    print(f"Labels: {y}")

# A trainer loader that drops the last batch

train_loader = DataLoader(train_ds, 
                          batch_size=2, 
                          shuffle=True, 
                          num_workers=0, 
                          drop_last=True)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx}:")
    print(f"Features: {x}")
    print(f"Labels: {y}")  