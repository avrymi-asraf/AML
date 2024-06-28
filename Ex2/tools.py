from torch.utils import data
import torch


class MyDataset(data.Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
