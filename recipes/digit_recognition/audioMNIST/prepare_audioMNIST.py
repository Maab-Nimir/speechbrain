import torch
from torch.utils.data import Dataset
import numpy as np
import hub


class AudioMNISTDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.ds = hub.load("hub://activeloop/spoken_mnist")

    def __len__(self):
        return len(self.ds.spectrograms)

    def __getitem__(self, index):
        x = torch.from_numpy(self.ds.spectrograms[index].numpy())
        y = torch.from_numpy(np.array(self.ds.labels[index], dtype=np.float32))
        return x, y
