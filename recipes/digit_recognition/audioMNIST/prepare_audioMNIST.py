import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import hub


class AudioMNISTDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.ds = hub.load("hub://activeloop/spoken_mnist")
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.ds.spectrograms)

    def __getitem__(self, index):
        x = self.ds.spectrograms[index].numpy()
        if self.transform:
            x = self.transform(x)
        else:
            print("NO data transform applied ...")

        y = torch.from_numpy(np.array(self.ds.labels[index]).astype(np.float32))
        return x, y.squeeze(0)
