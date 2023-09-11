import torchaudio
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class AudioMNISTDataset(Dataset):
    def __init__(self, AUDIO_DIR, transform):
        super().__init__()
        self.AUDIO_DIR = AUDIO_DIR
        self.filenames = os.listdir(self.AUDIO_DIR)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        f = os.path.join(self.AUDIO_DIR, filename)
        if os.path.isfile(f):
            signal, sr = torchaudio.load(f)
            # signal --> Tensor --> (1, num_samples)
            y = int(filename[0])
        signal = self.transform(signal)  # apply mel_spectrogram transform
        return signal.squeeze(0), y


def pad_collate(batch):
    (xx, yy) = zip(*batch)

    # calculate the max length in the batch
    max_length = 0
    for x in xx:
        if x.shape[1] > max_length:
            max_length = x.shape[1]

    # right pad the batch sequences with the max length in the batch
    new_xx = []
    for x in xx:
        # print("x shape before padding ", x.shape, "max_length = ", max_length)
        if x.shape[1] < max_length:
            right_padding = max_length - x.shape[1]
            x = F.pad(input=x, pad=(0, right_padding, 0, 0), mode="constant", value=0)
        x = x.unsqueeze(0)  # 1 input channel --> [1, mel_size, batch_max_length]
        # print("x shape after padding ", x.shape)
        new_xx.append(x)
    # new_xx = tuple(new_xx)
    new_xx = torch.stack(new_xx, dim=0)

    return new_xx, torch.tensor(yy)
