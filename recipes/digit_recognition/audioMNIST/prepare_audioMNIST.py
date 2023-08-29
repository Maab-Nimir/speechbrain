import torch
import torchaudio
import os
from torch.utils.data import Dataset

# import hub


class AudioMNISTDataset(Dataset):
    def __init__(self, AUDIO_DIR, transform, signal_sample_rate, num_samples):
        super().__init__()
        self.AUDIO_DIR = AUDIO_DIR
        self.signal_sample_rate = signal_sample_rate
        self.num_samples = num_samples
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

        # signal = self.resample_if_necessary(signal, sr) # to make all the samples with the same sample rate
        # signal = self.mix_down_if_necessary(signal) # to make all the signals to be mono (with 1 channel)

        signal = self.cut_if_necessary(
            signal
        )  # cut if the signal has more samples than what we have defined
        signal = self.right_pad_if_necessary(signal)

        # print("signal shape before mel_spectrogram = ", signal.shape)
        signal = self.transform(signal)
        return signal, y

    def cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def right_pad_if_necessary(self, signal):
        signal_length = signal.shape[1]
        if signal_length < self.num_samples:
            pad_length = self.num_samples - signal_length
            signal = torch.nn.functional.pad(signal, (0, pad_length))
        return signal

    def resample_if_necessary(self, signal, sr):
        if sr != self.signal_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            signal = resampler(signal)
        return signal

    def mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


# class AudioMNISTDataset(Dataset):
#     def __init__(self):
#         super().__init__()
#         self.ds = hub.load("hub://activeloop/spoken_mnist")
#         self.transform = torchvision.transforms.Compose(
#             [
#                 torchvision.transforms.ToTensor(),
#             ]
#         )

#     def __len__(self):
#         return len(self.ds.spectrograms)

#     def __getitem__(self, index):
#         x = self.ds.audio[index].numpy()
#         # x = self.ds.spectrograms[index].numpy()
#         if self.transform:
#             x = self.transform(x)
#         else:
#             print("NO data transform applied ...")

#         y = torch.from_numpy(np.array(self.ds.labels[index]).astype(np.float32))
#         return x, y.squeeze(0)
