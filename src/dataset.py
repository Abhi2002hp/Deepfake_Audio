import os
import torch
import librosa
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        self.labels = []

        for label, folder in enumerate(["real", "fake"]):
            path = os.path.join(root_dir, folder)

            for file in os.listdir(path):
                if file.endswith(".wav"):
                    self.files.append(os.path.join(path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.files[idx], sr=16000)

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        mel = librosa.power_to_db(mel)

        mel = torch.tensor(mel).float().unsqueeze(0)
        label = torch.tensor(self.labels[idx])

        return mel, label