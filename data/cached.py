"""Dataset over cached Wav2Vec features (see cache_features.py)."""
import torch
from torch.utils.data import Dataset


class CachedFeatures(Dataset):
    def __init__(self, path):
        blob = torch.load(path)
        self.feats = blob["feats"]            # (N, T, 768) fp16
        self.labels = blob["labels"]
        self.filenames = blob["filenames"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.feats[idx].float(), self.labels[idx]
