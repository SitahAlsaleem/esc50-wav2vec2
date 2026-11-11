"""ESC-50 dataset with the official fold split.

Split used in this project (no data leakage, folds never mix):
    train: folds 1-3   (1200 clips)
    val:   fold 4      (400 clips)
    test:  fold 5      (400 clips)

Expected folder layout (from `git clone https://github.com/karolpiczak/ESC-50`):
    <root>/meta/esc50.csv
    <root>/audio/*.wav
"""
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from utils.seed import seed_worker

SAMPLE_RATE = 16_000          # Wav2Vec 2.0 expects 16 kHz
CLIP_SECONDS = 5
NUM_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
NUM_CLASSES = 50

TRAIN_FOLDS = (1, 2, 3)
VAL_FOLDS = (4,)
TEST_FOLDS = (5,)


def load_waveform(path: Path) -> torch.Tensor:
    """Load a wav file -> mono float tensor of exactly NUM_SAMPLES at 16 kHz."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
    wav = torch.from_numpy(wav).mean(dim=1)                         # mono, (T,)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.numel() < NUM_SAMPLES:                                   # pad
        wav = torch.nn.functional.pad(wav, (0, NUM_SAMPLES - wav.numel()))
    return wav[:NUM_SAMPLES]                                        # trim


class ESC50Dataset(Dataset):
    """
    feature="wave": returns waveform of shape (80000,)         -> for Wav2Vec 2.0
    feature="mel" : returns log-Mel of shape (1, n_mels, 251)   -> for the 2D-CNN
    augment=True  : random time shift + light Gaussian noise (use for train only)
    """

    def __init__(self, root, folds, feature: str = "wave",
                 augment: bool = False, n_mels: int = 64):
        assert feature in ("wave", "mel")
        self.root = Path(root)
        self.feature = feature
        self.augment = augment

        meta = pd.read_csv(self.root / "meta" / "esc50.csv")
        self.meta = meta[meta["fold"].isin(folds)].reset_index(drop=True)
        self.folds = tuple(folds)
        self.filenames = self.meta["filename"].tolist()
        self.labels = torch.tensor(self.meta["target"].values, dtype=torch.long)

        # Load + resample once, keep in RAM (1200 clips ~ 400 MB).
        self.waves = torch.stack(
            [load_waveform(self.root / "audio" / f) for f in self.filenames]
        )

        if feature == "mel":
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=320, n_mels=n_mels
            )
            self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.labels)

    def _augment(self, wav: torch.Tensor) -> torch.Tensor:
        shift = int(torch.randint(-SAMPLE_RATE // 2, SAMPLE_RATE // 2 + 1, (1,)))
        wav = torch.roll(wav, shifts=shift)                         # time shift +-0.5 s
        noise_level = 0.01 * torch.rand(1) * wav.std()
        return wav + noise_level * torch.randn_like(wav)            # light noise

    def __getitem__(self, idx):
        wav = self.waves[idx]
        if self.augment:
            wav = self._augment(wav)
        if self.feature == "mel":
            x = self.to_db(self.mel(wav)).unsqueeze(0)              # (1, n_mels, T)
        else:
            x = wav
        return x, self.labels[idx]


def make_datasets(root, feature: str = "wave", n_mels: int = 64):
    """Return (train, val, test) datasets and verify the folds do not overlap."""
    train = ESC50Dataset(root, TRAIN_FOLDS, feature, augment=True, n_mels=n_mels)
    val = ESC50Dataset(root, VAL_FOLDS, feature, augment=False, n_mels=n_mels)
    test = ESC50Dataset(root, TEST_FOLDS, feature, augment=False, n_mels=n_mels)

    # Leakage guard: no clip may appear in two splits.
    s_tr, s_va, s_te = set(train.filenames), set(val.filenames), set(test.filenames)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), \
        "Data leakage: a clip appears in more than one split!"
    return train, val, test


def make_loaders(train, val, test, batch_size: int = 32, seed: int = 42,
                 num_workers: int = 2):
    g = torch.Generator()
    g.manual_seed(seed)
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              worker_init_fn=seed_worker, generator=g)
    return (DataLoader(train, shuffle=True, **kw),
            DataLoader(val, shuffle=False, **kw),
            DataLoader(test, shuffle=False, **kw))
