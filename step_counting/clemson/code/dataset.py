import torch
from torch.utils.data import Dataset
import numpy as np


class StepDataset(Dataset):
    """PyTorch dataset for image-like windows, dense labels, and subject ids."""

    def __init__(self, X, y, chunk_counts, subj_ids):
        """
        X:            (N, H, T, C)
        y:            (N, T)
        chunk_counts: (N, n_chunks)
        subj_ids:     (N,) integer subject ids
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.chunk_counts = chunk_counts.astype(np.float32)
        self.subj_ids = np.asarray(subj_ids, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if x.ndim != 3:
            raise ValueError(f"Expected x.ndim == 3, got {x.ndim}")

        # Convert NumPy image layout to PyTorch channel-first layout.
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        c = torch.tensor(self.chunk_counts[idx], dtype=torch.float32)
        s = torch.tensor(self.subj_ids[idx], dtype=torch.long)

        return x, y, c, s
