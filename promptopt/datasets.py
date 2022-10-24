import numpy as np
import torch
from torch.utils.data import DataLoader


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, batch_size=32, train_frac=0.9):
        self.batch_size = batch_size
        self.train_frac = train_frac

    def split(self):
        n = len(self)
        n_train = int(self.train_frac * n)
        n_val = n - n_train
        train_data, val_data = torch.utils.data.random_split(self, [n_train, n_val])
        make_dataloader = lambda data: DataLoader(
            data, batch_size=self.batch_size, shuffle=True, collate_fn=numpy_collate
        )
        self.train_dataloader = make_dataloader(train_data)
        self.val_dataloader = make_dataloader(val_data)


class EmbeddingDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = []

    def append(self, embedding):
        self.embeddings.append(embedding)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


class PrefDataset(Dataset):
    def __init__(self, embedding_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_dataset = embedding_dataset
        self.a_idxes = []
        self.b_idxes = []
        self.prefs = []

    def append(self, a_idx, b_idx, pref):
        n = len(self.embedding_dataset)
        assert a_idx < n and b_idx < n
        assert pref == 0 or pref == 1
        self.a_idxes.append(a_idx)
        self.b_idxes.append(b_idx)
        self.prefs.append(pref)
        assert len(self.a_idxes) == len(self.b_idxes) == len(self.prefs)

    def __len__(self):
        return len(self.prefs)

    def __getitem__(self, i):
        a_embedding = self.embedding_dataset[self.a_idxes[i]]
        b_embedding = self.embedding_dataset[self.b_idxes[i]]
        return a_embedding, b_embedding, self.prefs[i]
