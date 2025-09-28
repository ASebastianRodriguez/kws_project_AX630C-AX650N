from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split

CLASSES = ["ACOPLAR","CANCELAR","CONTINUAR","FONDO",
           "LEVANTADO","PRINCIPAL","REPETIR","SALIR"]

class KWSPTDataset(Dataset):
    def __init__(self, root: str = "tensores"):
        self.paths = sorted(Path(root).rglob("*.pt"))
        if not self.paths:
            raise RuntimeError(f"No hay .pt en {root}")
    def __len__(self): return len(self.paths)
    def __getitem__(self, i: int):
        pack = torch.load(self.paths[i], map_location="cpu")
        feat = pack["feat"].float()      # [T, F]
        label = int(pack["label"])       # 0..7
        return feat, label, feat.shape[0]

def pad_collate(batch):
    feats, labels, lengths = zip(*batch)
    F = feats[0].shape[1]
    Tm = max(lengths)
    out = feats[0].new_zeros((len(feats), Tm, F))
    for i, x in enumerate(feats):
        t = x.shape[0]
        out[i, :t] = x
    return out, torch.tensor(labels), torch.tensor(lengths)

def make_loaders(root="tensores", bs=32, val_split=0.15, test_split=0.1, seed=42):
    ds = KWSPTDataset(root)
    n = len(ds)
    n_test = int(n * test_split)
    n_val  = int(n * val_split)
    n_train = n - n_val - n_test
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=g)

    mk = lambda d, sh: DataLoader(d, batch_size=bs, shuffle=sh, collate_fn=pad_collate,
                                  num_workers=0, pin_memory=True)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)
