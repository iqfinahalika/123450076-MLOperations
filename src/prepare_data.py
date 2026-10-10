"""Unduh Fashion-MNIST, ambil subset kecil (deterministik), simpan ke data/fmnist_subset.npz.
Setelah itu versikan dengan:  dvc add data/fmnist_subset.npz
"""
import numpy as np
from torchvision import datasets

from .config import DATA_PATH, ROOT

N_TRAIN, N_VAL, N_TEST = 10_000, 2_000, 2_000


def main():
    raw = ROOT / "data" / "raw"
    tr = datasets.FashionMNIST(raw, train=True, download=True)
    te = datasets.FashionMNIST(raw, train=False, download=True)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(tr))[: N_TRAIN + N_VAL]
    tidx = rng.permutation(len(te))[:N_TEST]
    x, y = tr.data.numpy()[idx], tr.targets.numpy()[idx]

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_PATH,
        x_train=x[:N_TRAIN], y_train=y[:N_TRAIN],
        x_val=x[N_TRAIN:], y_val=y[N_TRAIN:],
        x_test=te.data.numpy()[tidx], y_test=te.targets.numpy()[tidx],
    )
    print(f"Tersimpan: {DATA_PATH}  (train={N_TRAIN}, val={N_VAL}, test={N_TEST})")


if __name__ == "__main__":
    main()
