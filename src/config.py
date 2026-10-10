import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "fmnist_subset.npz"
DVC_FILE = ROOT / "data" / "fmnist_subset.npz.dvc"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "fmnist-cnn-e2e"
REGISTERED_MODEL = "fmnist-cnn"

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Baseline: CNN sederhana, tanpa augmentasi & dropout
BASELINE_CFG = dict(n_filters=16, hidden=64, dropout=0.0, lr=1e-3,
                    batch_size=64, weight_decay=0.0, augment=False)

SEEDS = [0, 1, 2]
