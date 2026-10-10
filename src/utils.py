import hashlib
import json
import random
import re
import sys
import urllib.request

import mlflow
import numpy as np
import torch

from .config import DATA_PATH, DVC_FILE, EXPERIMENT, TRACKING_URI


def setup_mlflow():
    if TRACKING_URI.startswith("http"):
        try:
            urllib.request.urlopen(TRACKING_URI.rstrip("/") + "/health", timeout=3)
        except Exception:
            sys.exit(f"[!] MLflow server tidak terjangkau di {TRACKING_URI}.\n"
                     "    Jalankan dulu di terminal lain: bash scripts/start_server.sh")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dataset_hash():
    """Hash dataset dari file .dvc (md5 DVC). Fallback: md5 file langsung."""
    if DVC_FILE.exists():
        m = re.search(r"md5:\s*([0-9a-f]{32})", DVC_FILE.read_text())
        if m:
            return m.group(1), "dvc"
    if DATA_PATH.exists():
        h = hashlib.md5()
        with open(DATA_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest(), "file_md5"
    return "unknown", "none"


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def load_json(path):
    if not path.exists():
        sys.exit(f"[!] {path} belum ada. Jalankan tahap sebelumnya dulu.")
    return json.loads(path.read_text())
