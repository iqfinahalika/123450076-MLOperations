"""Tahap 3: baseline CNN sederhana (log params, metrik per epoch, model)."""
import argparse

import mlflow

from .config import BASELINE_CFG, RESULTS
from .data import load_arrays
from .train import train_and_log
from .utils import save_json, setup_mlflow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    setup_mlflow()
    arrays = load_arrays()
    with mlflow.start_run(run_name="baseline_cnn"):
        mlflow.set_tag("stage", "baseline")
        res = train_and_log(BASELINE_CFG, seed=0, epochs=args.epochs, arrays=arrays, log_model=True)
    save_json(RESULTS / "baseline.json", {"cfg": BASELINE_CFG, "epochs": args.epochs, **res})
    print(f"Baseline  val_acc={res['val_acc']:.4f}  test_acc={res['test_acc']:.4f}")


if __name__ == "__main__":
    main()
