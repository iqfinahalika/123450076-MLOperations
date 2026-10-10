"""Tahap 5: konfigurasi terbaik x 3 seed -> mean ± std."""
import argparse

import mlflow
import pandas as pd

from .config import RESULTS, SEEDS
from .data import load_arrays
from .train import train_and_log
from .utils import load_json, save_json, setup_mlflow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    setup_mlflow()
    arrays = load_arrays()
    cfg = load_json(RESULTS / "tuning.json")["best_params"]

    rows = []
    with mlflow.start_run(run_name="best_config_3seeds"):
        mlflow.set_tag("stage", "seeds")
        mlflow.log_params({**{f"cfg_{k}": v for k, v in cfg.items()}, "epochs": args.epochs})
        for seed in SEEDS:
            with mlflow.start_run(run_name=f"seed_{seed}", nested=True):
                mlflow.set_tag("stage", "seed_run")
                res = train_and_log(cfg, seed, args.epochs, arrays, log_model=True)
            rows.append({"seed": seed, **res})

        df = pd.DataFrame(rows)
        summary = {}
        for m in ["val_acc", "test_acc"]:
            summary[f"{m}_mean"] = float(df[m].mean())
            summary[f"{m}_std"] = float(df[m].std(ddof=1))
        mlflow.log_metrics(summary)
        df.to_csv(RESULTS / "seeds.csv", index=False)
        save_json(RESULTS / "seeds_summary.json", {"cfg": cfg, "epochs": args.epochs, **summary})
        mlflow.log_artifact(str(RESULTS / "seeds.csv"))

    print(f"test_acc = {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}  "
          f"(val_acc = {summary['val_acc_mean']:.4f} ± {summary['val_acc_std']:.4f})")


if __name__ == "__main__":
    main()
