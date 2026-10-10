"""Tahap 6: ablation faktorial 2x2 (augmentasi x dropout), masing-masing 3 seed."""
import argparse

import mlflow
import pandas as pd

from .config import RESULTS, SEEDS
from .data import load_arrays
from .train import train_and_log
from .utils import load_json, setup_mlflow

VARIANTS = {
    "full": {},
    "no_augmentation": {"augment": False},
    "no_dropout": {"dropout": 0.0},
    "no_aug_no_dropout": {"augment": False, "dropout": 0.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    setup_mlflow()
    arrays = load_arrays()
    best = load_json(RESULTS / "tuning.json")["best_params"]

    rows = []
    with mlflow.start_run(run_name="ablation"):
        mlflow.set_tag("stage", "ablation")
        mlflow.log_params({**{f"cfg_{k}": v for k, v in best.items()}, "epochs": args.epochs})
        for name, override in VARIANTS.items():
            cfg = {**best, **override}
            for seed in SEEDS:
                with mlflow.start_run(run_name=f"{name}_seed{seed}", nested=True):
                    mlflow.set_tags({"stage": "ablation_run", "variant": name})
                    res = train_and_log(cfg, seed, args.epochs, arrays)
                rows.append({"variant": name, "seed": seed, **res})

        df = pd.DataFrame(rows)
        for name, g in df.groupby("variant"):
            mlflow.log_metrics({f"{name}_test_acc_mean": float(g.test_acc.mean()),
                                f"{name}_test_acc_std": float(g.test_acc.std(ddof=1))})
        df.to_csv(RESULTS / "ablation.csv", index=False)
        mlflow.log_artifact(str(RESULTS / "ablation.csv"))

    print(df.groupby("variant")[["val_acc", "test_acc"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
