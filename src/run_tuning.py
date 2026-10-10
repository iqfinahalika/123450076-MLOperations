"""Tahap 4: hyperparameter tuning Optuna (>=20 trial), tiap trial = nested run MLflow."""
import argparse

import mlflow
import optuna

from .config import RESULTS
from .data import load_arrays
from .train import train_and_log
from .utils import save_json, setup_mlflow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    setup_mlflow()
    arrays = load_arrays()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with mlflow.start_run(run_name="optuna_tuning"):
        mlflow.set_tags({"stage": "tuning"})
        mlflow.log_params({"n_trials": args.n_trials, "tuning_epochs": args.epochs,
                           "sampler": "TPE", "objective": "val_acc (seed=0)"})

        def objective(trial):
            cfg = dict(
                n_filters=trial.suggest_categorical("n_filters", [8, 16, 32, 64]),
                hidden=trial.suggest_categorical("hidden", [64, 128, 256]),
                dropout=trial.suggest_float("dropout", 0.1, 0.6),
                lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                augment=True,  # tetap aktif; kontribusinya diuji di ablation
            )
            with mlflow.start_run(run_name=f"trial_{trial.number:02d}", nested=True):
                mlflow.set_tags({"stage": "tuning_trial", "optuna_trial": str(trial.number)})
                res = train_and_log(cfg, seed=0, epochs=args.epochs, arrays=arrays)
            return res["val_acc"]

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=args.n_trials)

        best_cfg = {**study.best_params, "augment": True}
        mlflow.log_params({f"best_{k}": v for k, v in best_cfg.items()})
        mlflow.log_metric("best_val_acc", study.best_value)

        save_json(RESULTS / "tuning.json", {"best_params": best_cfg, "best_val_acc": study.best_value,
                                            "n_trials": len(study.trials), "epochs": args.epochs})
        study.trials_dataframe().to_csv(RESULTS / "tuning_trials.csv", index=False)
        mlflow.log_artifact(str(RESULTS / "tuning.json"))
        mlflow.log_artifact(str(RESULTS / "tuning_trials.csv"))

    print(f"Best val_acc={study.best_value:.4f}  params={best_cfg}")


if __name__ == "__main__":
    main()
