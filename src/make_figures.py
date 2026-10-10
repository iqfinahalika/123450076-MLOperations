"""Buat grafik ringkasan dari results/*.csv (tidak butuh MLflow/torch)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RESULTS

FIG = RESULTS / "figures"
ORDER = ["full", "no_augmentation", "no_dropout", "no_aug_no_dropout"]


def tuning_fig():
    t = pd.read_csv(RESULTS / "tuning_trials.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cmap = plt.get_cmap("viridis")
    filters = sorted(t.params_n_filters.unique())
    for i, f in enumerate(filters):
        d = t[t.params_n_filters == f]
        ax[0].scatter(d.params_lr, d.value, s=60, color=cmap(i / max(len(filters) - 1, 1)), label=f"n_filters={f}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("learning rate"); ax[0].set_ylabel("val_acc")
    ax[0].set_title("20 trial Optuna: lr vs val_acc"); ax[0].legend(fontsize=8)
    g = t.groupby("params_n_filters").value
    ax[1].bar([str(f) for f in filters], g.mean().values, yerr=g.std(ddof=1).fillna(0).values, capsize=4, color="#4c78a8")
    for i, f in enumerate(filters):
        ax[1].text(i, 0.77, f"n={int(g.count()[f])}", ha="center", fontsize=8)
    ax[1].set_ylim(0.75, 0.9); ax[1].set_xlabel("n_filters"); ax[1].set_ylabel("rata-rata val_acc")
    ax[1].set_title("Rata-rata val_acc per n_filters")
    fig.tight_layout(); fig.savefig(FIG / "tuning_overview.png", dpi=150); plt.close(fig)
    print(t.groupby("params_n_filters").value.agg(["count", "mean", "max"]).round(4))


def ablation_fig():
    a = pd.read_csv(RESULTS / "ablation.csv")
    a["gap"] = a.train_acc - a.val_acc
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (col, title) in enumerate([("test_acc", "test_acc (mean ± std, titik = seed)"),
                                      ("gap", "gap train_acc − val_acc")]):
        g = a.groupby("variant")[col]
        m, s = g.mean().reindex(ORDER), g.std(ddof=1).reindex(ORDER)
        ax[k].bar(range(4), m.values, yerr=s.values, capsize=4, color="#59a14f" if k == 0 else "#e15759", alpha=.8)
        for i, v in enumerate(ORDER):
            ax[k].scatter([i] * 3, a[a.variant == v][col], color="k", s=14, zorder=3)
        ax[k].set_xticks(range(4)); ax[k].set_xticklabels(ORDER, rotation=15, fontsize=8)
        ax[k].set_title(title)
    ax[0].set_ylim(0.87, 0.91)
    fig.tight_layout(); fig.savefig(FIG / "ablation_overview.png", dpi=150); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tuning_fig(); ablation_fig()
    print("Grafik tersimpan di", FIG)


if __name__ == "__main__":
    main()
