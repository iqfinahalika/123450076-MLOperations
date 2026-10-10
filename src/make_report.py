"""Hasilkan reports/REPORT.md dari results/*.json|csv (angka diambil otomatis dari hasil run)."""
import json

import pandas as pd

from .config import REPORTS, RESULTS


def load_json(path):
    return json.loads(path.read_text())


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in df.astype(object).itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def pm(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def verdict(delta, std_a, std_b):
    if delta > max(std_a, std_b):
        return f"meningkatkan akurasi test sebesar {delta*100:.2f} poin (lebih besar dari std) → **hipotesis didukung**"
    if delta > 0:
        return (f"cenderung meningkatkan akurasi {delta*100:.2f} poin, namun selisihnya masih dalam kisaran std "
                "→ **belum terbukti jelas** (butuh lebih banyak seed)")
    return f"tidak meningkatkan akurasi (selisih {delta*100:.2f} poin) → **hipotesis tidak didukung**"


def main():
    base = load_json(RESULTS / "baseline.json")
    tune = load_json(RESULTS / "tuning.json")
    seeds = load_json(RESULTS / "seeds_summary.json")
    seeds_df = pd.read_csv(RESULTS / "seeds.csv")
    abl = pd.read_csv(RESULTS / "ablation.csv")
    trials = pd.read_csv(RESULTS / "tuning_trials.csv")

    # --- tabel ---
    g = abl.groupby("variant")
    abl_tab = pd.DataFrame({
        "varian": g.size().index,
        "val_acc (mean ± std)": [pm(x.val_acc.mean(), x.val_acc.std(ddof=1)) for _, x in g],
        "test_acc (mean ± std)": [pm(x.test_acc.mean(), x.test_acc.std(ddof=1)) for _, x in g],
        "train_acc (mean)": [f"{x.train_acc.mean():.4f}" for _, x in g],
    })
    stats = {k: (x.test_acc.mean(), x.test_acc.std(ddof=1)) for k, x in g}

    pcols = [c for c in trials.columns if c.startswith("params_")]
    top = trials.sort_values("value", ascending=False).head(5)[["number", "value"] + pcols]
    top.columns = [c.replace("params_", "") for c in top.columns]

    seed_tab = seeds_df[["seed", "val_acc", "test_acc"]]
    best_cfg_md = "\n".join(f"- `{k}` = {v}" for k, v in tune["best_params"].items())

    d_aug = stats["full"][0] - stats["no_augmentation"][0]
    d_drop = stats["full"][0] - stats["no_dropout"][0]
    d_tune = seeds["test_acc_mean"] - base["test_acc"]
    gap_full = abl[abl.variant == "full"].eval("train_acc - val_acc").mean()
    gap_noaug = abl[abl.variant == "no_augmentation"].eval("train_acc - val_acc").mean()

    md = f"""# Laporan Singkat: Eksperimen End-to-End CNN dengan MLflow

**Dataset:** Fashion-MNIST subset (10.000 train / 2.000 val / 2.000 test), diversi dengan DVC.
**Tracking:** MLflow server lokal (SQLite + folder artifact), experiment `fmnist-cnn-e2e`.

## 1. Hipotesis
- **H1** — Hyperparameter tuning (Optuna, {tune['n_trials']} trial) menghasilkan konfigurasi yang lebih baik daripada baseline CNN default.
- **H2** — Augmentasi data (random crop + horizontal flip) meningkatkan generalisasi (akurasi test naik, gap train–val mengecil).
- **H3** — Dropout mengurangi overfitting sehingga meningkatkan akurasi test.

## 2. Metode singkat
1. Baseline: CNN 2 blok konv, tanpa augmentasi/dropout, {base['epochs']} epoch, seed 0.
2. Tuning: Optuna TPE, {tune['n_trials']} trial (nested run), objektif = val_acc, {tune['epochs']} epoch/trial.
   Ruang cari: n_filters, hidden, dropout, lr, batch_size, weight_decay (augmentasi dibuat tetap aktif).
3. Konfigurasi terbaik dijalankan dengan 3 seed (0, 1, 2), {seeds['epochs']} epoch; std = sampel (ddof=1).
4. Ablation faktorial 2×2: augmentasi × dropout, masing-masing 3 seed.
5. Model terbaik (val_acc tertinggi, bukan test) diregistrasi ke Model Registry dengan alias `@champion`.
6. Hash dataset DVC (`dataset_md5`) dicatat pada setiap run.

Konfigurasi terbaik hasil tuning:
{best_cfg_md}

## 3. Hasil
### 3.1 Baseline vs konfigurasi terbaik
| Model | val_acc | test_acc |
|---|---|---|
| Baseline (1 seed) | {base['val_acc']:.4f} | {base['test_acc']:.4f} |
| Terbaik (3 seed, mean ± std) | {pm(seeds['val_acc_mean'], seeds['val_acc_std'])} | {pm(seeds['test_acc_mean'], seeds['test_acc_std'])} |

Hasil per seed:

{md_table(seed_tab)}

### 3.2 Top-5 trial Optuna
{md_table(top)}

### 3.3 Ablation (3 seed)
{md_table(abl_tab)}

Gap train–val rata-rata: full = {gap_full:.4f}, tanpa augmentasi = {gap_noaug:.4f}.

### 3.4 Screenshot MLflow UI (compare view)
![Compare trial tuning](../docs/screenshots/compare_tuning.png)
![Compare ablation](../docs/screenshots/compare_ablation.png)

## 4. Kesimpulan
- **H1:** konfigurasi tuned dibanding baseline: selisih test_acc {d_tune*100:+.2f} poin. {"Tuning memberi perbaikan." if d_tune > 0 else "Tuning tidak memberi perbaikan pada test."} (Catatan: baseline hanya 1 seed dan tuned memakai epoch lebih banyak, sehingga perbandingan ini indikatif.)
- **H2:** augmentasi {verdict(d_aug, stats['full'][1], stats['no_augmentation'][1])}.
- **H3:** dropout {verdict(d_drop, stats['full'][1], stats['no_dropout'][1])}.
- **Keterbatasan:** subset kecil, hanya 3 seed, tuning memakai 1 seed dan epoch pendek; selisih kecil antarvarian perlu dibaca dengan hati-hati.

## 5. Reproduksi
Lihat `README.md`. Model terbaik dapat dimuat dengan `models:/fmnist-cnn@champion`.
"""
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "REPORT.md").write_text(md, encoding="utf-8")
    print(f"Laporan ditulis ke {REPORTS / 'REPORT.md'}")


if __name__ == "__main__":
    main()
