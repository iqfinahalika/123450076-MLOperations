# MLflow End-to-End: CNN Fashion-MNIST

Alur eksperimen lengkap: MLflow tracking server lokal (SQLite + folder artifact), baseline CNN,
tuning Optuna (nested runs), 3 seed (mean ± std), ablation, Model Registry (`@champion`), dan DVC.

## Struktur
```
src/
  prepare_data.py    # unduh Fashion-MNIST 
  run_baseline.py    # tahap 3
  run_tuning.py      # tahap 4 (Optuna, >=20 trial, nested)
  run_seeds.py       # tahap 5 (3 seed -> mean ± std)
  run_ablation.py    # tahap 6 (augmentasi x dropout)
  register_predict.py# tahap 7 (registry + alias @champion + prediksi)
  make_figures.py    # grafik ringkasan 
scripts/             # start_server.sh, run_all.sh
results/             # 
reports/REPORT.md    # laporan
docs/screenshots/    # screenshot MLflow
```

## Memuat model champion
```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.pyfunc.load_model("models:/fmnist-cnn@champion")
```
