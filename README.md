# MLflow End-to-End: CNN Fashion-MNIST

Alur eksperimen lengkap: MLflow tracking server lokal (SQLite + folder artifact), baseline CNN,
tuning Optuna (nested runs), 3 seed (mean ± std), ablation, Model Registry (`@champion`), dan DVC.

## Struktur
```
src/
  prepare_data.py    # unduh Fashion-MNIST -> data/fmnist_subset.npz
  run_baseline.py    # tahap 3
  run_tuning.py      # tahap 4 (Optuna, >=20 trial, nested)
  run_seeds.py       # tahap 5 (3 seed -> mean ± std)
  run_ablation.py    # tahap 6 (augmentasi x dropout)
  register_predict.py# tahap 7 (registry + alias @champion + prediksi)
  make_report.py     # draf otomatis laporan (menimpa reports/REPORT.md!)
  make_figures.py    # grafik ringkasan dari results/*.csv
scripts/             # start_server.sh, run_all.sh
results/             # JSON/CSV hasil (di-commit)
reports/REPORT.md    # laporan
docs/screenshots/    # screenshot MLflow UI (compare view)
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1) Jalankan MLflow server (terminal terpisah)
```bash
bash scripts/start_server.sh        # UI: http://127.0.0.1:5000
```
Windows (tanpa bash): `mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000`

## 2) Siapkan & versikan dataset dengan DVC (sekali saja)
```bash
git init && dvc init
python -m src.prepare_data
dvc add data/fmnist_subset.npz
git add .dvc data/fmnist_subset.npz.dvc data/.gitignore
git commit -m "Track dataset with DVC"
# opsional remote:  dvc remote add -d storage ../dvc-storage && dvc push
```
Hash md5 dibaca dari `data/fmnist_subset.npz.dvc` dan dicatat sebagai param `dataset_md5` di setiap run.
Jangan jalankan ulang `prepare_data` setelah `dvc add` (file npz terkompresi bisa berubah byte-nya → hash berubah).

## 3) Jalankan pipeline
```bash
bash scripts/run_all.sh
```
atau per tahap:
```bash
python -m src.run_baseline
python -m src.run_tuning --n-trials 20
python -m src.run_seeds
python -m src.run_ablation
python -m src.register_predict
python -m src.make_report
```
Opsi `--epochs` tersedia di tiap skrip. Perkiraan waktu di CPU: 20–60 menit total (jauh lebih cepat dengan GPU).

## 4) Screenshot (deliverable)
Di MLflow UI → experiment `fmnist-cnn-e2e` → centang run (mis. semua trial di `optuna_tuning`, atau varian ablation)
→ **Compare**. Simpan sebagai `docs/screenshots/compare_tuning.png` dan `docs/screenshots/compare_ablation.png`,
lalu tangkap juga halaman **Models → fmnist-cnn** (alias `champion`).

## 5) Push ke GitHub
```bash
git add .
git commit -m "MLflow end-to-end experiment"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```
`mlflow.db` dan `mlartifacts/` di-ignore (besar); hasil ringkas ada di `results/` dan `reports/`.

## Memuat model champion
```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.pyfunc.load_model("models:/fmnist-cnn@champion")
```
