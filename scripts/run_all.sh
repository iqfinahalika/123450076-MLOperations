#!/usr/bin/env bash
# Jalankan seluruh pipeline (server MLflow harus sudah hidup di terminal lain).
set -e
cd "$(dirname "$0")/.."
python -m src.run_baseline
python -m src.run_tuning --n-trials 20
python -m src.run_seeds
python -m src.run_ablation
python -m src.register_predict
python -m src.make_report
