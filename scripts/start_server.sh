#!/usr/bin/env bash
# MLflow tracking server lokal: backend SQLite + artifact di folder ./mlartifacts
set -e
cd "$(dirname "$0")/.."
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 --port 5000
