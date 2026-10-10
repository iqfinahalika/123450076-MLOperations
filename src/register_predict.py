"""Tahap 7: registrasi model terbaik, alias @champion, load via models:/...@champion & prediksi."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # paksa CPU: model disimpan di CPU (format pt2)

import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow import MlflowClient

from .config import CLASS_NAMES, EXPERIMENT, REGISTERED_MODEL
from .data import load_arrays, normalize
from .utils import dataset_hash, setup_mlflow


def main():
    setup_mlflow()
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)

    parent = client.search_runs([exp.experiment_id],
                                filter_string="tags.mlflow.runName = 'best_config_3seeds'",
                                order_by=["attribute.start_time DESC"], max_results=1)
    if not parent:
        raise SystemExit("[!] Run 'best_config_3seeds' belum ada. Jalankan python -m src.run_seeds")
    children = client.search_runs([exp.experiment_id],
                                  filter_string=f"tags.mlflow.parentRunId = '{parent[0].info.run_id}'",
                                  order_by=["metrics.final_val_acc DESC"])  # pilih via VAL, bukan test
    best = children[0]
    model_uri = best.data.tags.get("model_uri") or f"runs:/{best.info.run_id}/model"
    print(f"Model terbaik: run {best.info.run_name} (val_acc={best.data.metrics['final_val_acc']:.4f})")

    mv = mlflow.register_model(model_uri, REGISTERED_MODEL)
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", mv.version)
    client.set_model_version_tag(REGISTERED_MODEL, mv.version, "dataset_md5", dataset_hash()[0])
    print(f"Terdaftar: {REGISTERED_MODEL} v{mv.version}  alias @champion")

    # --- Load via alias & prediksi ---
    champion_uri = f"models:/{REGISTERED_MODEL}@champion"
    with mlflow.start_run(run_name="champion_predict"):
        mlflow.set_tags({"stage": "serving_check", "model_uri": champion_uri,
                         "model_version": str(mv.version)})
        loaded = mlflow.pyfunc.load_model(champion_uri)
        arrays = load_arrays()
        x = normalize(arrays["x_test"]).numpy().astype(np.float32)
        logits = np.asarray(loaded.predict(x))
        pred = logits.argmax(1)
        acc = float((pred == arrays["y_test"]).mean())
        mlflow.log_metric("champion_test_acc", acc)

    print(f"Akurasi test via {champion_uri}: {acc:.4f}")
    for i in range(5):
        print(f"  sampel {i}: prediksi={CLASS_NAMES[pred[i]]:<12} asli={CLASS_NAMES[arrays['y_test'][i]]}")


if __name__ == "__main__":
    main()
