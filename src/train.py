import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn

from .data import make_loaders
from .model import SimpleCNN
from .utils import dataset_hash, get_device, set_seed


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += loss_fn(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def train_and_log(cfg, seed, epochs, arrays, log_model=False):
    """Latih 1 model & log ke MLflow. HARUS dipanggil di dalam mlflow.start_run()."""
    set_seed(seed)
    device = get_device()
    md5, src = dataset_hash()

    train_dl, val_dl, test_dl = make_loaders(arrays, cfg["batch_size"], cfg["augment"], seed)
    model = SimpleCNN(cfg["n_filters"], cfg["hidden"], cfg["dropout"]).to(device)

    mlflow.log_params({**cfg, "seed": seed, "epochs": epochs, "device": device.type,
                       "dataset_md5": md5,
                       "n_model_params": sum(p.numel() for p in model.parameters())})
    mlflow.set_tag("dataset_hash_source", src)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss = tot_correct = n = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(y)
            tot_correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        val_loss, val_acc = evaluate(model, val_dl, device)
        mlflow.log_metrics({"train_loss": tot_loss / n, "train_acc": tot_correct / n,
                            "val_loss": val_loss, "val_acc": val_acc}, step=epoch)

    test_loss, test_acc = evaluate(model, test_dl, device)
    mlflow.log_metrics({"final_val_acc": val_acc, "final_val_loss": val_loss,
                        "test_acc": test_acc, "test_loss": test_loss})

    if log_model:
        model.cpu().eval()
        example = np.zeros((1, 1, 28, 28), dtype=np.float32)
        info = mlflow.pytorch.log_model(model, name="model", input_example=example)
        mlflow.set_tag("model_uri", info.model_uri)

    return {"run_id": mlflow.active_run().info.run_id,
            "val_acc": val_acc, "test_acc": test_acc,
            "train_acc": tot_correct / n}
