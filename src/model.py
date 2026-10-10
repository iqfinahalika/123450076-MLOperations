import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, n_filters=16, hidden=64, dropout=0.0, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),            # 14x14
            nn.Conv2d(n_filters, n_filters * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_filters * 2 * 7 * 7, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
