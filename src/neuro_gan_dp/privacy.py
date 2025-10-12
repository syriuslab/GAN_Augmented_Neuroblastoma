from __future__ import annotations
import torch, torch.nn as nn, torch.optim as optim
from opacus import PrivacyEngine
from .models import MLPClassifier
import numpy as np

def train_dp_mlp(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                 epochs: int = 25, batch_size: int = 64, lr: float = 1e-3,
                 max_grad_norm: float = 1.0, noise_multiplier: float = 1.0, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(in_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    train_dataset = torch.utils.data.TensorDataset(Xtr, ytr)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    model.to(device)

    best_val = -1.0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # quick val AUC proxy using accuracy (to avoid heavy deps); real AUC in evaluate
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(Xv))
            acc = ((preds > 0.5) == (yv > 0.5)).float().mean().item()
            if acc > best_val:
                best_val = acc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model
