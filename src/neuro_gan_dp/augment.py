from __future__ import annotations
import torch, torch.nn as nn, torch.optim as optim
import numpy as np

# Minimal tabular GAN (MLP-based). Intended as a lightweight baseline.
class Generator(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_gan(X_train: np.ndarray, n_synth: int = 500, epochs: int = 50, batch_size: int = 64, latent_dim: int = 32, lr: float = 1e-3, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = X_train.shape[1]
    G, D = Generator(latent_dim, in_dim).to(device), Discriminator(in_dim).to(device)
    optG, optD = optim.Adam(G.parameters(), lr=lr), optim.Adam(D.parameters(), lr=lr)
    X = torch.tensor(X_train, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        perm = torch.randperm(X.size(0), device=device)
        for i in range(0, X.size(0), batch_size):
            idx = perm[i:i+batch_size]
            real = X[idx]

            # Train D
            z = torch.randn(real.size(0), latent_dim, device=device)
            fake = G(z).detach()
            optD.zero_grad()
            lossD = - (torch.log(D(real) + 1e-8).mean() + torch.log(1 - D(fake) + 1e-8).mean())
            lossD.backward(); optD.step()

            # Train G
            z = torch.randn(real.size(0), latent_dim, device=device)
            fake = G(z)
            optG.zero_grad()
            lossG = - torch.log(D(fake) + 1e-8).mean()
            lossG.backward(); optG.step()

    with torch.no_grad():
        z = torch.randn(n_synth, latent_dim, device=device)
        synth = G(z).cpu().numpy()
    return synth
