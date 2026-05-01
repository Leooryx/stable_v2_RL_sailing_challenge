"""
wind_fno.py  —  Fourier Neural Operator for wind field prediction
==================================================================
Trains a small FNO to predict the wind field 1 and 5 steps ahead,
given only the current field. Once trained, weights are saved and
loaded by the Dyna-Q agent for free look-ahead planning.

Why FNO for this problem
------------------------
The wind update in this environment is a global smooth rotation:
    u(t+1) = R(θ) · u(t),    θ ~ Normal(mean, std)
This is a *linear operator on a function space*, which is exactly
what FNO learns: a mapping between two function spaces (wind_t →
wind_{t+k}) that generalises across initial conditions.

Because the dynamics are the same across all wind scenarios (only
the initial field differs), a single trained FNO generalises to
training_3 and the hidden competition set without retraining.

Data collection
---------------
Run collect_wind_data() to gather (field_t, field_{t+1}) pairs
from the environment. 

Usage
-----
# 1. Collect data during Dyna-Q training (see dyna_q.py for hook)
# 2. wind_fno.py          → trains and saves weights
# 3. In agent: fno = load_fno()      → inference-only, no grad
"""

import numpy as np
import torch
import torch.nn as nn
import torch.fft
from pathlib import Path
from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = Path("src/wind_data.npz")
WEIGHTS_PATH = Path("src/wind_fno_weights.pt")

# ── Constants ──────────────────────────────────────────────────────────────
GRID_H, GRID_W = 128, 128   # spatial dims of the wind field
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Model
# =============================================================================

class SpectralConv2d(nn.Module):
    """
    Spectral convolution layer: multiply low-frequency Fourier modes by
    learnable complex weights, then invert.
    """

    def __init__(self, in_ch: int, out_ch: int, modes_x: int, modes_y: int):
        super().__init__()
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_ch * out_ch)
        # Shape: (in_ch, out_ch, modes_x, modes_y) — complex
        self.weights = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes_x, modes_y, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Real FFT along spatial dims → (B, C, H, W//2+1) complex
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Allocate output in frequency space
        out_ft = torch.zeros(B, self.weights.shape[1], H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # Multiply only the low-frequency corner
        # einsum: batch b, in-channel i, out-channel o, modes (m, n)
        out_ft[:, :, :self.modes_x, :self.modes_y] = torch.einsum(
            "bimn,iomn->bomn",
            x_ft[:, :, :self.modes_x, :self.modes_y],
            self.weights
        )

        # Inverse FFT back to spatial domain
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    """One Fourier layer: spectral path + bypass 1*1 conv + activation."""

    def __init__(self, width: int, modes_x: int, modes_y: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.bypass   = nn.Conv2d(width, width, kernel_size=1)
        self.norm     = nn.BatchNorm2d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(self.spectral(x) + self.bypass(x)))


class WindFNO(nn.Module):
    """
    Small FNO for wind field prediction.

    Input : (B, H, W, 2)  — current wind field (u, v)
    Output: (B, H, W, 2)  — predicted wind field k steps ahead

    Architecture:
        fc0  : lift input channels 2 → width
        4×   : FNOBlock (spectral conv + bypass)
        fc1/2: project width → 2

    """

    def __init__(self, modes_x: int = 12, modes_y: int = 12, width: int = 32):
        super().__init__()
        self.width = width

        self.fc0 = nn.Linear(2, width)

        self.blocks = nn.ModuleList([
            FNOBlock(width, modes_x, modes_y) for _ in range(4)
        ])

        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, 2)
        x = self.fc0(x)                     # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)           # (B, width, H, W) — for Conv2d

        for block in self.blocks:
            x = block(x)

        x = x.permute(0, 2, 3, 1)           # (B, H, W, width)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)                      # (B, H, W, 2)
        return x

    @torch.no_grad()
    def rollout(self, u0: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Autoregressive rollout: predict `steps` future wind fields.

        Args:
            u0   : (1, H, W, 2) tensor — current wind field
            steps: number of steps to predict ahead

        Returns:
            preds: (steps, H, W, 2) tensor — predicted future fields
        """
        preds = []
        u = u0
        for _ in range(steps):
            u = self(u)
            preds.append(u.squeeze(0))   # (H, W, 2)
        return torch.stack(preds)        # (steps, H, W, 2)


# =============================================================================
# Data collection
# =============================================================================

def collect_wind_data(
    scenarios: list[str] = ("training_1", "training_2"),
    episodes_per_scenario: int = 50,
    steps_per_episode: int = 200,
    predict_horizons: tuple[int, ...] = (1, 5),
    save_path: Path = DATA_PATH,
) -> None:
    """
    Collect (wind_t, wind_{t+k}) pairs from the environment.

    Each entry is a snapshot of the full 128*128*2 wind field.
    We store pairs for each horizon in predict_horizons so we can
    train a single model that predicts both 1-step and 5-step ahead
    (via autoregressive rollout at inference).

    The data collection is intentionally lightweight — it runs
    independently of the Dyna-Q agent and only needs ~100 episodes
    (~10k transitions) to capture the full range of wind dynamics.

    Saves: wind_data.npz with arrays:
        inputs  : (N, H, W, 2)  — wind at time t
        targets1: (N, H, W, 2)  — wind at time t+1
        targets5: (N, H, W, 2)  — wind at time t+5
    """
    print("Collecting wind field data...")
    inputs_list  = []
    target1_list = []
    target5_list = []

    for scenario in scenarios:
        for ep in range(episodes_per_scenario):
            env = SailingEnv(**get_wind_scenario(scenario))
            observation, _ = env.reset(seed=ep * 100)

            # Buffer of recent wind fields for multi-step targets
            # We need a window of size max(predict_horizons)+1
            horizon = max(predict_horizons)
            window  = []

            for step in range(steps_per_episode + horizon):
                # Extract current wind field from observation
                wf_flat = observation[6: 6 + GRID_H * GRID_W * 2]
                wf = wf_flat.reshape(GRID_H, GRID_W, 2).copy()
                window.append(wf)

                if len(window) > horizon + 1:
                    window.pop(0)

                # Once we have enough history, record pairs
                if len(window) == horizon + 1:
                    inputs_list.append(window[0].copy())
                    target1_list.append(window[1].copy())           # t+1
                    target5_list.append(window[min(5, horizon)].copy())  # t+5

                # Step with action 0 (direction doesn't matter for wind)
                observation, _, done, truncated, _ = env.step(0)
                if done or truncated:
                    break

        print(f"  {scenario}: {len(inputs_list)} pairs so far")

    inputs  = np.array(inputs_list,  dtype=np.float32)
    target1 = np.array(target1_list, dtype=np.float32)
    target5 = np.array(target5_list, dtype=np.float32)

    np.savez_compressed(save_path, inputs=inputs, targets1=target1, targets5=target5)
    print(f"Saved {len(inputs)} samples to {save_path}")
    print(f"  Input shape : {inputs.shape}")
    print(f"  Target1     : {target1.shape}")
    print(f"  Target5     : {target5.shape}")


# =============================================================================
# Training
# =============================================================================

def train_fno(
    data_path: Path    = DATA_PATH,
    weights_path: Path = WEIGHTS_PATH,
    epochs: int        = 10,
    batch_size: int    = 32,
    lr: float          = 1e-3,
    val_split: float   = 0.1,
) -> WindFNO:
    """
    Train the FNO on collected wind data.

    Loss: MSE on 1-step prediction + 0.5 * MSE on 5-step rollout.
    The 5-step term encourages the model to stay accurate over the
    planning horizon used by the Dyna-Q agent.
    """
    print(f"Loading data from {data_path}...")
    data    = np.load(data_path)
    inputs  = torch.tensor(data["inputs"],  dtype=torch.float32)
    target1 = torch.tensor(data["targets1"], dtype=torch.float32)
    target5 = torch.tensor(data["targets5"], dtype=torch.float32)

    N = len(inputs)
    n_val = int(N * val_split)
    n_train = N - n_val

    # Normalise: zero-mean unit-variance per channel
    mean = inputs[:n_train].mean(dim=(0, 1, 2), keepdim=True)
    std  = inputs[:n_train].std(dim=(0, 1, 2),  keepdim=True).clamp(min=1e-6)

    def norm(x):  return (x - mean) / std
    def denorm(x): return x * std + mean

    inputs_n  = norm(inputs)
    target1_n = norm(target1)
    target5_n = norm(target5)

    train_idx = torch.arange(n_train)
    val_idx   = torch.arange(n_train, N)

    model = WindFNO(modes_x=12, modes_y=12, width=32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.MSELoss()

    print(f"Training FNO: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  {n_train} train / {n_val} val samples | device={DEVICE}")
    print()

    best_val = float("inf")

    for epoch in tqdm(range(epochs)):
        model.train()
        perm  = torch.randperm(n_train)
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i: i + batch_size]
            x  = inputs_n[idx].to(DEVICE)    # (B, H, W, 2)
            y1 = target1_n[idx].to(DEVICE)
            y5 = target5_n[idx].to(DEVICE)

            # 1-step prediction
            pred1 = model(x)
            loss1 = loss_fn(pred1, y1)

            # 5-step autoregressive rollout (with gradient)
            u = x
            for _ in range(5):
                u = model(u)
            loss5 = loss_fn(u, y5)

            loss = loss1 + 0.5 * loss5

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Validation
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                x_val  = inputs_n[val_idx].to(DEVICE)
                y1_val = target1_n[val_idx].to(DEVICE)
                val_loss = loss_fn(model(x_val), y1_val).item()

            avg_train = total_loss / n_batches
            marker = "  *best*" if val_loss < best_val else ""
            print(f"Epoch {epoch+1:>3}/{epochs} | "
                  f"train={avg_train:.5f} | val={val_loss:.5f}{marker}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model_state":  model.state_dict(),
                    "mean":         mean,
                    "std":          std,
                    "modes_x":      12,
                    "modes_y":      12,
                    "width":        32,
                }, weights_path)

    print(f"\nBest val loss: {best_val:.5f} — weights saved to {weights_path}")
    return model


# =============================================================================
# Inference helper
# =============================================================================

def load_fno(weights_path: Path = WEIGHTS_PATH) -> "WindFNOInference":
    """Load trained FNO weights and return an inference wrapper."""
    return WindFNOInference(weights_path)


class WindFNOInference:
    """
    CPU-friendly inference wrapper around WindFNO.

    Handles normalisation/denormalisation internally so callers
    work in the same units as the environment (raw wind vectors).

    Usage in agent:
        fno = load_fno()
        future_fields = fno.predict(current_wind_field_np, steps=5)
        # future_fields: (5, 128, 128, 2) numpy array
    """

    def __init__(self, weights_path: Path):
        ck = torch.load(weights_path, map_location="cpu")
        self.model = WindFNO(
            modes_x=ck["modes_x"],
            modes_y=ck["modes_y"],
            width=ck["width"],
        )
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()

        self.mean = ck["mean"]   # (1,1,1,2)
        self.std  = ck["std"]    # (1,1,1,2)

    def predict(self, wind_field_np: np.ndarray, steps: int = 5) -> np.ndarray:
        """
        Predict `steps` future wind fields from the current field.

        Args:
            wind_field_np : (128, 128, 2) numpy array — current wind field
            steps         : number of steps to predict ahead (1–5 recommended)

        Returns:
            (steps, 128, 128, 2) numpy array — predicted future fields
            Index 0 = t+1, index 4 = t+5.
        """
        x = torch.tensor(wind_field_np, dtype=torch.float32).unsqueeze(0)  # (1,H,W,2)
        x_norm = (x - self.mean) / self.std

        with torch.no_grad():
            preds_norm = self.model.rollout(x_norm, steps=steps)   # (steps,H,W,2)

        preds = preds_norm * self.std.squeeze(0) + self.mean.squeeze(0)
        return preds.numpy()


# =============================================================================
# Entry point: collect data then train
# =============================================================================

if __name__ == "__main__":
    import sys

    if "--collect" in sys.argv or not DATA_PATH.exists():
        collect_wind_data(
            scenarios=["training_1", "training_2"],
            episodes_per_scenario=50,
            steps_per_episode=300,
        )

    train_fno(epochs=10)