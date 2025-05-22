import numpy as np
import torch
import torch.nn as nn
from alg_collocation import generate_piecewise_chebyshev_basis

class FourierNoiseNet(nn.Module):
    def __init__(self, noise_dim=2, hidden_dim=64, num_frequencies=16):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_frequencies = num_frequencies

        # Random Fourier frequency matrix B: [noise_dim, num_frequencies]
        B = torch.randn(noise_dim, num_frequencies) * 10.0
        self.register_buffer('B', B)

        # Input size becomes 2 * num_frequencies after sin/cos
        input_dim = 2 * num_frequencies

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, noise_dim)
        )

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        x_encoded = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(x_encoded)

class CustomScoreTracker:
    def __init__(self, noise_dim=2, hidden_dim=64):
        self.call_count = 0
        self.recording = False
        
        self.noise_net = FourierNoiseNet(noise_dim=noise_dim, hidden_dim=hidden_dim, num_frequencies=4)

    def start_record(self, n_samples):
        self.call_count = 0
        self.n_samples = n_samples
        self.recording = True

    def end_record(self):
        if self.recording:
            return_count = self.call_count
        else:
            print("[Warning] NFE Recording was not started.")
            return_count = 0
        self.call_count = 0
        self.recording = False
        return return_count

    def score(self, x, t):
        return self.score_fn4(x, t)

    def _broadcast_t(self, t, n):
        t_arr = np.asarray(t)
        if t_arr.ndim == 0:
            t_arr = np.full((n,), t_arr)
        elif t_arr.ndim == 2 and t_arr.shape[1] == 1:
            t_arr = t_arr.ravel()
        else:
            t_arr = np.broadcast_to(t_arr, (n,))
        return t_arr

    def complex_score(self, x, t):
        if self.recording:
            self.call_count += x.shape[0] / self.n_samples

        x = np.asarray(x)
        t_arr = np.asarray(t)

        if t_arr.ndim == 0:
            t_arr = np.full((x.shape[0],), t_arr)
        elif t_arr.ndim == 2 and t_arr.shape[1] == 1:
            t_arr = t_arr.ravel()
        else:
            t_arr = np.broadcast_to(t_arr, (x.shape[0],))

        x1, x2 = x[:, 0], x[:, 1]
        r2 = x1 ** 2 + x2 ** 2
        exp_term = np.exp(-0.1 * r2)

        sinx1, cosx1 = np.sin(x1), np.cos(x1)
        sinx2, cosx2 = np.sin(x2), np.cos(x2)

        a, b = 2.0, 3.0
        tf = np.cos(np.pi * t_arr)  # time factor
        sech1_sq = 1.0 / np.cosh(x1 - 2.5) ** 2
        sech2_sq = 1.0 / np.cosh(x2 + 2.5) ** 2

        grad1 = a * (cosx1 * cosx2 - 0.2 * x1 * sinx1 * cosx2) * exp_term + b * sech1_sq * tf
        grad2 = a * (-sinx1 * sinx2 - 0.2 * x2 * sinx1 * cosx2) * exp_term + b * sech2_sq * tf

        return np.stack([grad1, grad2], axis=1)  # (n,2)

    def score_fn1(self, x, t):
        if self.recording:
            self.call_count += x.shape[0] / self.n_samples

        x = np.asarray(x)
        n = x.shape[0]
        t_arr = self._broadcast_t(t, n)

        x1, x2 = x[:, 0], x[:, 1]
        r2 = x1 ** 2 + x2 ** 2
        omega = 4.0
        amp = 1.5 + 0.5 * np.sin(2 * np.pi * t_arr)

        grad_r = -0.2 * r2
        grad_theta = omega

        cos_th, sin_th = x1 / np.sqrt(r2 + 1e-8), x2 / np.sqrt(r2 + 1e-8)
        g1 = amp * (grad_r * cos_th - grad_theta * sin_th)
        g2 = amp * (grad_r * sin_th + grad_theta * cos_th)
        return np.stack([g1, g2], axis=1)

    def score_fn2(self, x, t):
        if self.recording:
            self.call_count += x.shape[0] / self.n_samples

        x = np.asarray(x)

        t_arr = np.asarray(t)
        if t_arr.ndim == 0:
            t_arr = np.full((x.shape[0],), t_arr, dtype=x.dtype)
        elif t_arr.ndim == 2 and t_arr.shape[1] == 1:
            t_arr = t_arr.ravel()
        else:
            t_arr = np.broadcast_to(t_arr, (x.shape[0],))

        poly_coeff = 1.0 - 2.0 * t_arr + 1.5 * t_arr ** 2
        rot_coeff = np.sin(np.pi * t_arr)

        x1, x2 = x[:, 0], x[:, 1]
        g1 = poly_coeff * x1 + rot_coeff * x2  # (aI + sin*J)·x
        g2 = poly_coeff * x2 - rot_coeff * x1
        return np.stack([g1, g2], axis=1)

    def score_fn3(self, x, t):
        if self.recording:
            self.call_count += x.shape[0] / self.n_samples

        x = np.asarray(x, dtype=float)
        t_arr = np.asarray(t, dtype=float)

        # --- broadcast t ---
        if t_arr.ndim == 0:
            t_arr = np.full(x.shape[:-1], t_arr)
        else:
            t_arr = np.broadcast_to(t_arr, x.shape[:-1])

        # --- coefficients ---
        s = t_arr - 0.5
        edge_window = 1.0 - 4.0 * s ** 2

        lam = (1.0 - 2.0 * s ** 2) \
              + 0.35 * edge_window * np.sin(12.0 * np.pi * t_arr)

        omg = 4.0 * np.sin(8.0 * np.pi * t_arr) \
              + 0.8 * edge_window * np.sin(20.0 * np.pi * t_arr)

        x1, x2 = x[..., 0], x[..., 1]
        g1 = lam * x1 + omg * x2
        g2 = lam * x2 - omg * x1
        return np.stack([g1, g2], axis=-1)
    
    def score_fn4(self, x, t):
        if self.recording:
            self.call_count += x.shape[0] / self.n_samples

        # Convert x to numpy if it's not already
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-6
        x1, x2 = x[:, 0:1], x[:, 1:2]
        
        wells = np.tanh((x - 4)**2) - np.tanh((x + 4)**2)
        radial_target = 10 * x / norm
        radial = np.sin(norm * 2 * np.pi / 10) * (radial_target - x)
        s = (1 - t)
        score = 8.0 * s * wells + 2.0 * s * radial

        # Use a neural network to generate deterministic "noise"
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            noise = self.noise_net(x_tensor).numpy()
    
        return - score + noise * 5