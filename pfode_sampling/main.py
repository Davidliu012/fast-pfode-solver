import os
import torch
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

from models import ncsnpp  # or ddpm depending on config
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint
import datasets
import sde_lib
from sampling import get_score_fn
from alg_collocation import generate_piecewise_chebyshev_basis, iterative_collocation

import importlib.util

# Sampling algorithms to run
sample_algorithms = [
    ["euler", 100],
    ["euler", 200],
    ["euler", 500],
    ["euler", 1500],
    ["heun", 50],
    ["heun", 100],
    ["heun", 250],
    ["heun", 500],
    ["heun", 1000],
    ["collocation", (5, 3, 15)],
    ["collocation", (20, 3, 15)],
    ["collocation", (40, 3, 15)],
    ["collocation", (60, 3, 15)]
]

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

def euler_sampler(model, sde, t_init, shape, inverse_scaler, num_steps=1000, eps=1e-3, device='cuda', x_init=None):
    with torch.no_grad():
        if x_init is None:
            x = sde.prior_sampling(shape).to(device, dtype=torch.float32)
        else:
            x = x_init.to(device, dtype=torch.float32)

        t_steps = torch.linspace(t_init, eps, num_steps, device=device, dtype=torch.float32)
        dt = -(t_init - eps) / (num_steps - 1)

        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)

        for step, t in tqdm(enumerate(t_steps), total=num_steps, desc="Euler Sampling"):
            vec_t = torch.ones(x.shape[0], device=device) * t
            drift, _ = rsde.sde(x, vec_t)
            x = x + drift * dt

        return inverse_scaler(x)

def heun_sampler(model, sde, t_init, shape, inverse_scaler, num_steps=1000, eps=1e-3, device='cuda', x_init=None):
    with torch.no_grad():
        if x_init is None:
            x = sde.prior_sampling(shape).to(device, dtype=torch.float32)
        else:
            x = x_init.to(device, dtype=torch.float32)

        t_steps = torch.linspace(t_init, eps, num_steps, device=device, dtype=torch.float32)
        dt = -(t_init - eps) / (num_steps - 1)

        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)

        for step, t in tqdm(enumerate(t_steps), total=num_steps, desc="Heun Sampling"):
            vec_t = torch.ones(x.shape[0], device=device) * t
            drift1, _ = rsde.sde(x, vec_t)
            x_predict = x + dt * drift1
            drift2, _ = rsde.sde(x_predict, vec_t)
            x = x + (dt / 2.0) * (drift1 + drift2)

        return inverse_scaler(x)

def collocation_sampler(model, sde, t_init, shape, inverse_scaler, K, N, iter_nums, eps=1e-3, device='cuda', x_init=None):
    assert 0 < t_init <= 1, "t_init should be in the range (0, 1]"
    with torch.no_grad():
        if x_init is None:
            x_init = sde.prior_sampling(shape).to(device, dtype=torch.float32)
        else:
            x_init = x_init.to(device, dtype=torch.float32)

        B = x_init.shape[0]
        D = int(np.prod(x_init.shape[1:]))
        x_init_flat = x_init.reshape(B, D).cpu().numpy()

        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)

        def target_ode(x_numpy, t_scalar):
            t = t_init * (1 - t_scalar)
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=device).reshape(*x_init.shape)
            vec_t = torch.full((x_tensor.shape[0],), t, dtype=torch.float32, device=device)
            drift, _ = rsde.sde(x_tensor, vec_t)
            return - t_init * drift.reshape(x_tensor.shape[0], -1).cpu().numpy()

        interval_nodes, eval_nodes, basis = generate_piecewise_chebyshev_basis(K, N, eps)
        
        x_fn, trained_results, nfe = iterative_collocation(
            x_init_flat, target_ode, iter_nums, interval_nodes, eval_nodes, basis, 
            adaptive_early_stopping=True, 
            show_iterative_progress=True
        )
        
        t_final = 1 - eps
        x_final = x_fn(t_final, data=trained_results)
        x_final = torch.tensor(x_final, dtype=torch.float32, device=device).reshape(shape)
        return inverse_scaler(x_final), nfe

def main():
    # Load config
    ckpt_path = "checkpoints/subvp/cifar10_ddpmpp_deep_continuous/checkpoint_18.pth"
    config_path = "configs/subvp/cifar10_ddpmpp_deep_continuous.py"
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.get_config()

    # Load model
    score_model = mutils.create_model(config).to(device)
    optimizer = torch.optim.Adam(score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    state = restore_checkpoint(ckpt_path, state, device=device)
    ema.copy_to(score_model.parameters())
    score_model.eval()

    num_params = sum(p.numel() for p in score_model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")

    # Setup SDE
    sde_name = config.training.sde.lower()
    if sde_name == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif sde_name == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif sde_name == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"Unknown SDE: {sde_name}")

    # Data shape
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    shape = (config.eval.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)
    total_samples = config.eval.num_samples
    batch_size = config.eval.batch_size
    num_batches = total_samples // batch_size

    for method, param in sample_algorithms:
        if method == "collocation":
            K, N, I = param
            param_str = f"K{K}N{N}M{I}"
        else:
            param_str = f"{param}"
        output_dir = f"samples_from_ode/{method}_{param_str}_cifar10"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n=== Sampling with {method.upper()} ===")
        total_nfe = 0

        for i in range(num_batches):
            t_init = sde.T
            if method == 'euler':
                samples = euler_sampler(score_model, sde, t_init, shape, inverse_scaler,
                                        num_steps=param, eps=sampling_eps, device=device)
                nfe = param
            elif method == 'heun':
                samples = heun_sampler(score_model, sde, t_init, shape, inverse_scaler,
                                       num_steps=param, eps=sampling_eps, device=device)
                nfe = param * 2
            elif method == 'collocation':
                samples, nfe = collocation_sampler(score_model, sde, t_init, shape, inverse_scaler,
                                                   K=K, N=N, iter_nums=I, eps=0, device=device)
            else:
                raise NotImplementedError(f"Sampling method {method} not implemented.")

            total_nfe += nfe

            samples = torch.clamp(samples, 0.0, 1.0)
            for j in range(samples.shape[0]):
                save_image(samples[j], os.path.join(output_dir, f"{i * batch_size + j:05d}.png"))

        print(f"[{method}] Average NFE per sample: {total_nfe / num_batches}")

if __name__ == "__main__":
    main()
