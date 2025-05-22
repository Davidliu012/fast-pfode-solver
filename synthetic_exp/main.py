"""
Collocation-based ODE Solver Benchmarking with Statistical Evaluation
Author: Anonymous
Date: 2025-05-17
"""

import csv
import numpy as np
from tqdm import tqdm

from synthetic_score import CustomScoreTracker
from alg_collocation import (
    generate_piecewise_chebyshev_basis,
    generate_rbf_basis,
    generate_fourier_interpolatory_basis,
    iterative_collocation
)
from utils import estimate_kl_divergence, runge_kutta_sampler

# --------------------------- Configuration ---------------------------
def get_config():
    return {
        "n_samples": 2000,
        "n_steps_gt": 10000,
        "iter_nums": 6,
        "K_list": [4, 2],
        "N_list": [3, 6],
        "eps": 0,
        "num_trials": 20,
        "score_fns": {
            "score_fn1": "score_fn1",
            "score_fn2": "score_fn2",
            "score_fn3": "score_fn3",
            "score_fn4": "score_fn4",
        },
        "basis_generators": {
            "chebyshev": generate_piecewise_chebyshev_basis,
            "fourier": generate_fourier_interpolatory_basis,
            "rbf_linear": lambda K, N, eps: generate_rbf_basis(K, N, eps, "linear"),
            "rbf_cubic": lambda K, N, eps: generate_rbf_basis(K, N, eps, "cubic"),
            "rbf_quintic": lambda K, N, eps: generate_rbf_basis(K, N, eps, "quintic"),
        }
    }

# ------------------------ Experiment Pipeline ------------------------
def run_collocation_experiments(cfg):
    np.random.seed(42)
    seeds = np.random.randint(0, 10000, cfg["num_trials"])
    results = []
    score_estimator = CustomScoreTracker()

    for score_name, score_attr in tqdm(cfg["score_fns"].items(), desc="Score functions"):
        np.random.seed(9999)
        samples_gt = np.random.multivariate_normal([0, 0], np.eye(2), size=cfg["n_samples"])
        x_gt = runge_kutta_sampler(samples_gt, cfg["n_steps_gt"], score_attr, score_estimator)

        for basis_name, basis_func in tqdm(cfg["basis_generators"].items(), desc=f"{score_name} Collocation", leave=False):
            for K, N in zip(cfg["K_list"], cfg["N_list"]):
                kl_list, nfe_list = [], []
                interval_nodes, eval_nodes, basis = basis_func(K, N, cfg["eps"])

                for seed in seeds:
                    np.random.seed(seed)
                    samples_test = np.random.multivariate_normal([0, 0], np.eye(2), size=cfg["n_samples"])

                    def target_ode(x, t):
                        return -0.5 * getattr(score_estimator, score_attr)(x, t)

                    score_estimator.start_record(cfg["n_samples"])
                    x_fn, data, _ = iterative_collocation(
                        x_init=samples_test,
                        target_ode=target_ode,
                        iter_nums=cfg["iter_nums"],
                        interval_nodes=interval_nodes,
                        eval_nodes=eval_nodes,
                        basis=basis
                    )
                    nfe = score_estimator.end_record()
                    x_out = x_fn(t=1.0, data=data)
                    kl = estimate_kl_divergence(x_gt, x_out)

                    kl_list.append(kl)
                    nfe_list.append(nfe)

                results.append([
                    score_name,
                    "Collocation",
                    basis_name,
                    K,
                    N,
                    f"{np.mean(nfe_list):.2f}±{np.std(nfe_list):.2f}",
                    f"{np.mean(kl_list):.4f}±{np.std(kl_list):.4f}"
                ])

        results.append(["", "", "", "", "", "", ""])
    return results

# --------------------------- Entry Point -----------------------------
def main():
    cfg = get_config()
    results = run_collocation_experiments(cfg)
    with open("basis_exp_results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["score_fn", "method", "basis", "K", "N", "NFE", "KL_divergence"])
        writer.writerows(results)
    print("Experiment completed. Results saved to 'basis_exp_results.csv'")

if __name__ == "__main__":
    main()