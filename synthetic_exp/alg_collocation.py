import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from utils import plot_all_piecewise_bases

def generate_piecewise_chebyshev_basis(K, N, eps):
    """
    Generate piecewise Chebyshev Lagrange basis functions on K uniform subintervals over [0, 1].

    Parameters
    ----------
    K : int
        Number of subintervals (uniformly spaced over [0, 1])
    N : int
        Number of Chebyshev nodes per subinterval

    Returns
    -------
    interval_nodes : np.ndarray, shape (K + 1,)
        Breakpoints T_0, ..., T_K of the full time domain.

    eval_nodes : list of np.ndarray
        eval_nodes[k] contains N Chebyshev points mapped to [T_k, T_{k+1}]

    basis : list of list of callables
        basis[k][j] is the j-th Lagrange basis φ_j^k on interval k.
    """

    # Step 1: Define uniform intervals
    interval_nodes = np.linspace(0.0, 1.0 - eps, K + 1)  # T_0, ..., T_K

    # Step 2: Chebyshev nodes on [-1, 1]
    j = np.arange(1, N + 1) # dimension N
    cheb_nodes = np.cos((2 * j - 1) / (2 * N) * np.pi)  # shape (N,)
    cheb_nodes_std = (cheb_nodes + 1) / 2  # map to [0, 1]

    eval_nodes = []
    basis = []

    for k in range(K):
        T_k, T_k1 = interval_nodes[k], interval_nodes[k + 1]
        # Rescale to [T_k, T_k+1]
        nodes_k = T_k + (T_k1 - T_k) * cheb_nodes_std
        nodes_k = np.sort(nodes_k)
        eval_nodes.append(nodes_k)

        # Build Lagrange basis for this interval
        def lagrange_phi(j_idx, nodes):
            def phi(t):
                result = 1.0
                for i, t_i in enumerate(nodes):
                    if i != j_idx:
                        result *= (t - t_i) / (nodes[j_idx] - t_i)
                return result
            return phi

        basis_k = [lagrange_phi(j_idx, nodes_k) for j_idx in range(N)]
        basis.append(basis_k)

    return interval_nodes, eval_nodes, basis

def generate_rbf_basis(K, N, eps, rbf_type, epsilon=1.0):
    interval_nodes = np.linspace(0.0, 1.0 - eps, K + 1)
    eval_nodes = []
    basis = []
    
    # Chebyshev nodes on [-1, 1]
    j = np.arange(1, N + 1) # dimension N
    cheb_nodes = np.cos((2 * j - 1) / (2 * N) * np.pi)  # shape (N,)
    cheb_nodes_std = (cheb_nodes + 1) / 2  # map to [0, 1]

    for k in range(K):
        T_k, T_k1 = interval_nodes[k], interval_nodes[k + 1]
        nodes_k = T_k + (T_k1 - T_k) * cheb_nodes_std
        nodes_k = np.sort(nodes_k)
        eval_nodes.append(nodes_k)

        basis_k = []
        for j in range(N):
            y = np.zeros(N)
            y[j] = 1.0
            rbf = Rbf(nodes_k, y, function=rbf_type, epsilon=epsilon)
            basis_k.append(rbf)
        basis.append(basis_k)
        
    return interval_nodes, eval_nodes, basis

def generate_fourier_interpolatory_basis(K, N, eps, shift_ratio=0.15):
    interval_nodes = np.linspace(0.0, 1.0 - eps, K + 1)
    eval_nodes = []
    basis = []

    for k in range(K):
        T_k, T_k1 = interval_nodes[k], interval_nodes[k + 1]
        
        # Basis shifting to handle endpoints better
        assert shift_ratio < 0.5, "shift_ratio must be less than 0.5"
        eps2 = ((T_k1 - T_k) / N) * shift_ratio
        eps = (T_k1 - T_k) * ((N - 2 * shift_ratio) / (N - 1.0) - 1)

        # Shifted interval
        shifted_T_k = T_k + eps2
        shifted_T_k1 = T_k1 + eps2
        L = shifted_T_k1 - shifted_T_k + eps  # slight extension to avoid endpoint artifacts

        # Sampling points (slightly extended)
        nodes_k = np.linspace(shifted_T_k, shifted_T_k1 + eps, N, endpoint=False)
        nodes_k = np.sort(nodes_k)
        eval_nodes.append(nodes_k)  # Truncate back to (original) domain

        # DFT setup
        x = (nodes_k - shifted_T_k) / L  # scale to [0, 1)
        omega = np.exp(2j * np.pi / N)
        F = np.array([[omega ** (i * j) for j in range(N)] for i in range(N)], dtype=np.complex128)
        F_inv = np.linalg.inv(F)

        basis_k = []
        for j in range(N):
            coeffs = F_inv[:, j]

            def phi(t, coeffs=coeffs.copy(), L=L, shifted_T_k=shifted_T_k):
                t_scaled = ((t - shifted_T_k) / L) % 1  # map to [0, 1)
                vals = np.array([np.exp(2j * np.pi * n * t_scaled) for n in range(N)])
                return np.real(np.dot(coeffs, vals))

            basis_k.append(phi)

        basis.append(basis_k)

    return interval_nodes, eval_nodes, basis

def iterative_collocation(x_init, target_ode, iter_nums, interval_nodes, eval_nodes, basis, integral_points=200, dtype=np.float32, adaptive_early_stopping=False, show_iterative_progress=False):
    """
    Solve the PF-ODE dx/dt = F(x, t) using iterative collocation for batched inputs.
    """
    K = len(interval_nodes) - 1
    B, D = x_init.shape

    # Ensure x_init is correct dtype
    x_init = x_init.astype(dtype)

    # === Precompute integrals of basis functions ===
    # print("Precomputing integrals...")
    integral_matrix = []
    for k in range(K):
        N_k = len(eval_nodes[k])
        row_k = []
        for j, t_kj in enumerate(eval_nodes[k]):
            t_kj = dtype(t_kj)
            row_kj = []
            for l in range(K):
                N_l = len(eval_nodes[l])
                a = dtype(interval_nodes[l])
                b = dtype(interval_nodes[l + 1])
                s_min = max(dtype(0), a)
                s_max = min(t_kj, b)

                if s_min >= s_max:
                    row_kj.append([dtype(0.0)] * N_l)
                    continue

                s_vals = np.linspace(s_min, s_max, integral_points, dtype=dtype)
                row_kj.append([
                    dtype(simpson([basis[l][n](float(s)) for s in s_vals], s_vals))
                    for n in range(N_l)
                ])
            row_k.append(row_kj)
        integral_matrix.append(row_k)
    # ==============================================

    # === Initialize guesses ===
    x_vals = np.stack([
        np.stack([x_init.copy() for _ in eval_nodes[k]], axis=0).astype(dtype)
        for k in range(K)
    ], axis=0)
    
    # For adaptive early stopping
    if show_iterative_progress:
        if adaptive_early_stopping:
            F_vals_prev = None
            early_stopping_mask = [[False] * len(eval_nodes[k]) for k in range(K)]
            early_thresh = 1e-4
            save_folder = "iterative_progress"
            os.makedirs(f"{save_folder}/early_stopping_{str(early_thresh)}", exist_ok=True)
        else:
            early_stopping_mask = [[False] * len(eval_nodes[k]) for k in range(K)]
            early_thresh = None
            save_folder = "iterative_progress"
            os.makedirs(f"{save_folder}/no_early_stopping", exist_ok=True)
        
    nfe_count = 0
    
    if show_iterative_progress:
        num_subplots = iter_nums - 1
        n_rows = (num_subplots + 1) // 2
        fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(10, 2.5 * n_rows))
    
    for m in range(iter_nums - 1):
        if adaptive_early_stopping:
            iter_nfe_count = 0
            
            # Compute F_vals with early stopping
            F_vals = []
            for l in range(K):
                F_l = []
                for n in range(len(eval_nodes[l])):
                    if early_stopping_mask[l][n] and F_vals_prev is not None:
                        F_l.append(F_vals_prev[l][n])  # Reuse previous value
                    else:
                        val = target_ode(x_vals[l, n], dtype(eval_nodes[l][n])).astype(dtype)
                        F_l.append(val)
                        iter_nfe_count += 1
                F_vals.append(F_l)
                
            nfe_count += iter_nfe_count
        else:
            F_vals = [
                [target_ode(x_vals[l, n], dtype(eval_nodes[l][n])).astype(dtype) for n in range(len(eval_nodes[l]))]
                for l in range(K)
            ]
            nfe_count += K * len(eval_nodes[0])
        
        

        x_new = np.stack([
            np.stack([
                x_init + sum(
                    F_vals[l][n] * dtype(integral_matrix[k][j][l][n])
                    for l in range(K)
                    for n in range(len(eval_nodes[l]))
                    if integral_matrix[k][j][l][n] != 0.0
                )
                for j in range(len(eval_nodes[k]))
            ], axis=0).astype(dtype)
            for k in range(K)
        ], axis=0)

        abs_diff = np.abs(x_new - x_vals)
        # print(f"Iter {m+1}: max={np.max(abs_diff):.3e}, mean={np.mean(abs_diff):.3e}, median={np.median(abs_diff):.3e}")
        
        if show_iterative_progress:
            flat_eval_nodes = [eval_nodes[k][j] for k in range(K) for j in range(len(eval_nodes[k]))]
            flat_abs_diff = [abs_diff[k, j] for k in range(K) for j in range(len(eval_nodes[k]))]
            median_abs_diff = [np.log10(1e4 * np.median(np.max(a, axis=0)) + 1e-2) for a in flat_abs_diff]
            sorted_pairs = sorted(zip(flat_eval_nodes, median_abs_diff), key=lambda x: x[0])
            sorted_x, sorted_y = zip(*sorted_pairs)

            row = m % n_rows
            col = m // n_rows
            ax = axes[row, col]
            for x, y, is_fixed in zip(flat_eval_nodes, median_abs_diff,
                                    [early_stopping_mask[k][j] for k in range(K) for j in range(len(eval_nodes[k]))]):
                ax.plot(x, y, marker='o', color='#BD72E6' if is_fixed else '#7373e6')

            ax.set_xlim(0, 1)
            ax.set_ylim(-2.2, 4.2)
            ax.set_title(f"Iter {m + 1}")
        
        if adaptive_early_stopping:
            # Update early stopping mask
            median_abs_diff = np.median(np.max(abs_diff, axis=2), axis=2)
            early_stopping_mask = median_abs_diff < early_thresh
            if np.all(early_stopping_mask):
                print(f"Early stopping at iteration {m + 1}")
                break
            F_vals_prev = F_vals  # Save current F_vals for next iteration
            
        x_vals = x_new.astype(dtype)
        
    if show_iterative_progress:
        plt.tight_layout()
        if adaptive_early_stopping:
            plt.savefig(f"{save_folder}/early_stopping_{str(early_thresh)}/iterative_progress_grid.pdf", format='pdf')
        else:
            plt.savefig(f"{save_folder}/no_early_stopping/iterative_progress_grid.pdf", format='pdf')
        plt.close(fig)

    F_vals = [
        [target_ode(x_vals[l, n], dtype(eval_nodes[l][n])).astype(dtype) for n in range(len(eval_nodes[l]))]
        for l in range(K)
    ]
    nfe_count += K * len(eval_nodes[0])

    # === Prepare return data ===
    Fy_basis_data = []
    for k in range(K):
        for j, _ in enumerate(eval_nodes[k]):
            Fy_basis_data.append({
                "F": F_vals[k][j],
                "phi": basis[k][j],
                "interval_start": dtype(interval_nodes[k]),
                "interval_end": dtype(interval_nodes[k + 1])
            })

    data = {
        "x_init": x_init,
        "Fy_basis_data": Fy_basis_data,
        "integral_points": integral_points,
        "dtype": dtype
    }

    # === Return function to evaluate x(t) ===
    def x_fn(t, data):
        B, D = data["x_init"].shape
        total = data["x_init"].copy()

        for item in data["Fy_basis_data"]:
            a, b = item["interval_start"], item["interval_end"]
            if t < a:
                continue

            s_min = max(a, dtype(0))
            s_max = min(t, b)
            if s_min >= s_max:
                continue

            s_vals = np.linspace(s_min, s_max, data["integral_points"], dtype=dtype)
            integrand_vals = np.array([item["phi"](float(s)) for s in s_vals], dtype=dtype)
            integral_val = dtype(simpson(integrand_vals, s_vals))
            total += item["F"] * integral_val

        return total.astype(dtype)

    return x_fn, data, nfe_count


if __name__ == "__main__":
    cases = [
        (2, 6, 1e-5, "dense"),
        (4, 3, 1e-3, "sparse")
    ]

    for K, N, eps, _ in cases:
        # Collect all bases and labels for plotting
        interval_nodes_list = []
        eval_nodes_list = []
        basis_list = []
        labels = []

        # Lagrange (Chebyshev) basis
        int_nodes, eval_nodes, basis = generate_piecewise_chebyshev_basis(K, N, eps)
        interval_nodes_list.append(int_nodes)
        eval_nodes_list.append(eval_nodes)
        basis_list.append(basis)
        labels.append(f"Lagrange Basis")

        # Fourier basis
        int_nodes, eval_nodes, basis = generate_fourier_interpolatory_basis(K, N, eps)
        interval_nodes_list.append(int_nodes)
        eval_nodes_list.append(eval_nodes)
        basis_list.append(basis)
        labels.append(f"Fourier Basis")

        # RBF bases
        for rbf_type in ["Linear", "Cubic", "Quintic"]:
            int_nodes, eval_nodes, basis = generate_rbf_basis(K, N, eps, rbf_type)
            interval_nodes_list.append(int_nodes)
            eval_nodes_list.append(eval_nodes)
            basis_list.append(basis)
            labels.append(f"{rbf_type.capitalize()} RBF Basis")

        # Plot all bases for this configuration
        plot_all_piecewise_bases(interval_nodes_list, eval_nodes_list, basis_list, labels)

