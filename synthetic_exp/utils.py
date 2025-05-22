import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ------------------------ Utility Functions --------------------------
def estimate_kl_divergence(p_samples, q_samples, bw_method='scott'):
    kde_p = gaussian_kde(p_samples.T, bw_method=bw_method)
    kde_q = gaussian_kde(q_samples.T, bw_method=bw_method)
    p_density = np.clip(kde_p(p_samples.T), 1e-10, None)
    q_density = np.clip(kde_q(p_samples.T), 1e-10, None)
    return max(np.mean(np.log(p_density / q_density)), 0.0)

def runge_kutta_sampler(x_init, n_steps, score_fn, score_estimator):
    dt = 1.0 / n_steps
    x = x_init.copy()
    for step in range(n_steps):
        t = (step + 1) / n_steps
        k1 = -0.5 * getattr(score_estimator, score_fn)(x, t)
        k2 = -0.5 * getattr(score_estimator, score_fn)(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = -0.5 * getattr(score_estimator, score_fn)(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = -0.5 * getattr(score_estimator, score_fn)(x + dt * k3, t + dt)
        x += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


# ------------------------ Visualization Functions --------------------------
def plot_all_piecewise_bases(interval_nodes_list, eval_nodes_list, basis_list, titles, output_dir="basis_plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    K = len(interval_nodes_list[0]) - 1
    N = len(eval_nodes_list[0][0])
    num_rows = len(titles)

    fig, axs = plt.subplots(num_rows, K, figsize=(16, 3 * num_rows), sharey='row')

    if num_rows == 1:
        axs = np.expand_dims(axs, axis=0)

    for row in range(num_rows):
        interval_nodes = interval_nodes_list[row]
        eval_nodes = eval_nodes_list[row]
        basis = basis_list[row]
        title = titles[row]

        for k in range(K):
            T_k, T_k1 = interval_nodes[k], interval_nodes[k + 1]
            t_dense = np.linspace(T_k, T_k1, 200)

            for j in range(N):
                y_dense = [basis[k][j](t) for t in t_dense]
                axs[row, k].plot(t_dense, y_dense, label=f"$\\varphi_{j}^{k}$")

            axs[row, k].scatter(eval_nodes[k], [1.0] * N, color='black', s=10, zorder=5)
            axs[row, k].grid(True)
            axs[row, k].set_xlabel("t", fontsize=16)
                
            if row == 0:
                axs[row, k].set_title(f"Interval {k}", fontsize=20)

            axs[row, k].tick_params(axis='both', which='major', labelsize=16)
            axs[row, k].tick_params(axis='both', which='minor', labelsize=16)

        axs[row, 0].annotate(
            title,
            xy=(0, 0.5),
            xytext=(-axs[row, 0].yaxis.labelpad - 10, 0),
            xycoords=axs[row, 0].yaxis.label,
            textcoords='offset points',
            size=18,
            ha='right',
            va='center',
            rotation=90
        )
        axs[row, 0].yaxis.set_label_coords(-0.06, 0.5)
        

    plt.tight_layout()
    plt.savefig(f"{output_dir}/piecewise_bases_{K}_{N}.pdf", format='pdf')