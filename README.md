Here's an improved version of your `README.md` that clarifies scope, enhances professionalism, and specifies where reproduction instructions can be found:

---

# ⚡ FAST PF-ODE Sampling
**Official Implementation of “On Provably Fast Consistency Model ODE Solvers” (NeurIPS 2025)**  

---

## 🗂️ Repository Overview

This repository provides the official implementation of our NeurIPS 2025 paper. It includes:

- [x] A modular implementation of our proposed PF-ODE solver (Algorithm 1)
- [x] Synthetic experiments with different basis function constructions (Chebyshev, RBF, Fourier)
- [x] Real-data experiments using CIFAR-10
- [x] Support for multiple ODE solvers, including Euler, Heun, and our proposed collocation-based method

> 📌 Detailed reproduction instructions for each experiment type are provided in the individual `README.md` files inside the `synthetic_exp/` and `pfode_sampling/` directories.

---

## 📁 Project Structure

```bash
.
├── README.md                # Main documentation
├── requirements.txt         # Python dependencies
├── synthetic_exp/           # Synthetic experiments with different basis functions
│   └── README.md            # Reproduction details for synthetic experiments
├── pfode_sampling/          # Experiments on real datasets (e.g., CIFAR-10)
│   └── README.md            # Reproduction details for PF-ODE sampling
```

---

## 🔧 Installation

We recommend using a virtual environment:

```bash
conda create -n fast-sampling python=3.11
conda activate fast-sampling
pip install -r requirements.txt
```

---

## 🖥️ Experimental Environment

All experiments were conducted on a workstation with the following configuration:

* **GPU**: NVIDIA RTX 5080 (16 GB VRAM)
* **CPU**: Intel(R) Core(TM) Ultra 7 265KF
* **RAM**: 128 GB DDR4
* **OS**: Ubuntu 24.04.2 LTS
* **Python**: 3.11
* **CUDA**: 12.8
* **cuDNN**: 9.8
* **PyTorch**: 2.8.0
