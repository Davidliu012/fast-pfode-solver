## Synthetic Experiments

This directory provides the code for running synthetic experiments used to analyze different basis function constructions under our PF-ODE solver framework.

---

### 🚀 Results Reproduction

To reproduce the experimental trends presented in our paper, simply run:

```bash
python main.py
```

> ⚠️ **Note**: This configuration is a slightly modified version of what was used in the paper. However, **the overall trends and conclusions remain unchanged**.

---

### 🎨 Basis Function Visualization

To visualize and compare different basis functions (Chebyshev, RBF, Fourier), run:

```bash
python all_collocation.py
```

This script will generate plots to help you intuitively understand the properties of each basis used in the collocation solver.