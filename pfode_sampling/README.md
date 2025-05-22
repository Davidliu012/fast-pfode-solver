# PF-ODE Sampling Experiments (CIFAR-10)

This folder contains the code to reproduce the PF-ODE sampling experiments on the CIFAR-10 dataset as described in our NeurIPS 2025 paper.

---

## 🔄 Base Code Acknowledgement

This implementation is **built upon** the official [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) repository by Yang Song et al., which implements the original Score-SDE and NCSN++ models.

We thank the authors for releasing their code and model checkpoints.  
Modifications were made to adapt the codebase for our PF-ODE sampling framework and custom ODE solvers.

---

## ⚙️ Key Modifications

Compared to the original repo, our key changes include:

- ✅ Added support for **probability flow ODE solvers**, including:
  - Euler
  - Heun
  - Collocation-based solver (our contribution)
- ✅ Removed unnecessary training code for focused sampling-only evaluation

---

## 📦 How to Run

### 🔑 Step 1: Download Pretrained Checkpoint

To reproduce our results, download the pretrained model checkpoint:

* [`subvp/cifar10_ddpmpp_deep_continuous/checkpoint_18.pth`](https://drive.google.com/drive/folders/16QGkviGcizSbIPRk37-YksUhlNIna4Ys)
  (from the official [score\_sde\_pytorch](https://github.com/yang-song/score_sde_pytorch) repo)

You can do this automatically using our helper script:

```bash
bash download_ckpt.sh
```

---

### 🗃️ Step 2: Download CIFAR-10 Dataset

Download and prepare the CIFAR-10 dataset using:

```bash
python download_cifar10.py
```

This will place the dataset in the correct directory for evaluation and sampling.

---

### 🛠️ Step 3: Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

---

### 🚀 Step 4: Run Collocation-Based Sampling

Launch a sampling experiment using our PF-ODE solver:

```bash
python main.py
```

---


### 📊 Evaluation (CMMD)

We evaluate the quality of generated samples using **CMMD** (CLIP-MMD), a perceptual metric that measures similarity between image distributions using CLIP embeddings and Maximum Mean Discrepancy (MMD) with an RBF kernel.

> 📖 CMMD was introduced in [Rethinking FID: Towards a Better Evaluation Metric for Image Generation](https://arxiv.org/pdf/2401.09603) (Google Research, 2024).

#### Setup

Please directly head to the 🔗 [Official implementation](https://github.com/google-research/google-research/tree/master/cmmd) for the correct setup.

#### Compute CMMD

We use the following script to evaluate the generated images:

```bash
python -m cmmd.main /path/to/reference/images /path/to/eval/images --batch_size=32 --max_count=5000
```

#### Notes

* We use the default CLIP model for CMMD computation.

---

## 📜 License

Please refer to the original [score\_sde\_pytorch LICENSE](https://github.com/yang-song/score_sde_pytorch/blob/main/LICENSE) for licensing terms of the base code.
Our additions are released under the same license (MIT), unless otherwise stated.

---

## 🙏 Acknowledgements

* [score\_sde\_pytorch](https://github.com/yang-song/score_sde_pytorch) by Yang Song et al.
* [CMMD](https://github.com/google-research/google-research/tree/master/cmmd) for evaluation

