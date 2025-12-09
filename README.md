# Diffusion Regularised Wasserstein Gradient Flow

> This repository is based on the implementation of [Repulsive Score Distillation](https://github.com/nzilberstein/Repulsive-score-distillation-RSD-.git). We thank the authors for open-sourcing their code.


This is the implementation of the NeurIPS FPI workshop paper *A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models* https://arxiv.org/abs/2509.19276.

## Installation
- Make sure to clone the submodules after cloning the repository:
```bash
git submodule update --init --recursive
```

- We use `uv` to manage the environment. Create the environment with:
```bash
uv venv dwgf --python=3.11
source dwgf/bin/activate
uv pip install -r requirements.txt
```

## File Structure
- The entry point is `main.py`. Scripts for evaluation are provided in `scripts/`.
- `configs` contains the configuration files for different experiments. The main experiment config is `ffhq_stable_diffusion.yaml`.
- `src/algos` contains the implementation of the different algorithms. Our algorithm is implemented in `dwgf.py`.


