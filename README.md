# TACoS

**Tree-based Asymmetric Contrast Segmentation (TACoS)** — a single-stage scribble-supervised semantic segmentation framework for two-dimensional (2D) material flakes in optical microscopy images.

This repository provides the official PyTorch Lightning implementation described in:

> **TACoS: Weakly Supervised Learning of Two-Dimensional Materials from Scribble Annotations to Precise Segmentation**  
> Jiabei Chen, Liping Zhang, Jiang-Bin Wu, Zhongming Wei, Enhao Ning, Su Yan, Weijun Li, Ping-Heng Tan, Xin Ning  
> arXiv, 2026

Using less than **0.6%** labeled pixels (scribble supervision), TACoS reaches **>96%** of fully supervised performance on graphene and MoS₂, with strong boundary stability under low-contrast edges and complex backgrounds.

---

## Highlights

- **Single-stage end-to-end training** — no offline pseudo-label generation or iterative retraining.
- **Three coupled modules:**
  - **UWSD** — unlabeled weak–strong distribution alignment on unlabeled pixels.
  - **TER** — tree energy regularization via dual minimum spanning trees (low/high DINOv2 features).
  - **ARCL** — asymmetric regional contrast learning (`L_Region` + `L_Boundary`).
- **Backbone & head:** DINOv2-S + DPT decoder (518×518 crops, slide inference at validation).
- **Datasets:** Yan (graphene, MoS₂) and Uslu (graphene, WSe₂) scribble benchmarks.

---

## Requirements

- Linux (recommended; CUDA extension build tested on Linux)
- Python ≥ 3.10
- CUDA-capable GPU (paper: 2× RTX 3090 24 GB; single-GPU training supported)
- PyTorch ≥ 2.5 with CUDA
- [mmcv](https://github.com/open-mmlab/mmcv) (for data transforms)

Core Python packages:

```
lightning
torchmetrics
numpy pandas pyyaml
opencv-python pillow
tensorboard tqdm matplotlib
```

Install PyTorch first, then other dependencies. A full pinned environment is in `requirements.txt` (note: regenerate or trim it for your platform before sharing).

---

## Installation

```bash
git clone https://github.com/chenjiabei0403/TACoS.git
cd TACoS

# 1) Python dependencies (after installing PyTorch + mmcv for your CUDA version)
pip install -r requirements.txt

# 2) CUDA extension for tree filtering (TER module)
cd model/functions/src
pip install -e .
cd ../../..
```

> **Note:** `model/functions/src/setup.py` requires a CUDA device and `LD_LIBRARY_PATH` pointing to your CUDA libraries when building.

---

## Pretrained backbone weights

Place DINOv2 LVD142M weights under `pretrained/` (not tracked by git):

| File | Source |
|------|--------|
| `pretrained/dinov2_small.pth` | [dinov2_vits14_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) |
| `pretrained/dinov2_base.pth` | [dinov2_vitb14_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) |
| `pretrained/dinov2_large.pth` | [dinov2_vitl14_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) |

Default configs use `backbone: dinov2_small`. DINOv2 is subject to [Meta’s license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).

---

## Dataset preparation

Configs expect scribble data **outside** the repo (see `.gitignore`). Paths in YAML are relative to the project root, e.g. `../scribbles/yan/graphene/...`.

### Data availability

This repository ships **training code and configs only**. Images and scribble annotations are **not** included.

As described in the TACoS paper (§4.1), scribbles follow the [Scribbles for All](https://arxiv.org/abs/2408.12489) paradigm. To try the code now, you can:

1. Obtain original microscopic images and dense masks from [Yan et al.](https://doi.org/10.1038/s41699-025-00720-3) and [Uslu et al.](https://doi.org/10.1088/2632-2153/ad4e8a) (subject to their licenses).
2. Generate scribbles with the public [Scribbles for All](https://github.com/wbkit/Scribbles4All) pipeline.
3. Arrange files into the directory layout below and point the YAML paths accordingly.

The exact scribble splits used in our TACoS experiments were produced by a microscopy-adapted **SCRIBE** protocol and will be released **only after** the companion **SCRIBE-Bench** paper is accepted (together with download links / archival DOI). Until then we do not distribute those SCRIBE-generated scribbles. This README will be updated when they become public.

### Directory layout

```
scribbles/
├── yan/
│   ├── graphene/
│   │   ├── images/train/          # {id}.jpg
│   │   ├── images/val/
│   │   ├── masks/train/           # {id}_scribble.png  (0=BG, 1=FG, 255=unknown)
│   │   ├── masks/val/             # {id}.png  (dense GT for evaluation)
│   │   └── masks_full/train/      # {id}.png  (dense GT for L_Boundary only)
│   └── MoS2/
│       └── ...
└── uslu/
    ├── graphene/
    └── WSe2/
```

Training uses sparse scribble masks; dense full masks in `masks_full/train/` are used only for the boundary branch of ARCL.

### Provided configs

| Config | Material | Source |
|--------|----------|--------|
| `configs/graphene_yan.yaml` | graphene | Yan et al. |
| `configs/MoS2_yan.yaml` | MoS₂ | Yan et al. |
| `configs/graphene_uslu.yaml` | graphene | Uslu et al. |
| `configs/WSe2_uslu.yaml` | WSe₂ | Uslu et al. |

Update `train_root`, `val_root`, and related paths in the YAML if your data lives elsewhere.

---

## Training

```bash
# Single GPU
python train.py --config configs/graphene_yan.yaml --gpus 1 --seed 42

# Multi-GPU (DDP)
python train.py --config configs/MoS2_yan.yaml --gpus 2

# Resume from checkpoint
python train.py --config configs/graphene_yan.yaml --gpus 1 --ckpt_path path/to/last.ckpt
```

**Training settings (paper §4.2):**

- Optimizer: AdamW, lr = 5×10⁻⁶, head lr multiplier = 40×
- LR schedule: poly decay (power 0.9)
- Epochs: 50, batch size: 2 per GPU
- Crop: 518×518, weak/strong augmentations (UniMatch V2 style)
- Validation: slide window 518×518, stride 448×448
- Metrics: mIoU + Boundary IoU (2% diagonal dilation)
- AMP (fp16) enabled

Checkpoints and logs are written to `save_root` in the config (best checkpoint monitored on `val/mIoU`). Per-epoch metrics are saved to `results.csv`.

---

## Visualization

Evaluate a checkpoint and save prediction triplets (image / GT / pred / error):

```bash
python visualize_best_epoch.py \
  --config configs/graphene_yan.yaml \
  --ckpt_path path/to/best.ckpt \
  --save_root ./vis_output \
  --num_samples 30
```

Options:

- `--visualize_all` — run on the full validation set
- `--sample_ids_file ids.json` — visualize a fixed list of image IDs
- `--samples N` — cap the number of saved figures

---

## Project structure

```
TACoS/
├── train.py                 # Training & validation entry point
├── visualize_best_epoch.py  # Checkpoint visualization
├── dataset.py               # Weak/strong view data loader
├── configs/                 # Per-dataset YAML configs
├── model/
│   ├── backbone/dinov2.py   # DINOv2 encoder
│   ├── semseg/dpt.py        # DPT decoder
│   ├── tree_energy.py       # TER (MST + tree filter)
│   ├── reco_branch.py       # ARCL representation branch
│   └── functions/           # CUDA ops (MST, BFS, refine)
└── util/
    ├── boundary.py          # Boundary IoU metric
    ├── inference.py         # Slide inference
    ├── ohem.py              # OHEM for L_sup
    └── reco_module_list.py  # L_Region (ReCo-style loss)
```

---

## Loss overview

Total objective (paper Eq. 18):

```
L = L_sup + λ_c · L_UWSD + λ_t · L_TER + λ_r · L_Region + λ_b · L_Boundary
```

| Term | Config block | Description |
|------|--------------|-------------|
| `L_sup` | `sup` | Scribble CE + OHEM on labeled pixels |
| `L_UWSD` | `uwsd` | Cosine consistency on unlabeled pixels (λ_c = 5) |
| `L_TER` | `ter` | Tree-filtered soft target on unlabeled pixels (λ_t = 1, σ_aff = 0.1) |
| `L_Region` | `region` | Regional contrast on hard pixels (τ_w=0.7, τ_s=0.97) |
| `L_Boundary` | `boundary` | Asymmetric boundary penalty using dense train masks |

Material-specific λ_r / λ_b are set in each YAML (see paper Fig. 7).

---

## Citation

If you use this code or method, please cite:

```bibtex
@article{chen2026tacos,
  title   = {{TACoS}: Weakly Supervised Learning of Two-Dimensional Materials from Scribble Annotations to Precise Segmentation},
  author  = {Chen, Jiabei and Zhang, Liping and Wu, Jiang-Bin and Wei, Zhongming and Ning, Enhao and Yan, Su and Li, Weijun and Tan, Ping-Heng and Ning, Xin},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## Acknowledgments

This repository implements TACoS as described in the paper. Module-level inspirations follow §2.3:

### Network backbone & decoder

- [DINOv2](https://github.com/facebookresearch/dinov2) (Oquab et al., 2023) — ViT encoder
- [DPT](https://github.com/isl-org/DPT) (Ranftl et al., ICCV 2021) — dense prediction head

### UWSD (Unlabeled Weak–Strong Distribution Alignment)

- [FixMatch](https://arxiv.org/abs/2001.07685) (Sohn et al., NeurIPS 2020) — weak/strong consistency on unlabeled pixels
- [UniMatch V2](https://arxiv.org/abs/2410.10777) (Yang et al., TPAMI 2025) — weak/strong augmentation pipeline (§4.2)

### TER (Tree Energy Regularization)

- [Tree Energy Loss](https://github.com/megvii-research/TreeEnergyLoss) (Liang et al., CVPR 2022) — dual-tree MST and tree filtering
- [TreeFilter-Torch](https://github.com/Megvii-BaseDetection/TreeFilter-Torch) (Song et al., NeurIPS 2019) — CUDA tree-filter operators in `model/functions/`

### ARCL (Asymmetric Regional Contrast Learning)

- [ReCo](https://arxiv.org/abs/2104.04465) (Liu et al., 2021) — regional contrast and hard-pixel sampling (`L_Region`)
- [GCBL](https://doi.org/10.1016/j.patcog.2024.110723) (Qiu et al., Pattern Recognition 2024) — confidence-guided boundary contrast (`L_Boundary`)

### Supervision, metrics & data

- [OHEM](https://arxiv.org/abs/1604.03540) (Shrivastava et al., CVPR 2016) — hard example mining for `L_sup`
- [Boundary IoU](https://arxiv.org/abs/2101.07504) (Cheng et al., CVPR 2021) — boundary evaluation metric
- [Scribbles for All](https://arxiv.org/abs/2408.12489) (Boettcher et al., NeurIPS 2024) — automated scribble generation (§4.1)
- [ScribbleSup](https://arxiv.org/abs/1606.09358) (Lin et al., CVPR 2016) — pioneering scribble-supervised segmentation

### Dataset credits

Microscopic images and dense masks from [Yan et al.](https://doi.org/10.1038/s41699-025-00720-3) and [Uslu et al.](https://doi.org/10.1088/2632-2153/ad4e8a). Please cite the original dataset papers when using the data.

### Funding

This work was supported by the National Key Research and Development Program of China (Grant No. 2024YFA1409700) and the CAS Project for Young Scientists in Basic Research (Grant No. YSBR-090).

---

## License

Code in this repository is released under the MIT License (see `LICENSE`). Third-party components ([DINOv2](https://github.com/facebookresearch/dinov2/blob/main/LICENSE), [mmcv](https://github.com/open-mmlab/mmcv), [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), etc.) are subject to their respective licenses.
