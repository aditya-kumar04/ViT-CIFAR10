# Vision Transformer (ViT) from scratch on CIFAR-10

Reduced-scale reproduction of **Dosovitskiy et al., _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale_ (ICLR 2021)** for the Deep Learning 2026 paper-implementation project (24CS3006, DL-065).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aditya-kumar04/ViT-CIFAR10/blob/main/ViT_CIFAR10_Colab.ipynb)

**What is reproduced.** The paper's central claim: a ViT has no convolutional inductive bias, so it trails comparable ResNets when data are limited and catches up only with large-scale pre-training. This repo tests that claim in miniature on a free Colab T4:

| # | Experiment | Paper reference |
|---|---|---|
| 1 | ViT vs ResNet-18, **from scratch**, on 10 / 25 / 50 / 100 % of CIFAR-10 | §4.2, Fig. 3-4 (data scale vs. inductive bias) |
| 2 | **Pre-trained** ViT-B/16 vs ResNet-50, fine-tuned on CIFAR-10 | Table 5 (CIFAR-10: 98.13 / 98.95 / 99.00 % for ImageNet-1k / ImageNet-21k / JFT-300M pre-training) |
| 3 | **Ablation:** position embedding none / 1-D learned / 2-D sin-cos | Appendix D.4 |
| 4 | Attention rollout grid, confusion matrix, failure cases | Fig. 6 (attention from the output token to the input) |

> This is **not** a reproduction of the paper's ImageNet/JFT-scale training. Differences (model size, schedule, checkpoints) are listed below and in the report.

## Results

Numbers are produced by the notebook and pasted below by `python analysis.py --update-readme README.md`. Until you have run it, the block is empty. Do not fill it by hand.

<!-- RESULTS:START -->
_Run the notebook, then `python analysis.py --out-dir . --update-readme README.md`._
<!-- RESULTS:END -->

Figures: `figures/` (data-fraction sweep, training curves, position-embedding ablation, rollout grid, confusion matrices). Raw per-run results (JSON with full history, per-epoch CSV, test predictions): `results/`.

### Earlier version (v1, kept for honesty)

The first iteration (`legacy/`, 15 epochs, crop + flip only, dropout 0.1, 1.2 M-parameter ViT, local Jupyter) reached 72.00 % (ViT) vs 89.41 % (ResNet-18) test accuracy, 51.11 % for the ViT on 20 % of the data, and 75.02 % with dropout 0 (single runs with different seeds, so that last difference is not reliable). The new recipe replaces it.

## Quick start

**Colab (recommended):** click the badge → Runtime ▸ T4 GPU → set `PRESET` in the second cell → Run all.

| `PRESET` | purpose | rough T4 time (estimate, measure your first epoch) |
|---|---|---|
| `dev` | tiny fake-data run that checks the whole pipeline | minutes |
| `quick` | 30 epochs @100 %, 2 fine-tune epochs | ~1.5-2 h in total |
| `full` | 100 epochs @100 %, 3 fine-tune epochs | ~4-5 h in total |

Finished runs are cached and interrupted runs resume from the last epoch; with `USE_DRIVE = True` results survive a Colab disconnect.

**Local:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/smoke_test.py            # ~1-2 min on CPU, no downloads
jupyter lab ViT_CIFAR10_Colab.ipynb   # needs a GPU for quick/full
```

**After the runs**
```bash
python analysis.py --out-dir . --report-dir reports/generated --update-readme README.md
cd reports && pdflatex ViT_CIFAR10_IEEE_Report.tex && pdflatex ViT_CIFAR10_IEEE_Report.tex
```
The report reads `reports/generated/*.tex` and `figures/*.png` automatically. Red text in the PDF marks sentences that depend on your numbers; check them, then set `\draftfalse`.

## What was written by hand vs. taken from libraries

| By hand (`vit_lib.py`) | From libraries |
|---|---|
| patch embedding, CLS token, 1-D learned / 2-D sin-cos / no position embedding | `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, `nn.GELU`, `nn.Dropout` |
| multi-head self-attention (explicit softmax(QKᵀ/√d)V, returns attention maps), pre-LN encoder, MLP head | AdamW / SGD, `torchvision` datasets, transforms (incl. RandAugment), `torchvision` ResNet-18 skeleton |
| mixup / CutMix, warm-up + cosine schedule, resumable training loop, attention rollout | `timm` **pre-trained** ViT-B/16 and ResNet-50 for Experiment 2 only |

Unit tests (`tests/smoke_test.py`) check, among other things, that the patch-embedding `Conv2d` equals flatten + `Linear`, that the hand-written attention matches `scaled_dot_product_attention`, that attention rows sum to 1, that rollout rows stay normalised, that data-fraction subsets are nested and the split disjoint, and that a crashed run resumes from its last checkpoint.

## Setup used and differences from the paper

- ViT-Tiny-sized model (patch 4, 64 tokens + CLS, D = 192, depth 8, 6 heads, MLP 768) on 32×32 images; the paper's smallest model (ViT-B/16) has 86 M parameters.
- 30-100 epochs from scratch (the paper pre-trains for hundreds of epochs on far larger data).
- Both from-scratch models share augmentation (crop, flip, RandAugment, mixup/CutMix, label smoothing 0.1), schedule shape and number of updates; optimisers differ (AdamW for ViT, Nesterov SGD for ResNet) because each uses its standard recipe.
- Smaller training subsets are trained for proportionally more epochs (`EQUAL_STEPS = True`) so that data quantity, not update count, is what changes.
- Experiment 2 uses `timm` checkpoints and a short fine-tuning schedule, not the paper's checkpoints.
- Seeds: `SEEDS` / `ABL_SEEDS` in the notebook (default one seed; use three for mean ± std).

## Reproducibility

Seed 42; fixed 45 000 / 5 000 train / validation split (from the 50 000 training images); the official 10 000 test images are evaluated once, on the best-validation checkpoint. No hyper-parameters are tuned on the test set. `results/*.json` stores the full config, history, GPU and library versions for every run.

## Repository layout

```
vit_lib.py                 model, data, training loop, rollout (from scratch)
analysis.py                summaries, plots, LaTeX tables/macros, README update
ViT_CIFAR10_Colab.ipynb    runs every experiment (Colab T4)
tests/smoke_test.py        CPU tests
reports/                   IEEE report (.tex), generated/ tables, v1_draft/
results/  figures/         committed outputs of the runs
legacy/                    v1 local notebook (15-epoch baseline)
```

## References and credits

1. A. Dosovitskiy et al., "An Image is Worth 16x16 Words," ICLR 2021 (arXiv:2010.11929). Only the paper was consulted; no ViT repository code was copied.
2. S. Abnar, W. Zuidema, "Quantifying Attention Flow in Transformers," ACL 2020 (attention rollout).
3. K. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.
4. A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," 2009.
5. R. Wightman, `timm` (PyTorch Image Models), used for pre-trained backbones in Experiment 2.

## License

MIT, see `LICENSE`.
