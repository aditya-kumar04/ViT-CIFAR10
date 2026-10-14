# Vision Transformer (ViT) on CIFAR-10

A compact, from-scratch PyTorch implementation of a Vision Transformer (ViT), evaluated on CIFAR-10 and compared with a CIFAR-adapted ResNet-18 baseline.

> **Scope:** This is a reduced-scale educational reproduction inspired by *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. It is not a reproduction of the original paper's large-scale ImageNet/JFT experiments.

## Results

Results below are from the local notebook run. The primary models were trained for 15 epochs with batch size 64, AdamW (learning rate `3e-4`, weight decay `0.05`), and cosine annealing.

| Experiment | Trainable parameters | Best validation accuracy | Test accuracy |
|---|---:|---:|---:|
| ViT, dropout 0.1, full training partition | 1,205,898 | 71.94% | 72.00% |
| ResNet-18 from scratch | 11,173,962 | 90.34% | 89.41% |
| ViT, 20% training data (9,000 images) | 1,205,898 | 51.22% | 51.11% |
| ViT, dropout 0.0 | 1,205,898 | 74.42% | 75.02% |

### Interpretation and caveats

- ResNet-18 exceeds the baseline ViT by **17.41 percentage points** in test accuracy for these specific configurations and this training budget.
- Training the ViT on 20% of the training partition lowers test accuracy by **20.89 percentage points** relative to the full-data baseline.
- The no-dropout run is **3.02 percentage points** higher than the baseline ViT, but this is a single-run comparison with different seeds. Treat it as exploratory, not a reliable estimate of the causal effect of dropout.
- The baseline ViT's best validation accuracy occurred at epoch 15, so the training budget may not establish convergence.
- Attention rollout is qualitative and does not prove causal feature importance.

Do not report these metrics as independently reproduced until you have rerun the notebook yourself.

## Model details

### Vision Transformer

- Input: RGB CIFAR-10 images, `32 × 32`
- Patch size: `4 × 4`
- Number of patches: `64`
- Tokens: `64` patch tokens + one CLS token = `65`
- Embedding dimension: `128`
- Transformer depth: `6`
- Attention heads: `4`
- MLP hidden dimension: `512`
- Dropout baseline: `0.1`
- Trainable parameters: `1,205,898`

### ResNet-18 baseline

ResNet-18 is trained from scratch. Its initial convolution is adapted to a `3 × 3`, stride-1 convolution, the initial max-pooling is removed, and the final layer outputs 10 classes.

## Project structure

```text
.
├── ViT_CIFAR10_Local_Resumable.ipynb
├── requirements_vit_local.txt
├── README.md
├── .gitignore
├── reports/
│   ├── ViT_CIFAR10_IEEE_Report.tex
│   └── ViT_CIFAR10_IEEE_Report_Draft.pdf
└── vit_cifar10_artifacts/        # generated at runtime; do not commit large checkpoints
    ├── checkpoints/
    ├── data/
    └── results/
```

## Setup: local Jupyter / Antigravity

Python 3.10–3.12 is recommended for broad PyTorch package compatibility. Check the current official PyTorch installation instructions for your OS and hardware if installation fails.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_vit_local.txt
python -m ipykernel install --user --name vit-cifar10 --display-name "Python (ViT CIFAR-10)"
```

Open `ViT_CIFAR10_Local_Resumable.ipynb` in Jupyter or Antigravity and select the `Python (ViT CIFAR-10)` kernel. Run cells from top to bottom.

The notebook automatically selects CUDA, Apple Silicon MPS, or CPU. The first run downloads CIFAR-10 into `vit_cifar10_artifacts/data/`.

## Checkpoint recovery

The notebook saves a latest checkpoint after each completed epoch, a best-validation checkpoint, per-epoch CSV history, and experiment metadata under `vit_cifar10_artifacts/`.

If a kernel crashes:
1. Keep the entire `vit_cifar10_artifacts/` directory.
2. Reopen the notebook and execute cells from the top.
3. Keep `RESUME_FROM_CHECKPOINT = True`.
4. Resume with the same experiment name, epoch budget, model configuration, data split, and training settings.

Back up the notebook and `vit_cifar10_artifacts/` folder together. Do not commit raw checkpoints or CIFAR-10 files to GitHub unless you intentionally want a large repository.

## Reproducibility

- Global seed: `42` for the primary configuration.
- Train/validation split: 45,000 / 5,000 from the 50,000-image training partition.
- Test set: official 10,000-image CIFAR-10 test partition.
- Training augmentation: random crop with padding 4 and random horizontal flip.
- Evaluation: deterministic normalization.
- Checkpoint selection: highest validation accuracy.
- Test metrics: calculated for the selected checkpoint.

For robust conclusions, run the optional multi-seed dropout experiment and report mean ± standard deviation. Keep validation for model selection and reserve the test set for final evaluation.

## Report

The IEEE-style report draft describes the methodology, equations, results, limitations, and references. Review all metadata and verify the final formatting and required page count before submission.

## References

1. A. Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” ICLR, 2021.
2. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” CVPR, 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.
3. A. Krizhevsky, “Learning Multiple Layers of Features from Tiny Images,” University of Toronto technical report, 2009.

## License

Choose and add a license before making the repository public. Do not claim a license for third-party assets or code unless its terms permit it.
