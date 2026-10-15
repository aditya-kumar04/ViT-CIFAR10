"""analysis.py -- turn finished runs into tables, figures and LaTeX report assets.

CLI (run from the repo root after the experiments):
    python analysis.py --out-dir . --report-dir reports/generated
reads  <out-dir>/results/*.json   (written by vit_lib.fit)
writes <out-dir>/results/summary.csv, results_table.md
       <report-dir>/results_macros.tex, tab_sweep.tex, tab_ablation.tex, tab_finetune.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from vit_lib import CLASSES, denormalize, rollout_maps

FRAC_WORD = {0.10: "Ten", 0.25: "TwentyFive", 0.50: "Fifty", 1.00: "Full"}

# Numbers reported by the ViT paper (Dosovitskiy et al., ICLR 2021), Table 5:
# top-1 accuracy (%) on CIFAR-10 after fine-tuning ViT-B/16 pre-trained on each dataset.
PAPER_VITB16_C10 = {"ImageNet-1k": 98.13, "ImageNet-21k": 98.95, "JFT-300M": 99.00}


# --------------------------------------------------------------------------- loading
def load_summary(out_dir) -> pd.DataFrame:
    rows = []
    for p in sorted((Path(out_dir) / "results").glob("*.json")):
        r = json.loads(p.read_text())
        if "config" not in r or "test_acc" not in r:      # e.g. failure_cases.json
            continue
        m = r.get("meta", {})
        rows.append(dict(name=r["name"], group=m.get("group", ""), arch=m.get("arch", ""),
                         variant=m.get("variant", ""), fraction=m.get("fraction", 1.0),
                         seed=m.get("seed", r["config"].get("seed", 0)), params=r["params"],
                         epochs=r["epochs"], best_epoch=r["best_epoch"], n_train=r["n_train"],
                         val_acc=100 * r["best_val_acc"], test_acc=100 * r["test_acc"],
                         test_loss=r["test_loss"], minutes=r["train_minutes"], gpu=r.get("gpu", "")))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(Path(out_dir) / "results" / "summary.csv", index=False)
    return df


def agg(df, by):
    """mean / std (over seeds) of test accuracy for each group in `by`."""
    g = df.groupby(by, as_index=False).agg(
        test_mean=("test_acc", "mean"), test_std=("test_acc", "std"), val_mean=("val_acc", "mean"),
        runs=("seed", "count"), params=("params", "first"), epochs=("epochs", "first"),
        n_train=("n_train", "first"), minutes=("minutes", "mean"))
    g["test_std"] = g["test_std"].fillna(0.0)
    return g


def _fmt(mean, std, runs):
    return f"{mean:.2f}" if runs < 2 else f"{mean:.2f} $\\pm$ {std:.2f}"


# --------------------------------------------------------------------------- figures
def plot_fraction_sweep(df, path):
    d = df[df.group == "sweep"]
    if d.empty:
        return
    a = agg(d, ["arch", "fraction"])
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for arch, mk, lab in (("vit", "o-", "ViT (from scratch)"), ("resnet18", "s--", "ResNet-18 (from scratch)")):
        s = a[a.arch == arch].sort_values("fraction")
        if len(s):
            ax.errorbar(s.fraction * 100, s.test_mean, yerr=s.test_std, fmt=mk, capsize=3, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100]); ax.set_xticklabels(["10%", "25%", "50%", "100%"])
    ax.set_xlabel("Fraction of the 45k training images"); ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_curves(out_dir, names, labels, path):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for n, lab in zip(names, labels):
        p = Path(out_dir) / "results" / f"{n}_history.csv"
        if not p.exists():
            continue
        h = pd.read_csv(p)
        axes[0].plot(h.epoch, h.val_loss, label=lab)
        axes[1].plot(h.epoch, h.val_acc * 100, label=lab)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation accuracy (%)")
    for a in axes:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_ablation(df, path):
    d = df[df.group.isin(["sweep", "ablation"]) & (df.arch == "vit") & (df.fraction == 1.0)]
    d = d[d.variant.isin(["learned", "sincos2d", "none"])]
    if d.empty:
        return
    a = agg(d, ["variant"]).set_index("variant").reindex(["none", "learned", "sincos2d"]).dropna(subset=["test_mean"])
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    labels = {"none": "none", "learned": "1-D learned", "sincos2d": "2-D sin-cos"}
    ax.bar([labels[i] for i in a.index], a.test_mean, yerr=a.test_std, capsize=4, color=["#999", "#4c72b0", "#55a868"][:len(a)])
    lo = max(0, a.test_mean.min() - 6)
    ax.set_ylim(lo, a.test_mean.max() + 2); ax.set_ylabel("Test accuracy (%)")
    for i, v in enumerate(a.test_mean):
        ax.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def confusion_matrix(preds, labels, n=10):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (labels.astype(int), preds.astype(int)), 1)
    return cm


def top_confusions(cm, k=5):
    off = cm.copy(); np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:k], off.shape))[0]
    return [(CLASSES[i], CLASSES[j], int(off[i, j])) for i, j in idx]


def plot_confusion(out_dir, name, path):
    p = Path(out_dir) / "results" / f"{name}_preds.npz"
    if not p.exists():
        return None
    z = np.load(p)
    cm = confusion_matrix(z["preds"], z["labels"])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=7); ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    per_class = cm.diagonal() / cm.sum(1)
    return dict(top=top_confusions(cm), per_class={c: float(a) for c, a in zip(CLASSES, per_class)})


@torch.no_grad()
def plot_rollout_grid(model, test_ds, device, path, n_correct=4, n_wrong=4, max_scan=2000):
    """Rows: correctly classified images / their rollout overlays / misclassified / overlays."""
    model.eval()
    good, bad = [], []
    for i in range(min(len(test_ds), max_scan)):
        x, y = test_ds[i]
        logits = model(x.unsqueeze(0).to(device))
        p = int(logits.argmax(1))
        (good if p == int(y) else bad).append((x, int(y), p))
        if len(good) >= n_correct and len(bad) >= n_wrong:
            break
    good, bad = good[:n_correct], bad[:n_wrong]
    groups = [("correct", good), ("misclassified", bad)]
    cols = max(len(good), len(bad), 1)
    fig, axes = plt.subplots(4, cols, figsize=(1.6 * cols, 6.6), squeeze=False)
    for gi, (title, items) in enumerate(groups):
        if not items:
            continue
        xs = torch.stack([it[0] for it in items]).to(device)
        _, maps = rollout_maps(model, xs)
        imgs = denormalize(xs).permute(0, 2, 3, 1).cpu().numpy()
        heat = torch.nn.functional.interpolate(maps.unsqueeze(1), size=(32, 32), mode="bilinear",
                                               align_corners=False)[:, 0].cpu().numpy()
        for c in range(cols):
            for r in (0, 1):
                axes[2 * gi + r, c].axis("off")
            if c >= len(items):
                continue
            _, y, p = items[c]
            axes[2 * gi, c].imshow(imgs[c]); axes[2 * gi, c].set_title(f"T:{CLASSES[y]}\nP:{CLASSES[p]}", fontsize=7)
            axes[2 * gi + 1, c].imshow(imgs[c]); axes[2 * gi + 1, c].imshow(heat[c], cmap="jet", alpha=0.5)
        axes[2 * gi, 0].text(-0.15, 0.5, title, transform=axes[2 * gi, 0].transAxes, rotation=90,
                             va="center", ha="right", fontsize=8, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return len(good), len(bad)


# --------------------------------------------------------------------------- report assets
def _row(cells):
    return " & ".join(str(c) for c in cells) + r" \\"


def build_report_assets(out_dir, report_dir):
    out_dir, report_dir = Path(out_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    df = load_summary(out_dir)
    macros, md = {}, []
    if df.empty:
        print("no results found")
        return df

    def setm(name, val):
        macros[name] = f"\\renewcommand{{\\{name}}}{{{val}}}"

    # --- data-fraction sweep
    sw = df[df.group == "sweep"]
    if len(sw):
        a = agg(sw, ["arch", "fraction"])
        lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
                 r"Train images & Epochs & ViT (\%) & ResNet-18 (\%) & Gap (pp) \\", r"\midrule"]
        md += ["| Train fraction (images) | Epochs | ViT test acc (%) | ResNet-18 test acc (%) | Gap (pp) |", "|---|---:|---:|---:|---:|"]
        for f in sorted(a.fraction.unique()):
            v = a[(a.arch == "vit") & (a.fraction == f)]
            r = a[(a.arch == "resnet18") & (a.fraction == f)]
            if v.empty or r.empty:
                continue
            v, r = v.iloc[0], r.iloc[0]
            gap = r.test_mean - v.test_mean
            n = int(v.n_train)
            lines.append(_row([f"{int(f * 100)}\\% ({n:,})".replace(",", "{,}"), int(v.epochs),
                               _fmt(v.test_mean, v.test_std, v.runs),
                               _fmt(r.test_mean, r.test_std, r.runs), f"{gap:+.2f}"]))
            md.append(f"| {int(f * 100)}% ({n:,}) | {int(v.epochs)} | {v.test_mean:.2f} | {r.test_mean:.2f} | {gap:+.2f} |")
            w = FRAC_WORD.get(round(float(f), 2))
            if w:
                setm(f"accVit{w}", f"{v.test_mean:.2f}"); setm(f"accRes{w}", f"{r.test_mean:.2f}")
                setm(f"gap{w}", f"{gap:.2f}")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (report_dir / "tab_sweep.tex").write_text("\n".join(lines))
        for arch, key in (("vit", "Vit"), ("resnet18", "Res")):
            s = sw[(sw.arch == arch) & (sw.fraction == 1.0)]
            if len(s):
                setm(f"params{key}", f"{int(s.params.iloc[0]):,}".replace(",", "{,}"))
                setm(f"epochs{key}", int(s.epochs.iloc[0]))
                setm(f"minutes{key}", f"{s.minutes.mean():.0f}")
        md.append("")

    # --- position-embedding ablation
    ab = df[df.group.isin(["sweep", "ablation"]) & (df.arch == "vit") & (df.fraction == 1.0)]
    ab = ab[ab.variant.isin(["learned", "sincos2d", "none", "learned-mean"])]
    if len(ab):
        a = agg(ab, ["variant"]).set_index("variant")
        names = {"none": "No position embedding", "learned": "1-D learned (paper default)",
                 "sincos2d": "2-D sin-cos (fixed)", "learned-mean": "1-D learned, mean pool (no CLS)"}
        lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Variant & Params & Test acc. (\%) \\", r"\midrule"]
        md += ["| Variant | Params | Test acc (%) |", "|---|---:|---:|"]
        for k in ("none", "learned", "sincos2d", "learned-mean"):
            if k in a.index:
                r = a.loc[k]
                lines.append(_row([names[k], f"{int(r.params):,}".replace(",", "{,}"), _fmt(r.test_mean, r.test_std, r.runs)]))
                md.append(f"| {names[k]} | {int(r.params):,} | {r.test_mean:.2f} |")
                setm("accAbl" + {"none": "None", "learned": "Learned", "sincos2d": "Sincos", "learned-mean": "Mean"}[k], f"{r.test_mean:.2f}")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (report_dir / "tab_ablation.tex").write_text("\n".join(lines))
        md.append("")

    # --- pretrained fine-tuning
    ft = df[df.group == "finetune"]
    paper_rows = [r"\midrule", r"\multicolumn{3}{@{}l}{\emph{Paper (Table 5), ViT-B/16 on CIFAR-10}} \\"]
    if len(ft):
        a = agg(ft, ["arch"]).set_index("arch")
        lab = {"vit_base_patch16_224": "ViT-B/16 (timm, pre-trained)", "resnet50": "ResNet-50 (timm, pre-trained)"}
        lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Model & Params (M) & Test acc. (\%) \\", r"\midrule"]
        md += ["| Model | Params (M) | Test acc (%) |", "|---|---:|---:|"]
        for k in a.index:
            r = a.loc[k]
            lines.append(_row([lab.get(k, k), f"{r.params / 1e6:.1f}", _fmt(r.test_mean, r.test_std, r.runs)]))
            md.append(f"| {lab.get(k, k)} | {r.params / 1e6:.1f} | {r.test_mean:.2f} |")
            setm("accFt" + ("Vit" if k.startswith("vit") else "Res"), f"{r.test_mean:.2f}")
            setm("epochsFt", int(r.epochs))
        lines += paper_rows
        for k, v in PAPER_VITB16_C10.items():
            lines.append(_row([f"\\quad pre-trained on {k}", "86", f"{v:.2f}"]))
        lines += [r"\bottomrule", r"\end{tabular}"]
        (report_dir / "tab_finetune.tex").write_text("\n".join(lines))
        md.append("")

    # --- failure-case macros (from failure_cases.json written by the notebook)
    fc = out_dir / "results" / "failure_cases.json"
    if fc.exists():
        z = json.loads(fc.read_text())
        for key, tag in (("vit", "Vit"), ("resnet18", "Res")):
            if key not in z or z[key] is None:
                continue
            top = z[key]["top"][:3]
            setm("topConf" + tag, ", ".join(f"{a}$\\\\to${b} ({n})" for a, b, n in top))
            pc = z[key]["per_class"]
            worst, best = min(pc, key=pc.get), max(pc, key=pc.get)
            setm("worstClass" + tag, worst); setm("worstAcc" + tag, f"{100 * pc[worst]:.1f}")
            setm("bestClass" + tag, best); setm("bestAcc" + tag, f"{100 * pc[best]:.1f}")

    (report_dir / "results_macros.tex").write_text("\n".join(macros.values()) + "\n")
    (out_dir / "results" / "results_table.md").write_text("\n".join(md) + "\n")
    print("wrote report assets to", report_dir)
    return df


def update_readme(readme, table_md):
    """Replace the block between <!-- RESULTS:START --> and <!-- RESULTS:END --> in the README."""
    readme = Path(readme)
    txt = readme.read_text()
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if a not in txt or b not in txt:
        print("README markers not found; nothing updated")
        return
    head, rest = txt.split(a, 1)
    _, tail = rest.split(b, 1)
    readme.write_text(head + a + "\n" + table_md.strip() + "\n" + b + tail)
    print("README results block updated")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--report-dir", default="reports/generated")
    ap.add_argument("--update-readme", default="", help="path of README.md to patch with the results tables")
    a = ap.parse_args()
    build_report_assets(a.out_dir, a.report_dir)
    if a.update_readme:
        update_readme(a.update_readme, (Path(a.out_dir) / "results" / "results_table.md").read_text())
