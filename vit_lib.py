"""vit_lib.py -- from-scratch Vision Transformer + training utilities for CIFAR-10.

Everything that constitutes the paper's core contribution is written here by hand:
patch embedding, CLS token, position embeddings (learned-1D / fixed 2D sin-cos / none),
multi-head self-attention (manual, returns attention maps), pre-LN encoder blocks,
MLP head, and attention rollout.  Standard building blocks (nn.Linear, nn.Conv2d,
nn.LayerNorm, optimisers, torchvision datasets/transforms, timm pretrained backbones
for the fine-tuning experiment) are used as permitted by the project brief.

Reference consulted for the *paper only*: Dosovitskiy et al., ICLR 2021 (arXiv:2010.11929);
attention rollout follows Abnar & Zuidema (2020, arXiv:2005.00928).
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


# --------------------------------------------------------------------------- utils
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- model
class PatchEmbedding(nn.Module):
    """Split the image into non-overlapping PxP patches and linearly project each one.

    A Conv2d with kernel = stride = P is mathematically identical to
    "flatten each patch to a P*P*C vector, then apply one shared nn.Linear(P*P*C, D)".
    Parameter count: (P*P*C)*D + D.
    """

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.grid = img_size // patch_size
        self.num_patches = self.grid ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                       # (B,C,H,W) -> (B,N,D)
        return self.proj(x).flatten(2).transpose(1, 2)


def sincos_2d_pos_embed(embed_dim: int, grid: int, cls_token: bool = True) -> torch.Tensor:
    """Fixed 2-D sine-cosine position embedding, shape (1, N(+1), D).

    Half of the channels encode the row index, half the column index, each with
    the sin/cos frequency ladder from "Attention Is All You Need".
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2-D sin-cos"
    d = embed_dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d, dtype=torch.float32) / d))
    ys, xs = torch.meshgrid(torch.arange(grid, dtype=torch.float32),
                            torch.arange(grid, dtype=torch.float32), indexing="ij")
    out_y = ys.flatten()[:, None] * omega[None]      # (N, d)
    out_x = xs.flatten()[:, None] * omega[None]
    pos = torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=1)  # (N, D)
    if cls_token:
        pos = torch.cat([torch.zeros(1, embed_dim), pos], dim=0)
    return pos.unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    """softmax(Q K^T / sqrt(d_h)) V, written out explicitly so the attention map is available."""

    def __init__(self, dim, heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.head_dim = heads, dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                   # each (B,H,N,d_h)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)   # (B,H,N,N)
        out = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj_drop(self.proj(out))
        return (out, attn) if return_attention else out


class MLP(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hidden, dim), nn.Dropout(drop))

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """Pre-LN Transformer encoder block: x + MSA(LN(x)); x + MLP(LN(x))."""

    def __init__(self, dim, heads, mlp_dim, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            y, attn = self.attn(self.norm1(x), return_attention=True)
        else:
            y, attn = self.attn(self.norm1(x)), None
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return (x, attn) if return_attention else x


class VisionTransformer(nn.Module):
    """Compact ViT.

    pos_embed: 'learned' (1-D learned, as in the paper) | 'sincos2d' (fixed) | 'none'
    pool:      'cls'  (prepend a learnable CLS token, classify from it, as in the paper)
               'mean' (no CLS token, global average pool of the patch tokens)
    """

    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 embed_dim=192, depth=8, heads=6, mlp_dim=768, drop=0.0, attn_drop=0.0,
                 pos_embed="learned", pool="cls"):
        super().__init__()
        assert pos_embed in ("learned", "sincos2d", "none")
        assert pool in ("cls", "mean")
        self.pos_kind, self.pool = pos_embed, pool
        self.use_cls = pool == "cls"
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        n_tok = self.patch_embed.num_patches + int(self.use_cls)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.use_cls else None

        if pos_embed == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, n_tok, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed == "sincos2d":
            self.register_buffer("pos_embed", sincos_2d_pos_embed(
                embed_dim, self.patch_embed.grid, cls_token=self.use_cls))
        else:
            self.register_buffer("pos_embed", torch.zeros(1, n_tok, embed_dim))
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList(
            [EncoderBlock(embed_dim, heads, mlp_dim, drop, attn_drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x, return_attention=False):
        x = self.patch_embed(x)
        if self.use_cls:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.drop(x + self.pos_embed)
        attns = []
        for blk in self.blocks:
            if return_attention:
                x, a = blk(x, return_attention=True)
                attns.append(a)
            else:
                x = blk(x)
        x = self.norm(x)
        feat = x[:, 0] if self.use_cls else x.mean(dim=1)
        logits = self.head(feat)
        return (logits, attns) if return_attention else logits


def build_resnet18_cifar(num_classes=10) -> nn.Module:
    """torchvision ResNet-18 with the standard CIFAR stem (3x3 stride-1 conv, no max-pool)."""
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


class ResizeWrapper(nn.Module):
    """Upsample 32x32 inputs on the GPU before an ImageNet-style backbone (fast fine-tuning)."""

    def __init__(self, model: nn.Module, size: int):
        super().__init__()
        self.model, self.size = model, size

    def forward(self, x):
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return self.model(x)


def build_pretrained(arch: str, num_classes=10, img_size=224, pretrained=True):
    """timm pretrained backbone (+ GPU resize).  Returns (model, mean, std)."""
    import timm
    kw = {"img_size": img_size} if arch.startswith("vit") and img_size != 224 else {}
    m = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, **kw)
    cfg = getattr(m, "pretrained_cfg", {}) or {}
    mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))
    return ResizeWrapper(m, img_size), mean, std


# --------------------------------------------------------------------------- data
@dataclass
class Data:
    train: torch.utils.data.Dataset      # 45k, training transform
    val: torch.utils.data.Dataset        # 5k, eval transform (disjoint from train)
    test: torch.utils.data.Dataset       # official 10k test set
    train_order: list                    # fixed permutation of train positions (nested subsets)

    def train_fraction(self, fraction: float) -> Subset:
        """Nested subsets: the 10% subset is contained in the 25% subset, etc."""
        n = int(round(len(self.train) * fraction))
        return Subset(self.train, self.train_order[:n])


def get_data(root, aug="strong", mean=CIFAR_MEAN, std=CIFAR_STD, val_size=5000,
             split_seed=42, fake_size=0) -> Data:
    """CIFAR-10 with a fixed 45k/5k train/val split taken from the 50k training images.

    aug='basic'  : random crop (pad 4) + horizontal flip
    aug='strong' : basic + RandAugment(2, 9)  (mixup/cutmix are applied in the train loop)
    fake_size>0  : random FakeData of that size (smoke tests only; no download)
    """
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    base = [T.RandomCrop(32, padding=4, padding_mode="reflect"), T.RandomHorizontalFlip()]
    if aug == "strong":
        base.append(T.RandAugment(num_ops=2, magnitude=9))
    train_tf, eval_tf = T.Compose(base + norm), T.Compose(norm)

    if fake_size:
        mk = lambda tf, n: torchvision.datasets.FakeData(
            size=n, image_size=(3, 32, 32), num_classes=10, transform=tf, random_offset=0)
        n_tr = fake_size
        tr_aug, tr_eval, test = mk(train_tf, n_tr), mk(eval_tf, n_tr), mk(eval_tf, max(fake_size // 2, 16))
        val_size = max(n_tr // 10, 8)
    else:
        mk = lambda tf, tr: torchvision.datasets.CIFAR10(root=str(root), train=tr, download=True, transform=tf)
        tr_aug, tr_eval, test = mk(train_tf, True), mk(eval_tf, True), mk(eval_tf, False)
        n_tr = 50000

    perm = torch.randperm(n_tr, generator=torch.Generator().manual_seed(split_seed)).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    order = torch.randperm(len(train_idx), generator=torch.Generator().manual_seed(split_seed + 1)).tolist()
    return Data(Subset(tr_aug, train_idx), Subset(tr_eval, val_idx), test, order)


def make_loader(ds, batch_size=128, shuffle=False, seed=0, num_workers=2, device=None):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=(device is not None and device.type == "cuda"),
                      generator=g, drop_last=False, persistent_workers=num_workers > 0)


# --------------------------------------------------------------------------- mixing
def rand_bbox(h, w, lam):
    cut = math.sqrt(1.0 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    return y1, y2, x1, x2


def mix_batch(x, y, mix_prob=0.0, mixup_alpha=0.8, cutmix_alpha=1.0):
    """Randomly apply mixup or cutmix (50/50) with probability mix_prob.
    Returns x, y_a, y_b, lam;  loss = lam*CE(y_a) + (1-lam)*CE(y_b)."""
    if mix_prob <= 0 or np.random.rand() > mix_prob:
        return x, y, y, 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    if np.random.rand() < 0.5:                                      # mixup
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        x = lam * x + (1 - lam) * x[perm]
    else:                                                           # cutmix
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        y1, y2, x1, x2 = rand_bbox(x.size(2), x.size(3), lam)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3)))
    return x, y, y[perm], lam


# --------------------------------------------------------------------------- training
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    optimizer: str = "adamw"            # 'adamw' | 'sgd'
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 3.0
    label_smoothing: float = 0.1
    mix_prob: float = 0.8               # 0 disables mixup/cutmix
    grad_clip: float = 1.0
    amp: bool = True                    # fp16 autocast on CUDA only
    num_workers: int = 2
    seed: int = 42


def _param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith("pos_embed") or "cls_token" in n else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def _make_optimizer(model, cfg: TrainConfig):
    groups = _param_groups(model, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.999))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=0.9, nesterov=True)
    raise ValueError(cfg.optimizer)


def _make_scheduler(opt, cfg: TrainConfig, steps_per_epoch):
    total = cfg.epochs * steps_per_epoch
    warm = int(cfg.warmup_epochs * steps_per_epoch)

    def f(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return 1e-3 + (1 - 1e-3) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def _scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate(model, loader, device, amp=False):
    model.eval()
    loss_sum, n, preds, labels = 0.0, 0, [], []
    use_amp = amp and device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(x)
        loss_sum += F.cross_entropy(out.float(), y, reduction="sum").item()
        n += y.numel()
        preds.append(out.argmax(1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return loss_sum / n, float((preds == labels).mean()), preds, labels


def fit(name, model, train_ds, val_ds, test_ds, cfg: TrainConfig, out_dir, device,
        meta=None, resume=True, verbose=True):
    """Train with warmup+cosine, early best-val selection, resumable checkpoints.

    Finished runs are cached: if <out_dir>/results/<name>.json exists, it is returned untouched.
    Test metrics are computed once, on the best-validation checkpoint.
    """
    out_dir = Path(out_dir)
    res_dir, ck_dir = out_dir / "results", out_dir / "checkpoints"
    res_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)
    res_path, last_p, best_p = res_dir / f"{name}.json", ck_dir / f"{name}_last.pt", ck_dir / f"{name}_best.pt"
    if res_path.exists():
        if verbose:
            print(f"[{name}] cached result found -> skipping")
        return json.loads(res_path.read_text())

    seed_everything(cfg.seed)
    model = model.to(device)
    train_loader = make_loader(train_ds, cfg.batch_size, True, cfg.seed + 1, cfg.num_workers, device)
    val_loader = make_loader(val_ds, 256, False, 0, cfg.num_workers, device)
    test_loader = make_loader(test_ds, 256, False, 0, cfg.num_workers, device)
    opt = _make_optimizer(model, cfg)
    sched = _make_scheduler(opt, cfg, len(train_loader))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = _scaler(use_amp)

    hist, start, best_val, best_ep, elapsed = [], 0, -1.0, 0, 0.0
    if resume and last_p.exists():
        ck = torch.load(last_p, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        hist, start, best_val, best_ep, elapsed = ck["hist"], ck["epoch"], ck["best_val"], ck["best_ep"], ck["elapsed"]
        if verbose:
            print(f"[{name}] resuming from epoch {start}/{cfg.epochs}")

    for ep in range(start, cfg.epochs):
        t0 = time.time()
        model.train()
        loss_sum, acc_sum, n = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, ya, yb, lam = mix_batch(x, y, cfg.mix_prob)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
            out = out.float()
            loss = lam * F.cross_entropy(out, ya, label_smoothing=cfg.label_smoothing) + \
                (1 - lam) * F.cross_entropy(out, yb, label_smoothing=cfg.label_smoothing)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            pred = out.argmax(1)
            bs = y.numel()
            loss_sum += loss.item() * bs
            acc_sum += (lam * (pred == ya).float().sum() + (1 - lam) * (pred == yb).float().sum()).item()
            n += bs
        v_loss, v_acc, _, _ = evaluate(model, val_loader, device, use_amp)
        dt = time.time() - t0
        elapsed += dt
        hist.append(dict(epoch=ep + 1, train_loss=loss_sum / n, train_acc=acc_sum / n,
                         val_loss=v_loss, val_acc=v_acc, lr=opt.param_groups[0]["lr"], seconds=dt))
        if v_acc > best_val:
            best_val, best_ep = v_acc, ep + 1
            torch.save(model.state_dict(), best_p)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), hist=hist, epoch=ep + 1, best_val=best_val,
                        best_ep=best_ep, elapsed=elapsed), last_p)
        if verbose:
            print(f"[{name}] ep {ep + 1:3d}/{cfg.epochs} loss {loss_sum / n:.3f} "
                  f"val {v_acc * 100:5.2f}% (best {best_val * 100:5.2f}@{best_ep}) {dt:.0f}s")

    model.load_state_dict(torch.load(best_p, map_location=device))
    t_loss, t_acc, preds, labels = evaluate(model, test_loader, device, use_amp)
    np.savez_compressed(res_dir / f"{name}_preds.npz", preds=preds.astype(np.int8), labels=labels.astype(np.int8))
    import pandas as pd
    pd.DataFrame(hist).to_csv(res_dir / f"{name}_history.csv", index=False)
    result = dict(name=name, meta=meta or {}, config=asdict(cfg), params=count_parameters(model),
                  epochs=cfg.epochs, best_epoch=best_ep, best_val_acc=best_val,
                  test_acc=t_acc, test_loss=t_loss, train_minutes=elapsed / 60,
                  n_train=len(train_ds), device=str(device),
                  gpu=(torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"),
                  torch=torch.__version__, history=hist)
    res_path.write_text(json.dumps(result, indent=1))
    last_p.unlink(missing_ok=True)       # finished: keep only the best checkpoint
    if verbose:
        print(f"[{name}] TEST acc {t_acc * 100:.2f}%  (best val {best_val * 100:.2f}% @ epoch {best_ep})")
    return result


def load_best(name, model, out_dir, device):
    p = Path(out_dir) / "checkpoints" / f"{name}_best.pt"
    model.load_state_dict(torch.load(p, map_location=device))
    return model.to(device).eval()


# --------------------------------------------------------------------------- rollout
@torch.no_grad()
def attention_rollout(attn_layers):
    """Abnar & Zuidema (2020).  attn_layers: list of (B,H,N,N).

    For each layer:  A_bar = mean over heads;  A_hat = row_normalise(0.5*A_bar + 0.5*I)
    (the identity models the residual connection).  Rollout = A_hat_L @ ... @ A_hat_1.
    Returns (B,N,N); row 0 is the CLS token's attribution over all tokens.
    """
    B, _, N, _ = attn_layers[0].shape
    eye = torch.eye(N, device=attn_layers[0].device).expand(B, N, N)
    joint = eye
    for a in attn_layers:
        a_hat = 0.5 * a.float().mean(dim=1) + 0.5 * eye
        a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True)
        joint = a_hat @ joint
    return joint


@torch.no_grad()
def rollout_maps(model, images):
    """CLS->patch rollout heat maps (B,g,g), min-max normalised per image, plus logits."""
    assert model.use_cls, "rollout needs a CLS token (pool='cls')"
    logits, attns = model(images, return_attention=True)
    r = attention_rollout(attns)[:, 0, 1:]
    g = model.patch_embed.grid
    r = r.reshape(-1, g, g)
    mn, mx = r.amin((1, 2), keepdim=True), r.amax((1, 2), keepdim=True)
    return logits, (r - mn) / (mx - mn + 1e-8)


def denormalize(x, mean=CIFAR_MEAN, std=CIFAR_STD):
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x * s + m).clamp(0, 1)
