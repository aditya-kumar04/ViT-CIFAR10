"""Fast CPU smoke test (no downloads, ~1-2 min):  python tests/smoke_test.py

Checks shapes, the equivalences claimed in the report, caching/resume, the analysis pipeline.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import analysis
import vit_lib as V

torch.set_num_threads(2)
dev = torch.device("cpu")


def test_model():
    for pos in ("learned", "sincos2d", "none"):
        m = V.VisionTransformer(embed_dim=64, depth=2, heads=4, mlp_dim=128, pos_embed=pos)
        x = torch.randn(3, 3, 32, 32)
        assert m(x).shape == (3, 10)
        logits, attns = m(x, return_attention=True)
        assert len(attns) == 2 and attns[0].shape == (3, 4, 65, 65)
        assert torch.allclose(attns[0].sum(-1), torch.ones(3, 4, 65), atol=1e-5)  # rows are distributions
    m = V.VisionTransformer(embed_dim=64, depth=2, heads=4, mlp_dim=128, pool="mean")
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10) and not m.use_cls
    # fixed embeddings are buffers, not trainable parameters
    n_learn = V.count_parameters(V.VisionTransformer(embed_dim=64, depth=2, heads=4, mlp_dim=128, pos_embed="learned"))
    n_sin = V.count_parameters(V.VisionTransformer(embed_dim=64, depth=2, heads=4, mlp_dim=128, pos_embed="sincos2d"))
    assert n_learn - n_sin == 65 * 64
    print("model ok")


def test_patch_embed_equivalence():
    """Conv2d(k=s=P) == flatten patches + shared Linear."""
    pe = V.PatchEmbedding(32, 4, 3, 16)
    x = torch.randn(2, 3, 32, 32)
    patches = x.unfold(2, 4, 4).unfold(3, 4, 4)                 # B,C,gh,gw,P,P
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(2, 64, 3 * 4 * 4)
    lin = torch.nn.Linear(48, 16)
    with torch.no_grad():
        lin.weight.copy_(pe.proj.weight.reshape(16, -1)); lin.bias.copy_(pe.proj.bias)
    assert torch.allclose(pe(x), lin(patches), atol=1e-5)
    print("patch-embed equivalence ok")


def test_attention_matches_torch():
    """Hand-written MHSA == torch scaled_dot_product_attention."""
    a = V.MultiHeadSelfAttention(32, 4)
    x = torch.randn(2, 10, 32)
    qkv = a.qkv(x).reshape(2, 10, 3, 4, 8).permute(2, 0, 3, 1, 4)
    ref = torch.nn.functional.scaled_dot_product_attention(*qkv.unbind(0))
    ref = a.proj(ref.transpose(1, 2).reshape(2, 10, 32))
    assert torch.allclose(a(x), ref, atol=1e-5)
    print("attention == SDPA ok")


def test_rollout():
    m = V.VisionTransformer(embed_dim=64, depth=3, heads=4, mlp_dim=128).eval()
    logits, maps = V.rollout_maps(m, torch.randn(2, 3, 32, 32))
    assert maps.shape == (2, 8, 8) and float(maps.min()) >= 0 and float(maps.max()) <= 1
    _, attns = m(torch.randn(2, 3, 32, 32), return_attention=True)
    r = V.attention_rollout(attns)
    assert torch.allclose(r.sum(-1), torch.ones(2, 65), atol=1e-4)    # rows stay normalised
    print("rollout ok")


def test_resume():
    """Crash in epoch 2, re-run, and check that training continues from the epoch-1 checkpoint."""
    out = Path(tempfile.mkdtemp())
    data = V.get_data(out / "data", aug="basic", fake_size=96)
    cfg = V.TrainConfig(epochs=3, batch_size=32, num_workers=0, amp=False, warmup_epochs=1, mix_prob=1.0)
    mk = lambda: V.VisionTransformer(embed_dim=32, depth=2, heads=2, mlp_dim=64)
    real, calls = V.mix_batch, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 5:                     # 3 batches/epoch -> dies inside epoch 2
            raise KeyboardInterrupt("simulated crash")
        return real(*a, **k)

    V.mix_batch = flaky
    try:
        V.fit("crash", mk(), data.train, data.val, data.test, cfg, out, dev)
        raise AssertionError("should have crashed")
    except KeyboardInterrupt:
        pass
    finally:
        V.mix_batch = real
    r = V.fit("crash", mk(), data.train, data.val, data.test, cfg, out, dev)
    assert [h["epoch"] for h in r["history"]] == [1, 2, 3], r["history"]
    print("resume ok")


def test_training_pipeline():
    out = Path(tempfile.mkdtemp())
    data = V.get_data(out / "data", aug="strong", fake_size=160)
    cfg = V.TrainConfig(epochs=2, batch_size=32, num_workers=0, amp=False, warmup_epochs=1)
    ds = data.train_fraction(0.5)
    assert len(ds) == int(round(len(data.train) * 0.5))
    assert set(data.train_fraction(0.25).indices) <= set(ds.indices)          # nested subsets
    assert not set(data.train.indices) & set(data.val.indices)                # disjoint split

    def mk():
        return V.VisionTransformer(embed_dim=32, depth=2, heads=2, mlp_dim=64)

    r1 = V.fit("vit-learned_f0.50_s42", mk(), ds, data.val, data.test, cfg, out, dev,
               meta=dict(group="sweep", arch="vit", variant="learned", fraction=0.5, seed=42))
    r2 = V.fit("vit-learned_f0.50_s42", mk(), ds, data.val, data.test, cfg, out, dev)   # cached
    assert r1["test_acc"] == r2["test_acc"]

    # resume: run 2 epochs, delete result, re-run with 3 epochs budget from a 'last' checkpoint
    cfg3 = V.TrainConfig(epochs=3, batch_size=32, num_workers=0, amp=False, warmup_epochs=1, mix_prob=0.0)
    V.fit("resume-test", mk(), ds, data.val, data.test, cfg3, out, dev,
          meta=dict(group="sweep", arch="vit", variant="learned", fraction=1.0, seed=1))
    for arch, build in (("resnet18", V.build_resnet18_cifar),):
        V.fit(f"{arch}_f0.50_s42", build(), ds, data.val, data.test,
              V.TrainConfig(epochs=1, batch_size=32, num_workers=0, amp=False, optimizer="sgd", lr=0.05, weight_decay=5e-4),
              out, dev, meta=dict(group="sweep", arch=arch, variant="", fraction=0.5, seed=42))
    # pretrained-style fine-tuning path (random init, tiny resolution so it runs on CPU)
    for arch in ("resnet50", "vit_base_patch16_224"):
        model, mean, std = V.build_pretrained(arch, img_size=64, pretrained=False)
        d2 = V.get_data(out / "data", aug="basic", mean=mean, std=std, fake_size=48)
        V.fit(f"ft-{arch}", model, d2.train, d2.val, d2.test,
              V.TrainConfig(epochs=1, batch_size=16, num_workers=0, amp=False, optimizer="sgd", lr=0.01,
                            weight_decay=0.0, mix_prob=0.0, label_smoothing=0.0),
              out, dev, meta=dict(group="finetune", arch=arch, variant="", fraction=1.0, seed=42))

    df = analysis.build_report_assets(out, out / "gen")
    assert len(df) >= 5
    m = V.load_best("vit-learned_f0.50_s42", mk(), out, dev)
    analysis.plot_rollout_grid(m, data.test, dev, out / "rollout.png", 2, 2)
    info = analysis.plot_confusion(out, "vit-learned_f0.50_s42", out / "cm.png")
    analysis.plot_fraction_sweep(df, out / "sweep.png")
    analysis.plot_ablation(df, out / "abl.png")
    analysis.plot_curves(out, ["vit-learned_f0.50_s42"], ["vit"], out / "curves.png")
    assert (out / "rollout.png").exists() and (out / "cm.png").exists() and info is not None
    print("pipeline ok ->", out)
    return out


if __name__ == "__main__":
    test_model(); test_patch_embed_equivalence(); test_attention_matches_torch(); test_rollout()
    test_resume(); test_training_pipeline()
    print("ALL SMOKE TESTS PASSED")
