"""CLIP 04: model diffing CLIP vs SigLIP. Same architecture, different objective, different readout.

Both defaults are ViT-B/16 at 224px (12 layers, width 768), so architecture is held fixed and
what differs is: training loss (softmax contrastive vs pairwise sigmoid), data (WIT-400M vs WebLI),
and pooling (CLS token vs attention-pooling head). Differences found here are candidates for
"what the objective/data does to a vision encoder". Data is a confounder we can't remove.

Four questions, one section each:
  A. Behavior: do they get the same images right? How confident, in each model's own terms?
  B. Geometry: how far apart are image and text embeddings (modality gap)? How spread out?
  C. Layers: where do the two encoders' representations agree (CKA), and where does class
     information become linearly decodable (probes)? For CLIP, CLS vs patch-mean.
  D. Text reading: how strongly does written text in the image drive each model's output?

Data: Imagenette (10 easy ImageNet classes), val split, `--per-class` images each.

    uv run python experiments/clip/04_clip_vs_siglip_diff.py
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import Imagenette

from mmsafety.dual_encoders import load_dual_encoder
from mmsafety.images import stamp_text
from mmsafety.similarity import linear_cka

TEMPLATES = ["a photo of a {}.", "a blurry photo of a {}.", "a close-up photo of a {}.", "a photo of the {}.", "an image of a {}."]


def load_images(per_class: int, seed: int):
    """Balanced subset so per-class numbers and probes aren't skewed by class frequency."""
    ds = Imagenette("data", split="val", size="160px", download=not Path("data/imagenette2-160").exists())
    classes = [names[0] for names in ds.classes]
    by_class: dict[int, list[int]] = {}
    for i, (_, y) in enumerate(ds._samples):  # private, but avoids decoding ~4k images just to read labels
        by_class.setdefault(y, []).append(i)
    rng = random.Random(seed)
    idx = [i for y in sorted(by_class) for i in rng.sample(by_class[y], per_class)]
    images = [ds[i][0].convert("RGB") for i in idx]
    labels = torch.tensor([ds[i][1] for i in idx])
    return images, labels, classes


def class_text_embeds(enc, classes):
    """Prompt-ensembled class embeddings (mean over templates), the standard zero-shot recipe.
    Averaging reduces sensitivity to any single template, which would otherwise add noise to the diff."""
    embs = torch.stack([enc.encode_text([t.format(c) for t in TEMPLATES]).mean(0) for c in classes])
    return (embs / embs.norm(dim=-1, keepdim=True)).cpu()


def probe_accuracy(feats: torch.Tensor, labels: torch.Tensor) -> float:
    """5-fold linear-probe accuracy: "is class linearly decodable from this layer?"
    Standardized, because raw residual streams have a few huge-magnitude dims that dominate otherwise."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return float(cross_val_score(clf, feats.numpy(), labels.numpy(), cv=5).mean())


def modality_gap(img: torch.Tensor, txt: torch.Tensor, labels: torch.Tensor) -> dict:
    """Liang et al. 2022: CLIP's image and text embeddings occupy separate cones. Gap = distance
    between the two centroids on the unit sphere. Anisotropy = mean pairwise cosine within a
    modality (1 = everything points the same way). Both shape how usable the joint space is."""
    def mean_pairwise_cos(e):
        s = e @ e.T
        n = len(e)
        return float((s.sum() - n) / (n * (n - 1)))

    return {
        "gap": float((img.mean(0) - txt.mean(0)).norm()),
        "mean_cos_matched": float((img * txt[labels]).sum(-1).mean()),
        "image_anisotropy": mean_pairwise_cos(img),
        "text_anisotropy": mean_pairwise_cos(txt),
    }


def zscore(x: torch.Tensor) -> torch.Tensor:
    """Per-dimension standardization. Needed because raw CKA is dominated by the highest-variance
    dimensions: CLIP develops a single outlier dim at layer 8 holding ~half the variance, which
    swamps raw CKA. Reporting both views keeps one dimension from dictating the conclusion."""
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def top_dim_share(x: torch.Tensor) -> float:
    """Fraction of total variance in the single largest-variance dimension: an outlier detector."""
    v = x.var(0)
    return float(v.max() / v.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs=2, default=["openai/clip-vit-base-patch16", "google/siglip-base-patch16-224"])
    p.add_argument("--per-class", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/clip_04_clip_vs_siglip")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    images, labels, classes = load_images(args.per_class, args.seed)
    n_cls = len(classes)

    # Typographic stimuli (section D), built once so both models see identical pixels.
    rng = random.Random(args.seed)
    stamped_cls = torch.tensor([(int(y) + rng.randint(1, n_cls - 1)) % n_cls for y in labels])  # always a wrong class
    attacked = [stamp_text(im, classes[c], 0.15) for im, c in zip(images, stamped_cls.tolist())]
    # Pure text on flat backgrounds: isolates reading from any object evidence.
    text_only = [stamp_text(Image.new("RGB", (224, 224), (g, g, g)), c, 0.12) for c in classes for g in (60, 120, 180, 230)]
    text_only_labels = torch.tensor([i for i in range(n_cls) for _ in range(4)])
    attacked[0].save(out / "example_attacked.png")

    feats, summary = {}, {}
    for model_id in args.models:
        enc = load_dual_encoder(model_id)
        txt = class_text_embeds(enc, classes)
        f = enc.encode_images(images)
        f_att = enc.encode_images(attacked).embeds
        f_txt = enc.encode_images(text_only).embeds
        feats[enc.name] = f

        scores = enc.label_scores(f.embeds, txt)
        pred = scores.argmax(-1)
        pred_att = enc.label_scores(f_att, txt).argmax(-1)
        pred_txt = enc.label_scores(f_txt, txt).argmax(-1)

        geo = modality_gap(f.embeds, txt, labels)
        summary[enc.name] = {
            "family": enc.family,
            "A_zero_shot_acc": float((pred == labels).float().mean()),
            "A_per_class_acc": {c: float((pred[labels == i] == i).float().mean()) for i, c in enumerate(classes)},
            # Confidence in each model's own output semantics (softmax vs sigmoid); see DualEncoder.label_scores.
            "A_mean_top_score": float(scores.max(-1).values.mean()),
            "A_frac_top_score_below_0.5": float((scores.max(-1).values < 0.5).float().mean()),
            "B_geometry": geo,
            "D_attack_acc": float((pred_att == labels).float().mean()),
            "D_attack_success": float((pred_att == stamped_cls).float().mean()),  # followed the text
            "D_text_only_acc": float((pred_txt == text_only_labels).float().mean()),
            "_pred": pred,
        }
        del enc  # free the first model before loading the second; both at once is tight on 16 GB
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    a, b = list(summary)
    pa, pb = summary[a].pop("_pred"), summary[b].pop("_pred")
    summary["A_agreement"] = {
        "same_prediction": float((pa == pb).float().mean()),
        f"only_{a}_right": float(((pa == labels) & (pb != labels)).float().mean()),
        f"only_{b}_right": float(((pa != labels) & (pb == labels)).float().mean()),
        "both_wrong": float(((pa != labels) & (pb != labels)).float().mean()),
    }

    # C1. Cross-model CKA, every layer of A vs every layer of B (mean-pooled patch states),
    # raw and per-dim standardized (see zscore for why both).
    fa, fb = feats[a].patch_means, feats[b].patch_means
    views = {"raw": (fa, fb), "standardized": (torch.stack([zscore(x) for x in fa]), torch.stack([zscore(x) for x in fb]))}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (view, (xa, xb)) in zip(axes, views.items()):
        cka = torch.tensor([[linear_cka(xa[i], xb[j]) for j in range(len(xb))] for i in range(len(xa))])
        summary[f"C_cka_diagonal_{view}"] = [round(float(cka[i, i]), 3) for i in range(min(len(xa), len(xb)))]
        im = ax.imshow(cka, vmin=0, vmax=1, cmap="viridis", origin="lower")
        ax.set_xlabel(f"{b} layer (0 = embeddings)")
        ax.set_ylabel(f"{a} layer")
        ax.set_title(f"linear CKA ({view}), mean-pooled patches", fontsize=9)
    fig.colorbar(im, ax=axes)
    fig.savefig(out / "cka.png", dpi=130, bbox_inches="tight")
    summary["C_cka_final_embeds"] = linear_cka(feats[a].embeds, feats[b].embeds)
    summary["C_top_dim_variance_share"] = {k: [round(top_dim_share(x), 3) for x in f.patch_means] for k, f in feats.items()}

    # C2. Linear probes per layer. For CLIP, CLS vs patch-mean: LLaVA consumes patches, not CLS.
    probes = {}
    for name, f in feats.items():
        probes[f"{name} patch-mean"] = [probe_accuracy(f.patch_means[l], labels) for l in range(len(f.patch_means))]
        if f.cls is not None:
            probes[f"{name} CLS"] = [probe_accuracy(f.cls[l], labels) for l in range(len(f.cls))]
    summary["C_probe_acc_by_layer"] = {k: [round(v, 3) for v in vs] for k, vs in probes.items()}

    fig, ax = plt.subplots(figsize=(7, 4))
    for k, vs in probes.items():
        ax.plot(range(len(vs)), vs, marker="o", label=k)
    ax.axhline(1 / n_cls, color="gray", ls=":", label="chance")
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("5-fold linear probe acc")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "probes.png", dpi=130)

    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for k in (a, b):
        s = summary[k]
        print(f"\n== {k} ({s['family']})")
        print(f"  A zero-shot acc {s['A_zero_shot_acc']:.3f} | mean top score {s['A_mean_top_score']:.3f} "
              f"| top score < 0.5 on {s['A_frac_top_score_below_0.5']:.0%} of images")
        g = s["B_geometry"]
        print(f"  B modality gap {g['gap']:.3f} | cos(img, own class text) {g['mean_cos_matched']:.3f} "
              f"| anisotropy img {g['image_anisotropy']:.3f} txt {g['text_anisotropy']:.3f}")
        print(f"  D attacked acc {s['D_attack_acc']:.3f} | followed stamped text {s['D_attack_success']:.3f} "
              f"| text-only acc {s['D_text_only_acc']:.3f}")
    print(f"\nagreement: {summary['A_agreement']}")
    print(f"CKA diagonal raw:          {summary['C_cka_diagonal_raw']}")
    print(f"CKA diagonal standardized: {summary['C_cka_diagonal_standardized']}")
    print(f"CKA final embeds: {summary['C_cka_final_embeds']:.3f}")
    for k, vs in summary["C_top_dim_variance_share"].items():
        print(f"top-dim variance share {k:>28}: {vs}")
    for k, vs in summary["C_probe_acc_by_layer"].items():
        print(f"probe {k:>40}: {vs}")
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
