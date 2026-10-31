"""Train and evaluate a tiny eye-state CNN on real driver footage.

Data: 32x32 grayscale eye crops cut from DMD face-camera video with the same
landmark-aligned crop the browser uses (ml/signals.py, lib/detection/eyeCrop.mjs),
labelled from DMD's frame-level `eyes_state` annotation.

Protocol:
  * 5-fold GroupKFold over drivers, so no person is ever in train and test.
  * Three models compared on identical folds: PCA + logistic regression on
    pixels, EyeNet-S (~15k parameters), and EyeNet-L (~60k parameters).
  * Each fold's EyeNet-S is also scored after int8 weight quantization, so
    the cost of shipping the small format is measured, not assumed.
  * The final model, trained on every DMD driver, is tested once on a public
    MRL Eye subset (near-infrared, different people and camera) as an
    out-of-domain check.

    python ml/train_eye_cnn.py            # full run
    python ml/train_eye_cnn.py --quick    # 3 epochs, for a smoke test
"""

from __future__ import annotations

import argparse
import io
import time
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold

from common import COLORS, DATA, FIGURES, LAB, PUBLIC_MODELS, REPORTS, bootstrap_ci, plot_style, write_json

torch.set_num_threads(8)
LABELS = {"open": 0, "close": 1}
MRL_URL = "https://huggingface.co/datasets/oliveirabruno01/litbench-mrl-eye/resolve/main/data/raw-00000-of-00001.parquet"


# --- data ---------------------------------------------------------------------

def load_crops():
    images, labels, subjects, sessions, frames = [], [], [], [], []
    for crop_file in sorted((DATA / "crops" / "dmd").glob("*.npz")):
        session = crop_file.stem
        archive = np.load(crop_file)
        meta = pd.read_parquet(DATA / "signals" / "dmd" / f"{session}.parquet",
                               columns=["frame", "eyes_state", "subject_id"]).set_index("frame")
        states = meta.loc[archive["frame"], "eyes_state"].to_numpy()
        keep = np.isin(states, list(LABELS))
        crops = archive["crops"][keep]
        n = len(crops)
        images.append(crops)
        labels.append(np.array([LABELS[s] for s in states[keep]]))
        subjects += [meta["subject_id"].iloc[0]] * n
        sessions += [session] * n
        frames.append(archive["frame"][keep])
    return (np.concatenate(images), np.concatenate(labels), np.array(subjects),
            np.array(sessions), np.concatenate(frames))


def standardize(batch: np.ndarray) -> np.ndarray:
    """Per-crop standardization; eyeNet.mjs applies the identical formula."""
    batch = batch.astype(np.float32)
    mean = batch.mean(axis=(-1, -2), keepdims=True)
    std = batch.std(axis=(-1, -2), keepdims=True)
    return (batch - mean) / (std + 8.0)


def per_eye(images, labels, groups):
    """(N, 2, 32, 32) frames -> 2N single-eye samples sharing the frame label."""
    return images.reshape(-1, 32, 32), np.repeat(labels, 2), np.repeat(groups, 2)


# --- models -------------------------------------------------------------------

class EyeNet(nn.Module):
    """Four 3x3 conv blocks and global average pooling. Kept deliberately small
    so a plain-JavaScript forward pass costs well under a millisecond."""

    def __init__(self, widths=(8, 16, 32, 32)):
        super().__init__()
        layers, channels = [], 1
        for index, width in enumerate(widths):
            layers += [nn.Conv2d(channels, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU()]
            if index < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
            channels = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 2)

    def forward(self, x):
        x = self.features(x)
        return self.head(x.mean(dim=(2, 3)))


def augment(batch: torch.Tensor) -> torch.Tensor:
    # Shifts of up to 2 px model landmark jitter; flips make one model serve
    # both eyes; gain and offset model exposure changes the standardization
    # does not fully remove (e.g. glare).
    n = batch.shape[0]
    padded = F.pad(batch, (2, 2, 2, 2), mode="replicate")
    dx, dy = np.random.randint(0, 5, 2)
    batch = padded[:, :, dy:dy + 32, dx:dx + 32]
    flip = torch.rand(n) < 0.5
    batch[flip] = batch[flip].flip(-1)
    gain = 1 + 0.15 * torch.randn(n, 1, 1, 1)
    return batch * gain + 0.1 * torch.randn(n, 1, 1, 1)


def train_net(x, y, widths, epochs, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EyeNet(widths)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(x) / 512))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, total_steps=steps)
    weights = torch.tensor(len(y) / (2 * np.bincount(y, minlength=2)), dtype=torch.float32)
    xt = torch.from_numpy(standardize(x)).unsqueeze(1)
    yt = torch.from_numpy(y).long()
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xt))
        total = 0.0
        for start in range(0, len(xt), 512):
            index = order[start:start + 512]
            logits = model(augment(xt[index].clone()))
            loss = F.cross_entropy(logits, yt[index], weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(index)
        history.append(total / len(xt))
    model.eval()
    return model, history


@torch.no_grad()
def predict(model, x) -> np.ndarray:
    xt = torch.from_numpy(standardize(x)).unsqueeze(1)
    out = [F.softmax(model(xt[i:i + 4096]), dim=1)[:, 1] for i in range(0, len(xt), 4096)]
    return torch.cat(out).numpy()


def fold_batchnorm(model: EyeNet):
    """Fold each BatchNorm into the preceding conv: w' = w*g/s, b' = beta - mu*g/s."""
    layers = []
    modules = list(model.features)
    for index, module in enumerate(modules):
        if isinstance(module, nn.Conv2d):
            bn = modules[index + 1]
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            weight = module.weight * scale[:, None, None, None]
            bias = bn.bias - bn.running_mean * scale
            pooled = index + 3 < len(modules) and isinstance(modules[index + 3], nn.MaxPool2d)
            layers.append({"weight": weight.detach().numpy(), "bias": bias.detach().numpy(), "pool": pooled})
    head = {"weight": model.head.weight.detach().numpy(), "bias": model.head.bias.detach().numpy()}
    return layers, head


def quantize(array: np.ndarray):
    """Symmetric per-tensor int8: q = round(w / s), s = max|w| / 127."""
    scale = float(np.abs(array).max() / 127) or 1.0
    return np.clip(np.round(array / scale), -127, 127).astype(np.int8), scale


def shipped_weights(model: EyeNet):
    """The exact tensors the browser runs: BatchNorm folded, then int8 round-trip."""
    layers, head = fold_batchnorm(model)

    def roundtrip(array):
        q, scale = quantize(array)
        return q.astype(np.float32) * scale

    return ([{"weight": roundtrip(l["weight"]), "bias": l["bias"].astype(np.float32), "pool": l["pool"]}
             for l in layers],
            {"weight": roundtrip(head["weight"]), "bias": head["bias"].astype(np.float32)})


@torch.no_grad()
def predict_shipped(weights, x) -> np.ndarray:
    """Forward pass over shipped weights; the reference eyeNet.mjs is tested against."""
    layers, head = weights
    out = []
    xt = torch.from_numpy(standardize(x)).unsqueeze(1)
    for start in range(0, len(xt), 4096):
        h = xt[start:start + 4096]
        for layer in layers:
            h = F.relu(F.conv2d(h, torch.from_numpy(layer["weight"]), torch.from_numpy(layer["bias"]), padding=1))
            if layer["pool"]:
                h = F.max_pool2d(h, 2)
        logits = F.linear(h.mean(dim=(2, 3)), torch.from_numpy(head["weight"]), torch.from_numpy(head["bias"]))
        out.append(F.softmax(logits, dim=1)[:, 1])
    return torch.cat(out).numpy()


def export(model: EyeNet, metrics: dict, samples: np.ndarray) -> dict:
    layers, head = fold_batchnorm(model)

    def pack(array):
        q, scale = quantize(array)
        return {"shape": list(array.shape), "scale": scale, "q": q.flatten().tolist()}

    payload = {
        "format": "eye-state-cnn",
        "version": 1,
        "input": {"size": 32, "standardize": "per-crop mean/(std+8)", "crop_scale": 1.6},
        "classes": ["open", "closed"],
        "layers": [{"weight": pack(l["weight"]), "bias": l["bias"].tolist(), "pool": l["pool"]} for l in layers],
        "head": {"weight": pack(head["weight"]), "bias": head["bias"].tolist()},
        "trained_on": "DMD drowsiness subset, all drivers",
        "metrics": metrics,
        # Reference outputs for tests/eye-net.test.mjs.
        "test_vectors": [
            {"pixels": crop.astype(int).flatten().tolist(), "p_closed": float(p)}
            for crop, p in zip(samples, predict_shipped(shipped_weights(model), samples))
        ],
    }
    write_json(PUBLIC_MODELS / "eye-state-cnn.json", payload, compact=True)
    return payload


# --- evaluation -----------------------------------------------------------------

def summarize(truth, probability, subjects, threshold=0.5):
    prediction = (probability >= threshold).astype(int)
    by_subject = {s: (truth[subjects == s], probability[subjects == s]) for s in np.unique(subjects)}

    def macro_f1(parts):
        t = np.concatenate([p[0] for p in parts])
        pr = np.concatenate([p[1] for p in parts])
        return f1_score(t, pr >= threshold, zero_division=0)

    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "f1_closed": float(f1_score(truth, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
        "f1_closed_ci95": bootstrap_ci(by_subject, macro_f1, repeats=1000),
        "confusion": confusion_matrix(truth, prediction).tolist(),
    }


def load_mrl():
    raw = urllib.request.urlopen(MRL_URL, timeout=180).read()
    table = pd.read_parquet(io.BytesIO(raw))
    from PIL import Image

    images = []
    for record in table["image"]:
        image = Image.open(io.BytesIO(record["bytes"])).convert("L").resize((32, 32), Image.BILINEAR)
        images.append(np.asarray(image))
    # The dataset encodes 0 = Closed_Eyes, 1 = Open_Eyes; flip to 1 = closed.
    labels = 1 - table["label"].to_numpy()
    return np.stack(images), labels, table["subject_id"].astype(str).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    epochs = 3 if args.quick else 8

    images, labels, subjects, sessions, frames = load_crops()
    print(f"{len(images)} labelled frames, {len(np.unique(subjects))} drivers, "
          f"{labels.mean():.1%} closed")

    folds = list(GroupKFold(n_splits=5).split(images, labels, subjects))
    configurations = {
        "pixels_pca_logreg": None,
        "eyenet_s": (8, 16, 32, 32),
        "eyenet_l": (16, 32, 64, 64),
    }
    oof = {name: np.zeros(len(images)) for name in configurations}
    oof["eyenet_s_int8"] = np.zeros(len(images))
    curves = {}
    started = time.time()
    for fold, (train, test) in enumerate(folds):
        x_train, y_train, _ = per_eye(images[train], labels[train], subjects[train])
        x_test = images[test].reshape(-1, 32, 32)
        for name, widths in configurations.items():
            if widths is None:
                flat = standardize(x_train).reshape(len(x_train), -1)
                pca = PCA(64, random_state=0).fit(flat[::4])
                classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
                classifier.fit(pca.transform(flat), y_train)
                test_flat = standardize(x_test).reshape(len(x_test), -1)
                probability = classifier.predict_proba(pca.transform(test_flat))[:, 1]
            else:
                model, history = train_net(x_train, y_train, widths, epochs, seed=fold)
                curves.setdefault(name, []).append(history)
                probability = predict(model, x_test)
                if name == "eyenet_s":
                    oof["eyenet_s_int8"][test] = predict_shipped(shipped_weights(model), x_test).reshape(-1, 2).mean(1)
            # A frame's score is the mean of its two eyes.
            oof[name][test] = probability.reshape(-1, 2).mean(1)
        print(f"fold {fold + 1}/5 done ({time.time() - started:.0f}s)", flush=True)

    results = {name: summarize(labels, score, subjects) for name, score in oof.items()}
    parameters = {name: sum(p.numel() for p in EyeNet(w).parameters()) for name, w in configurations.items() if w}
    for name, count in parameters.items():
        results[name]["parameters"] = int(count)
    results["eyenet_s_int8"]["parameters"] = parameters["eyenet_s"]
    results["pixels_pca_logreg"]["parameters"] = 64 * 1024 + 65

    # Per-frame out-of-fold scores feed ml/evaluate_signals.py.
    pd.DataFrame({"session": sessions, "frame": frames, "subject_id": subjects,
                  "label": labels, **{f"cnn_{k}": v for k, v in oof.items()}}
                 ).to_parquet(DATA / "eye_cnn_oof.parquet", index=False)

    # Final model on every driver, then a single out-of-domain test.
    x_all, y_all, _ = per_eye(images, labels, subjects)
    final, history = train_net(x_all, y_all, configurations["eyenet_s"], epochs, seed=99)
    final_q = shipped_weights(final)
    external = {}
    try:
        mrl_images, mrl_labels, mrl_subjects = load_mrl()
        probability = predict_shipped(final_q, mrl_images)
        external = summarize(mrl_labels, probability, mrl_subjects)
        external["images"] = int(len(mrl_images))
        external["subjects"] = int(len(np.unique(mrl_subjects)))
        print(f"MRL external: acc {external['accuracy']:.3f}, AUC {external['roc_auc']:.3f}")
    except Exception as error:  # the external check is reported as missing, not faked
        external = {"error": str(error)}

    # CPU latency of the float model, single crop, for the size/latency table.
    sample = torch.from_numpy(standardize(images[:1].reshape(-1, 32, 32)[:1])).unsqueeze(1)
    with torch.no_grad():
        for _ in range(20):
            final(sample)
        timer = time.perf_counter()
        for _ in range(500):
            final(sample)
    latency_ms = (time.perf_counter() - timer) / 500 * 1000

    probes = images.reshape(-1, 32, 32)[np.random.default_rng(5).choice(len(images) * 2, 4, replace=False)]
    payload = export(final, {"dmd_cv": results["eyenet_s_int8"], "mrl_external": external}, probes)
    size_kb = len((PUBLIC_MODELS / "eye-state-cnn.json").read_bytes()) / 1024

    report = {
        "dataset": {"frames": int(len(images)), "drivers": int(len(np.unique(subjects))),
                    "sessions": int(len(np.unique(sessions))), "closed_share": float(labels.mean()),
                    "folds": 5, "epochs": epochs},
        "models": results,
        "external_mrl": external,
        "final_model": {"parameters": parameters["eyenet_s"], "export_kb": round(size_kb, 1),
                        "weights_kb": round(parameters["eyenet_s"] / 1024, 1),
                        "torch_cpu_latency_ms": round(latency_ms, 3), "layers": len(payload["layers"])},
        "training_curves": {k: np.mean(v, axis=0).tolist() for k, v in curves.items()},
    }
    write_json(REPORTS / "eye_cnn_metrics.json", report)
    write_json(LAB / "eye_cnn.json", {**report, "roc": {
        name: dict(zip(("fpr", "tpr"), [c[::max(1, len(c) // 150)].tolist() for c in roc_curve(labels, oof[name])[:2]]))
        for name in oof}})
    figures(labels, oof, results)
    for name, r in results.items():
        print(f"{name:20s} acc {r['accuracy']:.3f}  F1 {r['f1_closed']:.3f}  AUC {r['roc_auc']:.3f}")


def figures(labels, oof, results):
    plt = plot_style()
    names = {"pixels_pca_logreg": "Pixels + PCA + LogReg", "eyenet_l": "EyeNet-L (float)",
             "eyenet_s": "EyeNet-S (float)", "eyenet_s_int8": "EyeNet-S (int8, shipped)"}
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for (key, label), color in zip(names.items(), [COLORS["muted"], COLORS["series"][1], COLORS["series"][2], COLORS["accent"]]):
        fpr, tpr, _ = roc_curve(labels, oof[key])
        ax.plot(fpr, tpr, color=color, lw=2.2 if key == "eyenet_s_int8" else 1.4,
                label=f"{label}  AUC {results[key]['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color=COLORS["muted"], lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate (closed eyes)",
           title="Eye-state CNN, held-out drivers (DMD, 5-fold)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.savefig(FIGURES / "eye_cnn_roc.png")
    plt.close(fig)

    # No crop thumbnails are rendered: DMD is CC BY-NC-ND, so adapted images
    # of its drivers are not redistributed in figures.


if __name__ == "__main__":
    main()
