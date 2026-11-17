"""Train a small residual projection adapter on BANKING77 5-shot BGE vectors.

This is the allowed training-based fallback (not SetFit).  The frozen BGE
encoder is not fine-tuned.  Only a 384->256->384 residual adapter is trained
with a supervised contrastive loss + identity regularization on the same 5-shot
support used by B0.  Thresholding is calibrated on the same validation split.

No test label is used for training or threshold selection.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "data" / "manifest"
CACHE = PROJECT_ROOT / "data" / "cache_embeddings" / "gate3"
RESULTS = PROJECT_ROOT / "results"
QWEN_PROMPT_HASH = "4bbdf1c169061f64ba742f64bca5e422d05e94ccc7261d29495c233ed9a90005"
BGE_REV = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_PROMPT_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
DATASET = "banking77"
SEED = 42
SHOT = 5


def load_cache() -> dict:
    path = CACHE / "bge" / DATASET / BGE_REV / f"L128_{BGE_PROMPT_HASH[:16]}" / "embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    return {
        "vectors": data["vectors"].astype(np.float32),
        "sample_ids": data["sample_ids"].astype(str),
        "roles": data["roles"].astype(str),
        "labels": data["labels"].astype(str),
        "intents": data["intents"].astype(str),
    }


def support_set(cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader((MANIFEST / f"gate3_{DATASET}_support_{SHOT}.csv").open("r", encoding="utf-8")))
    ids = [row["sample_id"] for row in rows]
    id_to_idx = {sid: i for i, sid in enumerate(cache["sample_ids"])}
    idx = np.asarray([id_to_idx[sid] for sid in ids], dtype=int)
    return cache["vectors"][idx], np.asarray([row["intent"] for row in rows]), np.asarray(ids)


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int = 384, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.fc2(F.gelu(self.fc1(x)))
        return F.normalize(x + residual, dim=1)


def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    n = len(z)
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mask = (y[:, None] == y[None, :]) & (~self_mask)
    counts = mask.sum(dim=1).clamp_min(1)
    return -((log_prob * mask).sum(dim=1) / counts).mean()


def encode_adapter(model: nn.Module, vectors: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        out = model(torch.from_numpy(vectors)).cpu().numpy()
    return out.astype(np.float32)


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cache = load_cache()
    support_X, support_y, support_ids = support_set(cache)
    classes = sorted(set(cache["labels"][cache["roles"] == "known_test"]))
    class_to_idx = {name: i for i, name in enumerate(classes)}
    y_idx = np.asarray([class_to_idx[name] for name in support_y], dtype=np.int64)
    if len(classes) != 40:
        raise RuntimeError(f"Expected 40 BANKING known classes, got {len(classes)}")
    if len(support_X) != 200:
        raise RuntimeError(f"Expected 200 support rows, got {len(support_X)}")

    model = ResidualAdapter(384, 256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.from_numpy(support_X)
    y = torch.from_numpy(y_idx)
    history = []
    model.train()
    for epoch in range(1, 151):
        permutation = torch.randperm(len(X))
        total_loss = 0.0
        total_identity = 0.0
        correct = 0
        total = 0
        for start in range(0, len(X), 64):
            idx = permutation[start : start + 64]
            xb = X[idx]
            yb = y[idx]
            optimizer.zero_grad()
            zb = model(xb)
            loss_c = supcon_loss(zb, yb)
            identity = (1.0 - (zb * xb).sum(dim=1)).mean()
            loss = loss_c + 0.05 * identity
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(idx)
            total_identity += float(identity) * len(idx)
            # nearest prototype training accuracy using current adapted support
            with torch.no_grad():
                all_z = model(X)
                protos = []
                for c in range(len(classes)):
                    center = all_z[y == c].mean(dim=0)
                    protos.append(F.normalize(center, dim=0))
                protos = torch.stack(protos)
                pred = (all_z @ protos.T).argmax(dim=1)
                correct = int((pred == y).sum())
                total = len(y)
        history.append(
            {
                "epoch": epoch,
                "loss": round(total_loss / len(X), 6),
                "contrastive_identity": round(total_identity / len(X), 6),
                "support_nearest_prototype_accuracy": correct / total,
            }
        )
        if epoch in (1, 10, 25, 50, 100, 150):
            print(f"epoch={epoch} loss={history[-1]['loss']:.4f} acc={history[-1]['support_nearest_prototype_accuracy']:.3f}")

    model.eval()
    support_Z = encode_adapter(model, support_X)
    adapted = {}
    for role in ("calibration_known", "known_test", "dev_unknown", "final_unknown"):
        mask = cache["roles"] == role
        adapted[role] = encode_adapter(model, cache["vectors"][mask])

    # prototypes from adapted support
    prototypes = []
    for name in classes:
        rows = support_Z[support_y == name]
        center = rows.mean(axis=0)
        prototypes.append(center / np.linalg.norm(center))
    prototypes = np.stack(prototypes)

    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.gate3.methods import evaluate, select_threshold

    val_Z = np.concatenate([adapted["calibration_known"], adapted["dev_unknown"]])
    val_true = np.concatenate(
        [
            np.asarray([class_to_idx[x] for x in cache["labels"][cache["roles"] == "calibration_known"]]),
            np.full(len(adapted["dev_unknown"]), -1, dtype=int),
        ]
    )
    test_Z = np.concatenate([adapted["known_test"], adapted["final_unknown"]])
    test_true = np.concatenate(
        [
            np.asarray([class_to_idx[x] for x in cache["labels"][cache["roles"] == "known_test"]]),
            np.full(len(adapted["final_unknown"]), -1, dtype=int),
        ]
    )
    val_sims = val_Z @ prototypes.T
    test_sims = test_Z @ prototypes.T
    tau, _, selection = select_threshold(val_sims, val_true, len(classes), classes)
    val_metrics, _, _ = evaluate(val_sims, val_true, classes, tau)
    test_metrics, test_pred, _ = evaluate(test_sims, test_true, classes, tau)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "gate3_adapter_training.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    metrics_rows = [
        {
            "method": "BGE-B0-5shot",
            "known_acceptance": None,
            "known_correct_routing": None,
            "oos_recall": None,
            "open_set_macro_f1": None,
            "note": "Filled from gate3_test_metrics.csv for the same model/dataset/shot.",
        },
        {
            "method": "BGE-Adapter-5shot",
            "known_acceptance": test_metrics["known_acceptance"],
            "known_correct_routing": test_metrics["known_correct_routing"],
            "oos_recall": test_metrics["oos_recall"],
            "open_set_macro_f1": test_metrics["open_set_macro_f1"],
            "note": "Residual projection adapter, supervised contrastive + identity, threshold on same validation.",
        },
    ]
    matrix = list(csv.DictReader((RESULTS / "gate3_test_metrics.csv").open("r", encoding="utf-8")))
    baseline = [
        row
        for row in matrix
        if row["dataset"] == DATASET and row["model"] == "bge" and int(row["shot"]) == SHOT and row["method"] == "B0"
    ]
    if len(baseline) != 1:
        raise RuntimeError("Missing BGE-B0 baseline row")
    b = baseline[0]
    metrics_rows[0].update(
        {
            "known_acceptance": float(b["known_acceptance"]),
            "known_correct_routing": float(b["known_correct_routing"]),
            "oos_recall": float(b["oos_recall"]),
            "open_set_macro_f1": float(b["open_set_macro_f1"]),
        }
    )
    with (RESULTS / "gate3_adapter_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    (RESULTS / "gate3_adapter_threshold.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "model": "bge",
                "shot": SHOT,
                "method": "BGE-Adapter",
                "tau": tau,
                "selection": selection,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "training_samples": len(support_X),
                "epochs": 150,
                "params": sum(p.numel() for p in model.parameters()),
                "identity_weight": 0.05,
                "temperature": 0.1,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pred_rows = []
    test_ids = np.concatenate(
        [
            cache["sample_ids"][cache["roles"] == "known_test"],
            cache["sample_ids"][cache["roles"] == "final_unknown"],
        ]
    )
    for i, true_idx in enumerate(test_true):
        pred_rows.append(
            {
                "sample_id": test_ids[i],
                "role": "known_test" if true_idx >= 0 else "final_unknown",
                "true_label": classes[true_idx] if true_idx >= 0 else "OOS",
                "pred_label": test_pred[i],
                "accepted": int(test_pred[i] != "OOS"),
                "s1": float(test_sims[i].max()),
                "tau": tau,
            }
        )
    with (RESULTS / "gate3_adapter_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    print(json.dumps({"selection": selection, "validation": val_metrics, "test": test_metrics}, ensure_ascii=False, indent=2))
    print("GATE3_ADAPTER_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
