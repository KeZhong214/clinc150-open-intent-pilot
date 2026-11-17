"""SetFit-BGE training on BANKING77 5-shot support (Gate 3, training-based)."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel, Trainer, TrainingArguments
from setfit.modeling import SetFitHead
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "data" / "manifest"
MODEL_CACHE = PROJECT_ROOT / "data" / "model_cache"
RESULTS = PROJECT_ROOT / "results"
DATASET = "banking77"
SEED = 42
SHOT = 5
BGE_REV = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
OOS = "OOS"


def threshold_select(probs, true_idx):
    scores = probs.max(axis=1)
    known = true_idx >= 0
    known_scores = scores[known]
    candidates = np.unique(np.quantile(known_scores, np.linspace(0, 1, 101)))
    rows = []
    best = None
    for tau in candidates:
        accept = scores >= tau
        known_acceptance = float(np.mean(accept[known]))
        oos_recall = float(np.mean(~accept[~known]))
        pred = np.where(accept, probs.argmax(axis=1), -1)
        y = true_idx.copy()
        f1 = float(
            f1_score(
                y,
                pred,
                labels=list(range(40)) + [-1],
                average="macro",
                zero_division=0,
            )
        )
        row = {
            "tau": float(tau),
            "known_acceptance": known_acceptance,
            "oos_recall": oos_recall,
            "open_set_macro_f1": f1,
            "constraint_met": known_acceptance >= 0.90,
        }
        rows.append(row)
        if row["constraint_met"]:
            key = (row["open_set_macro_f1"], row["oos_recall"], row["known_acceptance"], row["tau"])
            if best is None or key > best["_key"]:
                best = {**row, "_key": key}
    if best is None:
        best = max(rows, key=lambda row: (row["open_set_macro_f1"], row["oos_recall"]))
    return float(best["tau"]), rows, best


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    support_rows = list(csv.DictReader((MANIFEST / f"gate3_{DATASET}_support_{SHOT}.csv").open("r", encoding="utf-8")))
    classes = sorted({row["intent"] for row in support_rows})
    class_to_idx = {name: i for i, name in enumerate(classes)}
    texts = [row["text"] for row in support_rows]
    labels = [class_to_idx[row["intent"]] for row in support_rows]
    train_ds = Dataset.from_dict({"text": texts, "label": labels})

    body = SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        revision=BGE_REV,
        device="cpu",
        cache_folder=str(MODEL_CACHE),
    )
    body.max_seq_length = 128
    head = SetFitHead(in_features=384, out_features=len(classes), temperature=1.0)
    model = SetFitModel(
        model_body=body,
        model_head=head,
        labels=classes,
        normalize_embeddings=True,
    )
    args = TrainingArguments(
        batch_size=(16, 2),
        num_epochs=(1, 16),
        body_learning_rate=(2e-5, 1e-5),
        head_learning_rate=0.01,
        max_length=128,
        seed=SEED,
        report_to="none",
        show_progress_bar=False,
        use_amp=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    t0 = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - t0

    # Reload cached data texts for validation/test, preserving the Gate-3 rows.
    def rows_for(role: str):
        rows = [row for row in csv.DictReader((MANIFEST / f"gate3_{DATASET}_manifest.csv").open("r", encoding="utf-8")) if row["role"] == role]
        if not rows:
            raise RuntimeError(f"No rows for {role}")
        return rows

    def probs_for(role: str):
        rows = rows_for(role)
        p = model.predict_proba([row["text"] for row in rows], batch_size=32, show_progress_bar=False)
        if isinstance(p, torch.Tensor):
            p = p.cpu().numpy()
        return p.astype(np.float32), rows

    cal_p, cal_rows = probs_for("calibration_known")
    dev_p, dev_rows = probs_for("dev_unknown")
    test_known_p, test_known_rows = probs_for("known_test")
    test_unknown_p, test_unknown_rows = probs_for("final_unknown")

    val_p = np.concatenate([cal_p, dev_p])
    val_true = np.concatenate(
        [
            np.asarray([class_to_idx[row["open_set_label"]] for row in cal_rows], dtype=int),
            np.full(len(dev_rows), -1, dtype=int),
        ]
    )
    tau, grid, selected = threshold_select(val_p, val_true)

    test_p = np.concatenate([test_known_p, test_unknown_p])
    test_true = np.concatenate(
        [
            np.asarray([class_to_idx[row["open_set_label"]] for row in test_known_rows], dtype=int),
            np.full(len(test_unknown_rows), -1, dtype=int),
        ]
    )
    known = test_true >= 0
    accept = test_p.max(axis=1) >= tau
    pred = np.where(accept, test_p.argmax(axis=1), -1)
    known_acceptance = float(np.mean(accept[known]))
    known_routing = float(np.mean(pred[known] == test_true[known]))
    oos_recall = float(np.mean(~accept[~known]))
    open_f1 = float(f1_score(test_true, pred, labels=list(range(40)) + [-1], average="macro", zero_division=0))
    auroc = float(roc_auc_score(known.astype(int), test_p.max(axis=1)))
    val_accept = val_p.max(axis=1) >= tau
    val_pred = np.where(val_accept, val_p.argmax(axis=1), -1)
    val_f1 = float(f1_score(val_true, val_pred, labels=list(range(40)) + [-1], average="macro", zero_division=0))
    test_metrics = {
        "known_acceptance": known_acceptance,
        "known_correct_routing": known_routing,
        "oos_recall": oos_recall,
        "unknown_auroc": auroc,
        "open_set_macro_f1": open_f1,
        "open_set_accuracy": float(np.mean(pred == test_true)),
        "n_known": int(known.sum()),
        "n_unknown": int((~known).sum()),
        "known_rejected": int(np.sum(~accept[known])),
        "unknown_accepted": int(np.sum(accept[~known])),
    }
    val_metrics = {
        "tau": tau,
        "known_acceptance": float(np.mean(val_accept[val_true >= 0])),
        "oos_recall": float(np.mean(~val_accept[val_true < 0])),
        "open_set_macro_f1": val_f1,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    # Training history from the SentenceTransformers trainer if available.
    history = []
    try:
        history = list(trainer.st_trainer.state.log_history)
    except Exception:
        history = []
    with (RESULTS / "gate3_setfit_training.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": DATASET,
                "model": "BAAI/bge-small-en-v1.5 + SetFit",
                "revision": BGE_REV,
                "shot": SHOT,
                "n_training_samples": len(texts),
                "n_classes": len(classes),
                "train_seconds": round(train_seconds, 3),
                "training_arguments": {
                    "batch_size": [16, 2],
                    "num_epochs": [1, 16],
                    "body_learning_rate": [2e-5, 1e-5],
                    "head_learning_rate": 0.01,
                    "seed": SEED,
                    "max_length": 128,
                },
                "history": history,
                "selection": selected,
                "validation": val_metrics,
                "test": test_metrics,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    matrix = list(csv.DictReader((RESULTS / "gate3_test_metrics.csv").open("r", encoding="utf-8")))
    base = [
        row for row in matrix
        if row["dataset"] == DATASET and row["model"] == "bge" and int(row["shot"]) == SHOT and row["method"] == "B0"
    ][0]
    with (RESULTS / "gate3_setfit_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "known_acceptance", "known_correct_routing", "oos_recall", "unknown_auroc", "open_set_macro_f1", "note"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "model": "BGE-B0-5shot",
                "known_acceptance": base["known_acceptance"],
                "known_correct_routing": base["known_correct_routing"],
                "oos_recall": base["oos_recall"],
                "unknown_auroc": base["unknown_auroc"],
                "open_set_macro_f1": base["open_set_macro_f1"],
                "note": "Frozen Gate-3 matrix BGE B0 5-shot; same test.",
            }
        )
        writer.writerow(
            {
                "model": "SetFit-BGE-5shot",
                "known_acceptance": test_metrics["known_acceptance"],
                "known_correct_routing": test_metrics["known_correct_routing"],
                "oos_recall": test_metrics["oos_recall"],
                "unknown_auroc": test_metrics["unknown_auroc"],
                "open_set_macro_f1": test_metrics["open_set_macro_f1"],
                "note": "SetFit fine-tuned BGE; same support and validation; BANKING test already used by matrix.",
            }
        )
    with (RESULTS / "gate3_setfit_threshold.json").open("w", encoding="utf-8") as handle:
        json.dump({"tau": tau, "selection": selected, "validation": val_metrics}, handle, ensure_ascii=False, indent=2)

    test_rows = test_known_rows + test_unknown_rows
    with (RESULTS / "gate3_setfit_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["sample_id", "role", "true_label", "pred_label", "accepted", "max_prob", "tau"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(test_rows):
            true_label = row["open_set_label"] if row["role"] == "known_test" else "OOS"
            pred_label = classes[pred[i]] if pred[i] >= 0 else "OOS"
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "role": row["role"],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "accepted": int(accept[i]),
                    "max_prob": round(float(test_p[i].max()), 8),
                    "tau": round(tau, 8),
                }
            )
    print(json.dumps({"validation": val_metrics, "test": test_metrics, "train_seconds": train_seconds}, ensure_ascii=False, indent=2))
    print("GATE3_SETFIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
