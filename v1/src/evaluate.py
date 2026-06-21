"""Evaluate a trained model on a split.

Reports (stage-1 always; reranked too when --reranker is on):
  - accuracy
  - macro-F1                       ← headline metric (class imbalance)
  - per-intent precision/recall/F1 ← critical for the 8-example tail
  - confusion matrix (CSV)

The reranker is a parallel capability: run with it off and on to A/B test it
against the base approach on the SAME split. Reranked outputs are written with
a `_reranked` suffix so both coexist.

Examples
--------
    python -m src.evaluate --split test                  # stage-1 only
    python -m src.evaluate --split test --reranker on     # stage-1 + reranker, side by side
    python -m src.evaluate --split test --reranker auto   # follow config.reranker.enabled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import load_config, repo_path
from .predictor import Stage1Predictor


def _load_split(processed_dir: Path, name: str) -> pd.DataFrame:
    path = processed_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing split {path}.")
    return pd.read_parquet(path)


def _report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    *,
    split: str,
    report_dir: Path,
    tag: str = "",
    meta: dict | None = None,
) -> dict:
    """Print metrics + confusion matrix and persist them. `tag` suffixes filenames."""
    suffix = f"_{tag}" if tag else ""
    labels = list(range(len(class_names)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        zero_division=0, digits=4, output_dict=True,
    )

    header = f"stage-1 + reranker" if tag == "reranked" else "stage-1"
    print(f"\n[eval] === {header} ===")
    print(f"[eval] accuracy : {acc:.4f}")
    print(f"[eval] macro-F1 : {macro_f1:.4f}")
    print(
        classification_report(
            y_true, y_pred, labels=labels, target_names=class_names,
            zero_division=0, digits=4,
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_path = report_dir / f"confusion_matrix_{split}{suffix}.csv"
    cm_df.to_csv(cm_path)
    print(f"[eval] confusion matrix → {cm_path}")

    pairs = [
        (class_names[i], class_names[j], int(cm[i, j]))
        for i in range(len(class_names))
        for j in range(len(class_names))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda t: -t[2])
    if pairs:
        print("[eval] top confused pairs (true → predicted, count):")
        for true_lbl, pred_lbl, n in pairs[:10]:
            print(f"        {true_lbl:<32} → {pred_lbl:<32}  {n}")

    metrics = {
        "split": split,
        "tag": tag or "stage1",
        "n": int(len(y_true)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        **(meta or {}),
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": int(report[name]["support"]),
            }
            for name in class_names
            if name in report
        },
    }
    metrics_path = report_dir / f"metrics_{split}{suffix}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] metrics → {metrics_path}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier on a split.")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument(
        "--reranker",
        choices=["auto", "on", "off"],
        default="auto",
        help="auto = follow config.reranker.enabled; on/off force it",
    )
    parser.add_argument("--report-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    processed_dir = repo_path(args.processed_dir) if args.processed_dir else repo_path(cfg["paths"]["data_processed"])
    model_dir = repo_path(args.model_dir) if args.model_dir else repo_path(cfg["paths"]["models"])
    report_dir = repo_path(args.report_dir) if args.report_dir else (model_dir / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    df = _load_split(processed_dir, args.split)

    # Stage 1
    predictor = Stage1Predictor(cfg, model_dir)
    print(
        f"[eval] split={args.split}  n={len(df)}  approach={predictor.bundle.meta.get('approach')}  "
        f"encoder={predictor.encoder.model_name}"
    )
    class_names = predictor.label_map.class_names
    y_true = predictor.label_map.encode(df["label"])
    proba = predictor.predict_proba(df["text"].tolist(), show_progress_bar=True)
    y_pred_s1 = proba.argmax(axis=1)

    s1_metrics = _report(
        y_true, y_pred_s1, class_names,
        split=args.split, report_dir=report_dir, tag="",
        meta={"approach": predictor.bundle.meta.get("approach"),
              "encoder_model": predictor.bundle.meta.get("encoder_model") or predictor.bundle.meta.get("backbone")},
    )

    # Decide whether to also run the reranker
    use_reranker = args.reranker == "on" or (args.reranker == "auto" and cfg.get("reranker", {}).get("enabled", False))
    if not use_reranker:
        return 0

    from .reranker import Reranker, build_rerank_text, rerank_predictions

    try:
        reranker = Reranker.load(cfg, model_dir)
    except FileNotFoundError as exc:
        print(f"[eval] reranker requested but not available: {exc}")
        return 1

    queries = [build_rerank_text(q, h, cfg) for q, h in zip(df["query"], df["history"])]
    y_pred_rr, n_reranked = rerank_predictions(reranker, queries, proba, cfg)
    print(
        f"\n[eval] reranker applied to {n_reranked}/{len(df)} rows "
        f"(selective={cfg['reranker'].get('selective')}, margin={cfg['reranker'].get('uncertainty_margin')})"
    )

    rr_metrics = _report(
        y_true, y_pred_rr, class_names,
        split=args.split, report_dir=report_dir, tag="reranked",
        meta={"approach": predictor.bundle.meta.get("approach"),
              "reranker_base": reranker.meta.get("base_model"),
              "rows_reranked": int(n_reranked)},
    )

    # Side-by-side delta — the whole point of the A/B capability.
    d_acc = rr_metrics["accuracy"] - s1_metrics["accuracy"]
    d_f1 = rr_metrics["macro_f1"] - s1_metrics["macro_f1"]
    print("\n[eval] ===== stage-1 vs reranked =====")
    print("[eval]            accuracy   macro-F1")
    print(f"[eval] stage-1     {s1_metrics['accuracy']:.4f}    {s1_metrics['macro_f1']:.4f}")
    print(f"[eval] reranked    {rr_metrics['accuracy']:.4f}    {rr_metrics['macro_f1']:.4f}")
    print(f"[eval] delta       {d_acc:+.4f}    {d_f1:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
