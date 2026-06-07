"""Train entry point for both approaches.

Examples
--------
    # Approach A — frozen encoder + LR head (fast baseline)
    python -m src.train --approach frozen

    # Approach B — SetFit contrastive fine-tune + LR head (the "fine-tune")
    python -m src.train --approach setfit

The script reads processed splits from `data/processed/{train,dev}.parquet`
(produced by `scripts/prepare_data.py`) and writes:
    models/classifier.joblib                 (LR head + label_map + meta)
    models/encoder_finetuned/  (setfit only)  (fine-tuned SentenceTransformer)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .classifier import save_head, train_head
from .config import load_config, repo_path
from .data import LabelMap
from .encoder import Encoder


def _load_split(processed_dir: Path, name: str) -> pd.DataFrame:
    path = processed_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"missing split {path}. Run scripts/prepare_data.py first."
        )
    return pd.read_parquet(path)


def train_frozen(cfg: dict, processed_dir: Path, out_dir: Path) -> None:
    """Approach A: encode once with the frozen encoder, fit LR head."""
    train_df = _load_split(processed_dir, "train")
    label_map = LabelMap.from_labels(train_df["label"])
    print(f"[train] approach=frozen  train_n={len(train_df)}  classes={label_map.num_classes}")

    enc = Encoder(cfg)
    print(f"[train] encoder={enc.model_name}  dim={enc.dim}  max_seq_len={enc.max_seq_length}")

    X_train = enc.embed(train_df["text"].tolist(), show_progress_bar=True)
    y_train = label_map.encode(train_df["label"])

    bundle = train_head(
        X_train,
        y_train,
        label_map,
        cfg,
        meta={
            "approach": "frozen",
            "encoder_model": enc.model_name,
            "encoder_path": enc.model_name,  # HF repo id; no local fine-tuned dir
            "dim": int(enc.dim),
        },
    )
    classifier_path = out_dir / Path(cfg["paths"]["classifier"]).name
    save_head(bundle, classifier_path)
    print(f"[train] head saved → {classifier_path}")


def train_setfit_approach(cfg: dict, processed_dir: Path, out_dir: Path, args) -> None:
    """Approach B: SetFit fine-tune the encoder, then fit LR head."""
    from .setfit_train import train_setfit  # local import — heavy deps

    train_df = _load_split(processed_dir, "train")
    try:
        dev_df = _load_split(processed_dir, "dev")
    except FileNotFoundError:
        dev_df = None

    label_map = LabelMap.from_labels(train_df["label"])
    print(
        f"[train] approach=setfit  train_n={len(train_df)}  "
        f"dev_n={len(dev_df) if dev_df is not None else 0}  classes={label_map.num_classes}"
    )

    encoder_out_dir = out_dir / "encoder_finetuned"
    classifier_path = out_dir / Path(cfg["paths"]["classifier"]).name

    # Resolve hyperparameters: CLI flag (if passed) overrides config.setfit.* default.
    scfg = cfg.get("setfit", {})
    backbone = args.backbone if args.backbone is not None else scfg.get("backbone")
    num_epochs = args.epochs if args.epochs is not None else scfg.get("num_epochs", 1)
    num_iterations = args.iters if args.iters is not None else scfg.get("num_iterations", 20)
    batch_size = args.batch_size if args.batch_size is not None else scfg.get("batch_size", 16)

    train_setfit(
        train_texts=train_df["text"].tolist(),
        train_y=label_map.encode(train_df["label"]),
        dev_texts=dev_df["text"].tolist() if dev_df is not None else None,
        dev_y=label_map.encode(dev_df["label"]) if dev_df is not None else None,
        label_map=label_map,
        cfg=cfg,
        encoder_out_dir=encoder_out_dir,
        classifier_out_path=classifier_path,
        backbone=backbone,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        batch_size=batch_size,
        body_learning_rate=scfg.get("body_learning_rate", 2e-5),
        head_learning_rate=scfg.get("head_learning_rate", 1e-2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Approach A (frozen) or B (SetFit).")
    parser.add_argument("--approach", choices=["frozen", "setfit"], required=True)
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Where train/dev/test parquet files live (default: config.paths.data_processed)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to write models (default: config.paths.models)",
    )
    # SetFit-specific knobs (ignored for frozen). Defaults to None → fall back
    # to config.setfit.* so there are no magic numbers in the CLI layer.
    parser.add_argument("--backbone", type=str, default=None, help="SetFit: override config.setfit.backbone")
    parser.add_argument("--epochs", type=int, default=None, help="SetFit: override config.setfit.num_epochs")
    parser.add_argument("--iters", type=int, default=None, help="SetFit: override config.setfit.num_iterations")
    parser.add_argument("--batch-size", type=int, default=None, help="SetFit: override config.setfit.batch_size")
    # Reranker (parallel second stage). Trains on top of whichever stage-1 ran.
    parser.add_argument(
        "--with-reranker",
        action="store_true",
        help="Also train the cross-encoder reranker (default: config.reranker.enabled)",
    )
    parser.add_argument("--reranker-model", type=str, default=None, help="Reranker: override config.reranker.model_name")
    parser.add_argument("--reranker-epochs", type=int, default=None, help="Reranker: override config.reranker.num_epochs")
    parser.add_argument("--reranker-batch-size", type=int, default=None, help="Reranker: override config.reranker.batch_size")
    args = parser.parse_args()

    cfg = load_config()
    processed_dir = repo_path(args.processed_dir) if args.processed_dir else repo_path(cfg["paths"]["data_processed"])
    out_dir = repo_path(args.out_dir) if args.out_dir else repo_path(cfg["paths"]["models"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: the embedding classifier (frozen or setfit).
    if args.approach == "frozen":
        train_frozen(cfg, processed_dir, out_dir)
    else:
        train_setfit_approach(cfg, processed_dir, out_dir, args)

    # Stage 2 (optional, parallel): the cross-encoder reranker, layered on top.
    if args.with_reranker or cfg.get("reranker", {}).get("enabled", False):
        from .reranker import train_reranker

        print("\n[train] training reranker (stage 2) on top of stage-1 ...")
        train_reranker(
            cfg,
            processed_dir,
            out_dir,
            base_model=args.reranker_model,
            num_epochs=args.reranker_epochs,
            batch_size=args.reranker_batch_size,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
