"""Take the user's CSV, build model-ready input strings, and write stratified
train/dev/test parquet files to `data/processed/`.

Tiny-class safeguard: classes with very few examples (`min_samples_per_split`
from config governs this) get at least that many rows in dev and test. The
default of 2 means: for a class with only 8 rows, you end up with ≥2 in dev
and ≥2 in test (and the rest in train).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, repo_path  # noqa: E402
from src.data import describe, load_user_dataset  # noqa: E402
from src.preprocess import build_inputs  # noqa: E402


def _stratified_split(
    df: pd.DataFrame,
    test_size: float,
    dev_size: float,
    seed: int,
    min_per_split: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-class stratified split with a hard min-per-split floor.

    For each class:
      1. shuffle the rows
      2. peel off `max(min_per_split, round(test_size * n))` for test
      3. peel off `max(min_per_split, round(dev_size * n))` for dev
      4. rest → train
    If a class has fewer than `2 * min_per_split + 1` rows we still try, but
    you may get an empty train slice for that class — we'll warn loudly.
    """
    rng = np.random.default_rng(seed)
    train_parts, dev_parts, test_parts = [], [], []
    warnings: list[str] = []

    for label, grp in df.groupby("label"):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)

        n_test = max(min_per_split, int(round(test_size * n)))
        n_dev = max(min_per_split, int(round(dev_size * n)))
        # Don't drain the class.
        if n_test + n_dev >= n:
            n_test = min(min_per_split, max(1, n // 3))
            n_dev = min(min_per_split, max(1, n // 3))
            if n_test + n_dev >= n:
                warnings.append(
                    f"class={label!r} has only {n} rows; train slice will be tiny"
                )

        test_idx = idx[:n_test]
        dev_idx = idx[n_test : n_test + n_dev]
        train_idx = idx[n_test + n_dev :]

        train_parts.append(grp.loc[train_idx])
        dev_parts.append(grp.loc[dev_idx])
        test_parts.append(grp.loc[test_idx])

    for w in warnings:
        print(f"[split] warning: {w}")

    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    dev = pd.concat(dev_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, dev, test


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare stratified train/dev/test from the user CSV.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the user CSV (defaults to config.data.user_dataset_path)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (defaults to config.paths.data_processed)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config.training.seed")
    args = parser.parse_args()

    cfg = load_config()
    csv_path = repo_path(args.data) if args.data else repo_path(cfg["data"]["user_dataset_path"])
    out_dir = repo_path(args.out) if args.out else repo_path(cfg["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else cfg["training"]["seed"]

    print(f"[prepare] reading {csv_path}")
    df = load_user_dataset(csv_path, cfg)
    describe(df)

    df = build_inputs(df, cfg, out_col="text")

    train, dev, test = _stratified_split(
        df,
        test_size=cfg["training"]["test_size"],
        dev_size=cfg["training"]["dev_size"],
        seed=seed,
        min_per_split=cfg["data"].get("min_samples_per_split", 2),
    )

    for name, frame in [("train", train), ("dev", dev), ("test", test)]:
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        print(f"[prepare] wrote {path}  rows={len(frame)}  classes={frame['label'].nunique()}")

    print("\n[prepare] split sizes by class:")
    sizes = pd.DataFrame(
        {
            "train": train["label"].value_counts(),
            "dev": dev["label"].value_counts(),
            "test": test["label"].value_counts(),
        }
    ).fillna(0).astype(int).sort_values("train", ascending=False)
    print(sizes.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
