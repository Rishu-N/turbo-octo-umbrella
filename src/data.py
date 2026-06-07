"""Load the user-supplied banking CSV and turn it into a typed in-memory form.

Schema (from config.yaml `data.columns`):
  - query   : current_user_query   (str)
  - history : conversation_history (str; may be empty/NaN)
  - label   : expected_intent      (str)

Use `load_user_dataset` for the raw frame; use `to_xy` once you have
preprocessed input strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LabelMap:
    """Bidirectional string-label ↔ int mapping. Ordering = sorted unique labels."""

    label_to_id: dict[str, int]
    id_to_label: dict[int, str]

    @classmethod
    def from_labels(cls, labels: list[str] | pd.Series) -> "LabelMap":
        unique = sorted(set(str(x) for x in labels))
        label_to_id = {name: i for i, name in enumerate(unique)}
        id_to_label = {i: name for name, i in label_to_id.items()}
        return cls(label_to_id=label_to_id, id_to_label=id_to_label)

    @property
    def num_classes(self) -> int:
        return len(self.label_to_id)

    @property
    def class_names(self) -> list[str]:
        return [self.id_to_label[i] for i in range(self.num_classes)]

    def encode(self, labels: pd.Series | list[str]) -> np.ndarray:
        return np.array([self.label_to_id[str(x)] for x in labels], dtype=np.int64)

    def decode(self, ids: list[int] | np.ndarray) -> list[str]:
        return [self.id_to_label[int(i)] for i in ids]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_to_id": self.label_to_id,
            "id_to_label": {int(k): v for k, v in self.id_to_label.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabelMap":
        return cls(
            label_to_id=dict(d["label_to_id"]),
            id_to_label={int(k): v for k, v in d["id_to_label"].items()},
        )


def load_user_dataset(csv_path: Path | str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Read the user's CSV and return a normalized DataFrame.

    Columns in the returned frame: 'query', 'history', 'label'.
    The original column names come from cfg['data']['columns'].
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Update config['data']['user_dataset_path'] or pass --data <path>."
        )

    cols = cfg["data"]["columns"]
    qcol, hcol, lcol = cols["query"], cols["history"], cols["label"]

    df = pd.read_csv(csv_path)
    missing = [c for c in (qcol, hcol, lcol) if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {csv_path} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}. "
            "Update config['data']['columns'] if your headers differ."
        )

    out = pd.DataFrame(
        {
            "query": df[qcol].fillna("").astype(str),
            "history": df[hcol].fillna("").astype(str),
            "label": df[lcol].astype(str),
        }
    )

    # Drop rows with empty query or empty label — they're not trainable.
    before = len(out)
    out = out[(out["query"].str.strip() != "") & (out["label"].str.strip() != "")].reset_index(
        drop=True
    )
    dropped = before - len(out)
    if dropped:
        print(f"[data] dropped {dropped} rows with empty query/label")

    return out


def describe(df: pd.DataFrame) -> None:
    """Print a class-distribution summary — useful for spotting the tiny tail."""
    counts = df["label"].value_counts().sort_values(ascending=False)
    print(f"[data] rows={len(df)}  classes={len(counts)}")
    print(f"[data] min/median/max per class: {counts.min()} / {int(counts.median())} / {counts.max()}")
    print("[data] per-class counts:")
    for name, n in counts.items():
        print(f"        {name:<40} {n}")


def to_xy(df: pd.DataFrame, text_col: str, label_map: LabelMap) -> tuple[list[str], np.ndarray]:
    """Extract (texts, label_ids) given a frame that already has the built input text."""
    return df[text_col].tolist(), label_map.encode(df["label"])
