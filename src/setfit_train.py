"""Approach B: SetFit — contrastively fine-tune the sentence encoder on your
intents, then fit a downstream LR head.

Why SetFit for this project:
    - Designed for few-shot text classification — exactly the regime of the
      8-example minority intents in the user's dataset.
    - Encoder fine-tune happens here, off the hot path. At inference time
      we use the *fine-tuned* encoder identically to Approach A (just a
      forward pass + the LR head). Latency budget is unaffected.

After training, the fine-tuned encoder is saved to disk so `Encoder` can
load it through the same SentenceTransformer interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

from .classifier import HeadBundle, train_head
from .data import LabelMap
from .encoder import Encoder


def _to_hf(texts: list[str], y: np.ndarray) -> Dataset:
    return Dataset.from_dict({"text": texts, "label": y.tolist()})


def train_setfit(
    train_texts: list[str],
    train_y: np.ndarray,
    dev_texts: list[str] | None,
    dev_y: np.ndarray | None,
    label_map: LabelMap,
    cfg: dict[str, Any],
    encoder_out_dir: Path,
    classifier_out_path: Path,
    *,
    backbone: str | None = None,
    num_epochs: int = 1,
    batch_size: int = 16,
    num_iterations: int = 20,
    body_learning_rate: float = 2e-5,
    head_learning_rate: float = 1e-2,
    seed: int | None = None,
) -> tuple[Path, HeadBundle]:
    """Fine-tune the encoder with SetFit, then fit an LR head on top.

    Returns the path to the saved fine-tuned encoder and the HeadBundle.
    The encoder is saved with SentenceTransformer's `save()`, so it can be
    loaded by `Encoder(cfg, model_name_or_path=encoder_out_dir)`.
    """
    seed = seed if seed is not None else cfg["training"].get("seed", 42)
    backbone = backbone or cfg["encoder"]["model_name"]

    train_ds = _to_hf(train_texts, train_y)
    eval_ds = _to_hf(dev_texts, dev_y) if dev_texts is not None and dev_y is not None else None

    model = SetFitModel.from_pretrained(backbone)

    args = TrainingArguments(
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        body_learning_rate=body_learning_rate,
        head_learning_rate=head_learning_rate,
        seed=seed,
        end_to_end=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print(
        f"[setfit] backbone={backbone}  train_n={len(train_ds)}  "
        f"eval_n={len(eval_ds) if eval_ds else 0}  epochs={num_epochs}  iters={num_iterations}"
    )
    trainer.train()

    # Save the fine-tuned SentenceTransformer body so it can be loaded
    # identically by our `Encoder` (keeps inference path uniform with A).
    encoder_out_dir = Path(encoder_out_dir)
    encoder_out_dir.mkdir(parents=True, exist_ok=True)
    model.model_body.save(str(encoder_out_dir))
    print(f"[setfit] fine-tuned encoder saved → {encoder_out_dir}")

    # Refit our own LR head on top of the fine-tuned encoder, with
    # class_weight=balanced (SetFit's default head doesn't propagate it).
    enc = Encoder(cfg, model_name_or_path=encoder_out_dir)
    X_train = enc.embed(train_texts, show_progress_bar=True)

    bundle = train_head(
        X_train,
        train_y,
        label_map,
        cfg,
        meta={
            "approach": "setfit",
            "backbone": backbone,
            "encoder_path": str(encoder_out_dir),
            "dim": int(enc.dim),
            "num_epochs": num_epochs,
            "num_iterations": num_iterations,
        },
    )

    from .classifier import save_head

    save_head(bundle, classifier_out_path)
    print(f"[setfit] LR head saved → {classifier_out_path}")

    return encoder_out_dir, bundle
