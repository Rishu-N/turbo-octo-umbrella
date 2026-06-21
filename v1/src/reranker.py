"""Cross-encoder reranker — the optional, switchable second stage.

Stage 1 (encoder + LR head, see predictor.py) proposes the top-K intents.
This module re-scores those K candidates with a **cross-encoder**: a transformer
that reads `(query, intent_profile)` *jointly*, which a bi-encoder structurally
cannot. That joint attention is what helps pull apart confusable look-alikes
("card declined" vs "card not working") — at the cost of one transformer pass
per candidate, so it is layered *on top of* stage 1 rather than replacing it.

Design notes
------------
- It works on top of EITHER approach (frozen or setfit). That is the "parallel"
  property: you can test {frozen, setfit} x {rerank off, rerank on}.
- Each intent is described by a short "profile" = humanized label + a few sampled
  training utterances. Profiles are built at train time and saved alongside the
  cross-encoder, so inference needs no extra inputs.
- Hard negatives for training are mined from stage-1's own top-K (the intents it
  actually confuses), which matches the inference-time candidate distribution.
- `selective` reranking only fires when stage-1 is uncertain (small top1-top2
  margin), which keeps the latency cost off the easy majority of requests.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import LabelMap
from .predictor import Stage1Predictor


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def humanize(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").strip()


def build_rerank_text(query: str, history: str, cfg: dict[str, Any]) -> str:
    """Text fed to the cross-encoder as the 'query' side (no e5 prefix — that's
    only for the bi-encoder). Optionally prepend history."""
    query = (query or "").strip()
    if cfg["reranker"].get("use_history", False):
        history = (history or "").strip()
        if history:
            return f"{history} [SEP] {query}"
    return query


def build_intent_profiles(
    train_df: pd.DataFrame, cfg: dict[str, Any], seed: int = 42
) -> dict[str, str]:
    """One short descriptive text per intent: humanized label + sampled exemplars."""
    rng = random.Random(seed)
    n_ex = int(cfg["reranker"].get("exemplars_per_intent", 5))
    profiles: dict[str, str] = {}
    for label, grp in train_df.groupby("label"):
        exemplars = grp["query"].astype(str).tolist()
        rng.shuffle(exemplars)
        ex = [e for e in exemplars[:n_ex] if e.strip()]
        text = humanize(str(label))
        if ex:
            text = f"{text}. Examples: " + " | ".join(ex)
        profiles[str(label)] = text
    return profiles


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_reranker(
    cfg: dict[str, Any],
    processed_dir: Path | str,
    model_dir: Path | str,
    *,
    base_model: str | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
) -> Path:
    """Fine-tune a cross-encoder to rerank stage-1's top-K candidates.

    Requires stage-1 (classifier.joblib) to already be trained in `model_dir`,
    because hard negatives are mined from stage-1's top-K predictions.
    Returns the saved cross-encoder directory.
    """
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    rcfg = cfg["reranker"]
    base_model = base_model or rcfg["model_name"]
    num_epochs = num_epochs if num_epochs is not None else int(rcfg.get("num_epochs", 1))
    batch_size = batch_size if batch_size is not None else int(rcfg.get("batch_size", 16))
    top_k = int(rcfg.get("top_k", 5))
    num_neg = int(rcfg.get("num_negatives", 4))
    max_length = int(rcfg.get("max_length", 128))
    seed = seed if seed is not None else int(cfg["training"].get("seed", 42))

    processed_dir = Path(processed_dir)
    model_dir = Path(model_dir)
    train_df = pd.read_parquet(processed_dir / "train.parquet")

    profiles = build_intent_profiles(train_df, cfg, seed=seed)

    # Stage-1 model (must exist) for hard-negative mining.
    predictor = Stage1Predictor(cfg, model_dir)
    class_names = predictor.label_map.class_names
    print(f"[reranker] mining hard negatives from stage-1 (n={len(train_df)})")
    proba = predictor.predict_proba(train_df["text"].tolist(), show_progress_bar=True)
    topk = Stage1Predictor.topk_indices(proba, max(top_k, num_neg + 1))

    rng = random.Random(seed)
    examples = []
    queries = [
        build_rerank_text(q, h, cfg) for q, h in zip(train_df["query"], train_df["history"])
    ]
    labels = train_df["label"].astype(str).tolist()

    for i, (q, true) in enumerate(zip(queries, labels)):
        # positive pair
        examples.append(InputExample(texts=[q, profiles[true]], label=1.0))
        # hard negatives = wrong intents stage-1 ranked highly for this query
        negs = [class_names[j] for j in topk[i] if class_names[j] != true][:num_neg]
        while len(negs) < num_neg and len(class_names) > 1:
            cand = class_names[rng.randrange(len(class_names))]
            if cand != true and cand not in negs:
                negs.append(cand)
        for ng in negs:
            examples.append(InputExample(texts=[q, profiles.get(ng, humanize(ng))], label=0.0))

    rng.shuffle(examples)
    print(
        f"[reranker] base={base_model}  pairs={len(examples)}  "
        f"epochs={num_epochs}  batch={batch_size}  top_k={top_k}  neg={num_neg}"
    )

    model = CrossEncoder(base_model, num_labels=1, max_length=max_length)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    warmup = max(1, int(len(loader) * num_epochs * 0.1))

    rerank_dir = model_dir / Path(cfg["paths"]["reranker_dir"]).name
    model.fit(
        train_dataloader=loader,
        epochs=num_epochs,
        warmup_steps=warmup,
        output_path=str(rerank_dir),
        show_progress_bar=True,
    )
    print(f"[reranker] cross-encoder saved → {rerank_dir}")

    # Persist intent profiles + label map + meta next to the model.
    cand_path = model_dir / Path(cfg["paths"]["reranker_candidates"]).name
    cand_path.write_text(
        json.dumps(
            {
                "profiles": profiles,
                "label_map": predictor.label_map.to_dict(),
                "meta": {
                    "base_model": base_model,
                    "top_k": top_k,
                    "num_negatives": num_neg,
                    "exemplars_per_intent": int(rcfg.get("exemplars_per_intent", 5)),
                    "use_history": bool(rcfg.get("use_history", False)),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[reranker] intent profiles → {cand_path}")
    return rerank_dir


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
class Reranker:
    def __init__(self, model, profiles: dict[str, str], label_map: LabelMap, cfg: dict[str, Any], meta: dict):
        self.model = model
        self.profiles = profiles
        self.label_map = label_map
        self.cfg = cfg
        self.meta = meta

    @classmethod
    def load(cls, cfg: dict[str, Any], model_dir: Path | str) -> "Reranker":
        from sentence_transformers import CrossEncoder

        model_dir = Path(model_dir)
        rerank_dir = model_dir / Path(cfg["paths"]["reranker_dir"]).name
        cand_path = model_dir / Path(cfg["paths"]["reranker_candidates"]).name
        if not rerank_dir.exists() or not cand_path.exists():
            raise FileNotFoundError(
                f"reranker not found in {model_dir}. Train it with "
                "`python -m src.train --approach <a> --with-reranker`."
            )
        model = CrossEncoder(str(rerank_dir), max_length=int(cfg["reranker"].get("max_length", 128)))
        blob = json.loads(cand_path.read_text())
        return cls(
            model=model,
            profiles=blob["profiles"],
            label_map=LabelMap.from_dict(blob["label_map"]),
            cfg=cfg,
            meta=blob.get("meta", {}),
        )

    def _candidate_text(self, intent_id: int) -> str:
        name = self.label_map.id_to_label[int(intent_id)]
        return self.profiles.get(name, humanize(name))

    def rerank_row(
        self, query: str, candidate_ids: np.ndarray, stage1_probs_row: np.ndarray
    ) -> tuple[int, np.ndarray]:
        """Re-score the candidate intents for one query. Returns (best_id, final_scores)."""
        candidate_ids = np.asarray(candidate_ids)
        pairs = [[query, self._candidate_text(c)] for c in candidate_ids]
        rr = np.asarray(self.model.predict(pairs)).reshape(-1)

        rr_soft = _softmax(rr)
        s1 = stage1_probs_row[candidate_ids]
        s1_soft = s1 / (s1.sum() + 1e-12)

        w = float(self.cfg["reranker"].get("score_weight", 0.5))
        final = w * s1_soft + (1.0 - w) * rr_soft
        best = int(candidate_ids[int(np.argmax(final))])
        return best, final


def rerank_predictions(
    reranker: Reranker,
    queries: list[str],
    proba: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, int]:
    """Apply reranking across a whole split. Returns (y_pred, n_reranked).

    `n_reranked` is how many rows actually went through the cross-encoder (the
    rest were confident enough to keep stage-1's answer under `selective`).
    """
    rcfg = cfg["reranker"]
    top_k = int(rcfg.get("top_k", 5))
    selective = bool(rcfg.get("selective", True))
    margin = float(rcfg.get("uncertainty_margin", 0.15))

    y_pred = proba.argmax(axis=1).copy()
    topk = Stage1Predictor.topk_indices(proba, top_k)
    n_reranked = 0

    for i in range(len(queries)):
        row = proba[i]
        if selective:
            top2 = np.sort(row)[::-1][:2]
            gap = float(top2[0] - (top2[1] if top2.size > 1 else 0.0))
            if gap >= margin:
                continue  # stage-1 is confident; skip the expensive reranker
        best, _ = reranker.rerank_row(queries[i], topk[i], row)
        y_pred[i] = best
        n_reranked += 1

    return y_pred, n_reranked
