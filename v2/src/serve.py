"""Optional FastAPI app exposing POST /classify.  [Phase 6, optional]

Kept import-light: FastAPI is an optional dependency, imported lazily inside ``create_app`` — so
``import src.serve`` works even when fastapi is not installed. Inject a ``pipeline`` for offline tests;
otherwise one is built from ``cfg`` (which constructs a real ``JinaClient``, so set ``JINA_API_KEY``).

    POST /classify  {query, history?, previous_intent?}
        -> {intent, confidence, source, is_oos}   source in {exact_cache, semantic_cache, head, llm_fallback}

Run:  uvicorn src.serve:app --host 0.0.0.0 --port 8080      (after `app = create_app()`)
"""

from __future__ import annotations

from typing import Any


def create_app(cfg: dict[str, Any] | None = None, pipeline: Any | None = None) -> Any:
    """Build the FastAPI app around ``src.pipeline.Pipeline`` (fastapi imported lazily)."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    from src.config import load_config
    from src.pipeline import Pipeline

    cfg = cfg or load_config()
    engine = pipeline if pipeline is not None else Pipeline(cfg)

    app = FastAPI(title="Semantic-Cache Intent Classifier")

    class ClassifyRequest(BaseModel):
        query: str
        history: list[str] | None = None
        previous_intent: str | None = None

    @app.post("/classify")
    def classify(req: ClassifyRequest) -> dict[str, Any]:
        result = engine.classify(req.query, history=req.history, previous_intent=req.previous_intent)
        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "source": result.source,
            "is_oos": result.is_oos,
        }

    return app
