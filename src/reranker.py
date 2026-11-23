from __future__ import annotations

import os
from typing import Any

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def load_reranker(model_name: str | None = None):
    from sentence_transformers import CrossEncoder

    selected_model = model_name or os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    return CrossEncoder(selected_model)


def rerank(
    query: str,
    fused_results: list[dict[str, Any]],
    chunk_map: dict[int, dict[str, Any]],
    model: Any,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if not query.strip() or not fused_results:
        return []

    pairs: list[tuple[str, str]] = []
    valid_items: list[dict[str, Any]] = []
    for item in fused_results:
        chunk_id = int(item["chunk_id"])
        if chunk_id not in chunk_map:
            raise KeyError(f"Reranker 输入 chunk_id={chunk_id} 不存在于 chunk_map")
        chunk_text = str(chunk_map[chunk_id].get("text", ""))
        if not chunk_text.strip():
            continue
        pairs.append((query, chunk_text))
        valid_items.append(item)

    if not pairs:
        return []

    scores = model.predict(pairs)
    if len(scores) != len(valid_items):
        raise RuntimeError("Reranker 返回分数数量与输入 pair 数量不一致")

    reranked = [
        {**item, "rerank_score": float(score)}
        for item, score in zip(valid_items, scores)
    ]
    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_k]
