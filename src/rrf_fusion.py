from __future__ import annotations

from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = 60,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    if k <= 0:
        raise ValueError("RRF 的 k 必须大于 0")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    rrf_scores: dict[int, float] = defaultdict(float)
    detail_scores: dict[int, dict[str, float | int]] = defaultdict(dict)

    for source_name, results, score_key, rank_key in (
        ("vector", vector_results, "vector_score", "vector_rank"),
        ("bm25", bm25_results, "bm25_score", "bm25_rank"),
    ):
        seen_ids: set[int] = set()
        for rank, item in enumerate(results, start=1):
            chunk_id = int(item["chunk_id"])
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            original_rank = int(item.get(rank_key, rank))
            if original_rank <= 0:
                raise ValueError(f"{source_name} rank 必须大于 0")
            rrf_scores[chunk_id] += 1.0 / (k + original_rank)
            if score_key in item:
                detail_scores[chunk_id][score_key] = float(item[score_key])
            detail_scores[chunk_id][rank_key] = original_rank

    fused_results = [
        {"chunk_id": chunk_id, "rrf_score": float(score), **detail_scores[chunk_id]}
        for chunk_id, score in rrf_scores.items()
    ]
    fused_results.sort(key=lambda item: item["rrf_score"], reverse=True)
    return fused_results[:top_k]
