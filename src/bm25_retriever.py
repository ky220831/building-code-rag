from __future__ import annotations

from typing import Any

import jieba
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Chinese-aware BM25 retriever; results carry IDs and BM25 scores."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = [chunk for chunk in chunks if str(chunk.get("text", "")).strip()]
        if not self.chunks:
            raise ValueError("没有非空 chunk，无法建立 BM25 索引")

        tokenized_pairs = []
        for chunk in self.chunks:
            tokens = self.tokenize(chunk["text"])
            if tokens:
                tokenized_pairs.append(
                    (int(chunk["chunk_id"]), str(chunk.get("source", "")).strip(), tokens)
                )
        if not tokenized_pairs:
            raise ValueError("Chunk 分词后全部为空，无法建立 BM25 索引")

        self.chunk_ids = [item[0] for item in tokenized_pairs]
        self.chunk_sources = [item[1] for item in tokenized_pairs]
        self.bm25 = BM25Okapi([item[2] for item in tokenized_pairs])

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [token.strip() for token in jieba.lcut(text) if token.strip()]

    def search(
        self,
        query: str,
        top_k: int = 20,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not isinstance(query, str) or not query.strip():
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        target_source = source_filter.strip() if isinstance(source_filter, str) and source_filter.strip() else None
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np_argsort_descending(scores)
        results: list[dict[str, Any]] = []
        for index in ranked_indices:
            if target_source and self.chunk_sources[index] != target_source:
                continue
            score = float(scores[index])
            if score <= 0:
                continue
            results.append({
                "chunk_id": self.chunk_ids[index],
                "bm25_score": score,
                "bm25_rank": len(results) + 1,
            })
            if len(results) >= top_k:
                break
        return results


def np_argsort_descending(values: Any) -> list[int]:
    """Avoid exposing NumPy-specific objects in the retriever's return type."""
    return sorted(range(len(values)), key=lambda index: float(values[index]), reverse=True)
