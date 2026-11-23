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

        self.chunk_ids = [int(chunk["chunk_id"]) for chunk in self.chunks]
        tokenized_corpus = [self.tokenize(chunk["text"]) for chunk in self.chunks]
        non_empty_pairs = [
            (chunk_id, tokens)
            for chunk_id, tokens in zip(self.chunk_ids, tokenized_corpus)
            if tokens
        ]
        if not non_empty_pairs:
            raise ValueError("Chunk 分词后全部为空，无法建立 BM25 索引")

        self.chunk_ids = [chunk_id for chunk_id, _ in non_empty_pairs]
        self.bm25 = BM25Okapi([tokens for _, tokens in non_empty_pairs])

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [token.strip() for token in jieba.lcut(text) if token.strip()]

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not isinstance(query, str) or not query.strip():
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np_argsort_descending(scores)
        results: list[dict[str, Any]] = []
        for index in ranked_indices:
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
