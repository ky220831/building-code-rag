from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from sentence_transformers import SentenceTransformer

from src.bm25_retriever import BM25Retriever
from src.common import build_chunk_map, hydrate_results, load_chunks
from src.config import settings
from src.reranker import load_reranker, rerank
from src.rrf_fusion import reciprocal_rank_fusion
from src.vector_retriever import create_vector_retriever


class HybridSearch:
    def __init__(
        self,
        chunks_path: Path | None = None,
        embeddings_path: Path | None = None,
        embedding_model_name: str | None = None,
        reranker_model_name: str | None = None,
        use_reranker: bool | None = None,
    ):
        self.chunks_path = chunks_path or settings.chunks_path
        self.embeddings_path = embeddings_path or settings.embeddings_path
        self.chunks = load_chunks(self.chunks_path)
        self.chunk_map = build_chunk_map(self.chunks)
        if not self.chunk_map:
            raise ValueError("Chunk 文件中没有记录")

        self.embedding_model = SentenceTransformer(embedding_model_name or settings.embedding_model)
        self.vector_retriever = create_vector_retriever(settings.vector_backend, self.embeddings_path, self.embedding_model)
        embedded_ids = set(int(value) for value in self.vector_retriever.chunk_ids.tolist())
        unknown_ids = embedded_ids - set(self.chunk_map)
        if unknown_ids:
            raise ValueError(f"Embedding 中存在 Chunk 文件没有的 chunk_id：{sorted(unknown_ids)[:10]}")

        self.bm25_retriever = BM25Retriever(self.chunks)
        enable_reranker = settings.use_reranker if use_reranker is None else use_reranker
        self.reranker = load_reranker(reranker_model_name) if enable_reranker else None

    def search(
        self,
        query: str,
        vector_top_k: int | None = None,
        bm25_top_k: int | None = None,
        fused_top_k: int | None = None,
        top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            return []

        vector_limit = settings.vector_top_k if vector_top_k is None else vector_top_k
        bm25_limit = settings.bm25_top_k if bm25_top_k is None else bm25_top_k
        fused_limit = settings.fused_top_k if fused_top_k is None else fused_top_k
        final_limit = settings.final_top_k if top_k is None else top_k
        fusion_k = settings.rrf_k if rrf_k is None else rrf_k

        vector_results = self.vector_retriever.search(query, vector_limit)
        bm25_results = self.bm25_retriever.search(query, bm25_limit)
        fused_results = reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            k=fusion_k,
            top_k=fused_limit,
        )

        if not fused_results:
            return []
        if self.reranker is not None:
            final_ranked = rerank(
                query=query,
                fused_results=fused_results,
                chunk_map=self.chunk_map,
                model=self.reranker,
                top_k=final_limit,
            )
        else:
            if final_limit <= 0:
                raise ValueError("top_k 必须大于 0")
            final_ranked = fused_results[:final_limit]
        return hydrate_results(final_ranked, self.chunk_map)

    def list_sources(self) -> list[str]:
        return sorted({
            str(chunk.get("source", ""))
            for chunk in self.chunks
            if str(chunk.get("source", "")).strip()
        })


def print_results(results: list[dict[str, Any]]) -> None:
    if not results:
        print("没有找到相关结果。")
        return
    print(f"\n返回 Top {len(results)} 条结果")
    for index, item in enumerate(results, start=1):
        print("-" * 72)
        print(
            f"【第 {index} 条】chunk_id={item['chunk_id']} | "
            f"来源={item.get('source', '')} | 页码={item.get('start_page')}~{item.get('end_page')}"
        )
        for key in ("vector_score", "bm25_score", "rrf_score", "rerank_score"):
            if key in item:
                print(f"{key}：{float(item[key]):.6f}")
        text = str(item.get("text", ""))
        print(text[:500] + ("..." if len(text) > 500 else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector + BM25 + RRF + Reranker 混合检索")
    parser.add_argument("query", help="用户查询")
    parser.add_argument("--chunks", default=str(settings.chunks_path))
    parser.add_argument("--embeddings", default=str(settings.embeddings_path))
    parser.add_argument("--vector-top-k", type=int, default=settings.vector_top_k)
    parser.add_argument("--bm25-top-k", type=int, default=settings.bm25_top_k)
    parser.add_argument("--fused-top-k", type=int, default=settings.fused_top_k)
    parser.add_argument("--top-k", type=int, default=settings.final_top_k)
    parser.add_argument("--no-reranker", action="store_true")
    args = parser.parse_args()

    searcher = HybridSearch(
        chunks_path=Path(args.chunks),
        embeddings_path=Path(args.embeddings),
        use_reranker=not args.no_reranker,
    )
    results = searcher.search(
        args.query,
        vector_top_k=args.vector_top_k,
        bm25_top_k=args.bm25_top_k,
        fused_top_k=args.fused_top_k,
        top_k=args.top_k,
    )
    print_results(results)


if __name__ == "__main__":
    main()
