from __future__ import annotations

from typing import Any

class RAGPipeline:
    """Application-level flow: validate -> retrieve -> filter -> generate -> cite."""

    def __init__(self, searcher: Any, generator: Any):
        self.searcher = searcher
        self.generator = generator

    @staticmethod
    def _filter_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only unique results with a valid ID and non-empty source text."""
        valid_results: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                chunk_id = int(item["chunk_id"])
            except (KeyError, TypeError, ValueError):
                continue
            text = item.get("text")
            source = item.get("source")
            page_value = item.get("start_page", item.get("page"))
            if not isinstance(text, str) or not text.strip():
                continue
            if not isinstance(source, str) or not source.strip():
                continue
            try:
                start_page = int(page_value)
            except (TypeError, ValueError):
                continue
            if start_page <= 0 or chunk_id in seen_ids:
                continue
            cleaned = {
                **item,
                "chunk_id": chunk_id,
                "text": text.strip(),
                "source": source.strip(),
                "start_page": start_page,
            }
            valid_results.append(cleaned)
            seen_ids.add(chunk_id)
        return valid_results

    @staticmethod
    def _build_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for chunk in chunks:
            source = str(chunk.get("source", "")).strip()
            page_value = chunk.get("start_page", chunk.get("page"))
            if not source or page_value is None:
                continue
            try:
                page = int(page_value)
            except (TypeError, ValueError):
                continue
            key = (source, page)
            if key in seen:
                continue
            sources.append({"source": source, "page": page})
            seen.add(key)
        return sources

    def run(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空")
        normalized_query = query.strip()

        retrieved_results = self.searcher.search(normalized_query)
        if not isinstance(retrieved_results, list):
            raise RuntimeError("检索模块必须返回 list")

        valid_chunks = self._filter_results(retrieved_results)
        if not valid_chunks:
            return {"answer": "未找到相关内容。", "sources": []}

        answer = self.generator.generate(normalized_query, valid_chunks)
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Generation 返回了空答案")

        sources = self._build_sources(valid_chunks)
        return {"answer": answer.strip(), "sources": sources}


def build_pipeline() -> RAGPipeline:
    """Load heavyweight retrieval models once when the API application starts."""
    from src.generation import GenerationService
    from src.hybrid_search import HybridSearch

    generator = GenerationService()
    searcher = HybridSearch()
    return RAGPipeline(searcher=searcher, generator=generator)


def run_pipeline(query: str, pipeline: RAGPipeline) -> dict[str, Any]:
    """Small functional entry point for scripts or external callers."""
    return pipeline.run(query)
