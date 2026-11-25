from __future__ import annotations

import logging
import re
import time
from typing import Any, Iterator

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"资料\s*(\d+)")


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
    def _extract_cited_chunks(
        chunks: list[dict[str, Any]], answer: str | None
    ) -> list[dict[str, Any]]:
        """若回答中明确引用了 [资料N] / 【资料N】 / 资料N，则只保留被引用的片段。"""
        if not answer:
            return chunks
        cited_indices: list[int] = []
        seen: set[int] = set()
        for match in CITATION_RE.finditer(answer):
            idx = int(match.group(1))
            if 1 <= idx <= len(chunks) and idx not in seen:
                cited_indices.append(idx)
                seen.add(idx)
        if not cited_indices:
            return chunks
        return [chunks[idx - 1] for idx in cited_indices]

    @classmethod
    def _build_sources(
        cls, chunks: list[dict[str, Any]], answer: str | None = None
    ) -> list[dict[str, Any]]:
        selected = cls._extract_cited_chunks(chunks, answer)
        sources: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for chunk in selected:
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

    def _retrieve(self, normalized_query: str, source_filter: str | None) -> list[dict[str, Any]]:
        normalized_source = (
            source_filter.strip()
            if isinstance(source_filter, str) and source_filter.strip()
            else None
        )
        if normalized_source is not None:
            retrieved_results = self.searcher.search(
                normalized_query, source_filter=normalized_source
            )
        else:
            retrieved_results = self.searcher.search(normalized_query)

        if not isinstance(retrieved_results, list):
            raise RuntimeError("检索模块必须返回 list")
        return self._filter_results(retrieved_results)

    def run(self, query: str, source_filter: str | None = None) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空")
        normalized_query = query.strip()

        t0 = time.perf_counter()
        valid_chunks = self._retrieve(normalized_query, source_filter)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        if not valid_chunks:
            logger.info(
                "RAG(sync) query=%r hits=0 retrieval_ms=%.1f total_ms=%.1f",
                normalized_query[:40],
                retrieval_ms,
                retrieval_ms,
            )
            return {"answer": "未找到相关内容。", "sources": []}

        t_gen = time.perf_counter()
        answer = self.generator.generate(normalized_query, valid_chunks)
        generation_ms = (time.perf_counter() - t_gen) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Generation 返回了空答案")

        cleaned_answer = answer.strip()
        sources = self._build_sources(valid_chunks, cleaned_answer)
        logger.info(
            "RAG(sync) query=%r hits=%d cited=%d retrieval_ms=%.1f generation_ms=%.1f total_ms=%.1f",
            normalized_query[:40],
            len(valid_chunks),
            len(sources),
            retrieval_ms,
            generation_ms,
            total_ms,
        )
        return {"answer": cleaned_answer, "sources": sources}

    def run_stream(
        self, query: str, source_filter: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """流式执行问答，依次产出 delta、sources、done 事件并记录分段耗时。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空")
        normalized_query = query.strip()

        t0 = time.perf_counter()
        valid_chunks = self._retrieve(normalized_query, source_filter)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        if not valid_chunks:
            total_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "RAG(stream) query=%r hits=0 retrieval_ms=%.1f ttft_ms=0.0 total_ms=%.1f",
                normalized_query[:40],
                retrieval_ms,
                total_ms,
            )
            yield {"event": "delta", "data": {"text": "未找到相关内容。"}}
            yield {"event": "sources", "data": {"sources": []}}
            yield {
                "event": "done",
                "data": {
                    "retrieval_ms": round(retrieval_ms, 1),
                    "ttft_ms": 0.0,
                    "total_ms": round(total_ms, 1),
                },
            }
            return

        t_gen = time.perf_counter()
        ttft_ms: float | None = None
        parts: list[str] = []

        if hasattr(self.generator, "generate_stream"):
            token_iter = self.generator.generate_stream(normalized_query, valid_chunks)
        else:
            token_iter = iter([self.generator.generate(normalized_query, valid_chunks)])

        for piece in token_iter:
            if not isinstance(piece, str) or not piece:
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t_gen) * 1000.0
            parts.append(piece)
            yield {"event": "delta", "data": {"text": piece}}

        full_answer = "".join(parts).strip()
        if not full_answer:
            raise RuntimeError("Generation 流式返回为空")

        sources = self._build_sources(valid_chunks, full_answer)
        total_ms = (time.perf_counter() - t0) * 1000.0
        final_ttft = ttft_ms if ttft_ms is not None else 0.0

        logger.info(
            "RAG(stream) query=%r hits=%d cited=%d retrieval_ms=%.1f ttft_ms=%.1f total_ms=%.1f",
            normalized_query[:40],
            len(valid_chunks),
            len(sources),
            retrieval_ms,
            final_ttft,
            total_ms,
        )
        yield {"event": "sources", "data": {"sources": sources}}
        yield {
            "event": "done",
            "data": {
                "retrieval_ms": round(retrieval_ms, 1),
                "ttft_ms": round(final_ttft, 1),
                "total_ms": round(total_ms, 1),
            },
        }


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
