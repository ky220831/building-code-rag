from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.common import hydrate_results
from src.config import settings
from src.reranker import rerank
from src.rrf_fusion import reciprocal_rank_fusion


@dataclass
class EvalSummary:
    name: str
    total: int
    hit_at_1: float
    hit_at_5: float
    mrr_at_5: float
    avg_ms: float


def is_chunk_match(chunk: dict[str, Any], question: dict[str, Any]) -> bool:
    """判断单个召回 chunk 是否命中目标关键词（全部 expected_keywords 均出现即算命中）。"""
    text = str(chunk.get("text", ""))
    keywords = question.get("expected_keywords") or []
    expected_pages = question.get("expected_pages") or []

    if expected_pages:
        start_p = int(chunk.get("start_page", -1))
        end_p = int(chunk.get("end_page", start_p))
        if not any(start_p <= int(p) <= end_p for p in expected_pages):
            return False

    if not keywords:
        return bool(expected_pages)
    lowered = text.lower()
    return all(str(kw).lower() in lowered for kw in keywords)


def compute_metrics(
    name: str,
    questions: list[dict[str, Any]],
    search_fn: Callable[[dict[str, Any]], list[dict[str, Any]]],
    top_k: int = 5,
) -> tuple[EvalSummary, list[dict[str, Any]]]:
    """对单组检索策略计算 Hit@1、Hit@5、MRR@5 与平均耗时（ms）。"""
    if not questions:
        raise ValueError("评测集 questions 不能为空")

    hit1_count = 0
    hitk_count = 0
    rr_sum = 0.0
    total_ms = 0.0
    details: list[dict[str, Any]] = []

    for q in questions:
        t0 = time.perf_counter()
        results = search_fn(q)[:top_k]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_ms += elapsed_ms

        first_hit_rank: int | None = None
        for rank, item in enumerate(results, start=1):
            if is_chunk_match(item, q):
                first_hit_rank = rank
                break

        if first_hit_rank == 1:
            hit1_count += 1
        if first_hit_rank is not None:
            hitk_count += 1
            rr_sum += 1.0 / first_hit_rank

        details.append({
            "id": q.get("id"),
            "query": q.get("query"),
            "first_hit_rank": first_hit_rank,
            "latency_ms": round(elapsed_ms, 1),
        })

    n = len(questions)
    summary = EvalSummary(
        name=name,
        total=n,
        hit_at_1=hit1_count / n,
        hit_at_5=hitk_count / n,
        mrr_at_5=rr_sum / n,
        avg_ms=total_ms / n,
    )
    return summary, details


def format_markdown_table(summaries: list[EvalSummary]) -> str:
    lines = [
        "| 检索策略 | 评测题数 | Hit@1 | Hit@5 | MRR@5 | 平均耗时 (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in summaries:
        lines.append(
            f"| {s.name} | {s.total} | {s.hit_at_1 * 100:.1f}% | "
            f"{s.hit_at_5 * 100:.1f}% | {s.mrr_at_5:.3f} | {s.avg_ms:.1f} |"
        )
    return "\n".join(lines)


def run_benchmark(
    questions_path: Path,
    chunks_path: Path,
    embeddings_path: Path,
    backend: str,
    use_source_filter: bool,
    output_md: Path | None,
) -> None:
    from src.hybrid_search import HybridSearch

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    searcher = HybridSearch(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        use_reranker=True,
        vector_backend=backend,
    )

    # 预热一次，避免模型首次前向编译时间干扰首组策略统计
    _ = searcher.search(questions[0]["query"], fused_top_k=5, top_k=5)

    def _src(q: dict[str, Any]) -> str | None:
        return q.get("source") if use_source_filter else None

    strategies: list[tuple[str, Callable[[dict[str, Any]], list[dict[str, Any]]]]] = [
        (
            f"Vector Only ({backend})",
            lambda q: hydrate_results(
                searcher.vector_retriever.search(q["query"], top_k=5, source_filter=_src(q)),
                searcher.chunk_map,
            ),
        ),
        (
            "BM25 Only",
            lambda q: hydrate_results(
                searcher.bm25_retriever.search(q["query"], top_k=5, source_filter=_src(q)),
                searcher.chunk_map,
            ),
        ),
        (
            "Hybrid (Vector+BM25+RRF, 无重排)",
            lambda q: hydrate_results(
                reciprocal_rank_fusion(
                    searcher.vector_retriever.search(q["query"], settings.vector_top_k, source_filter=_src(q)),
                    searcher.bm25_retriever.search(q["query"], settings.bm25_top_k, source_filter=_src(q)),
                    k=settings.rrf_k,
                    top_k=5,
                ),
                searcher.chunk_map,
            ),
        ),
        (
            "Hybrid + Rerank (FUSED_TOP_K=10)",
            lambda q: searcher.search(
                q["query"], fused_top_k=10, top_k=5, source_filter=_src(q)
            ),
        ),
        (
            "Hybrid + Rerank (FUSED_TOP_K=20)",
            lambda q: searcher.search(
                q["query"], fused_top_k=20, top_k=5, source_filter=_src(q)
            ),
        ),
    ]

    summaries: list[EvalSummary] = []
    all_details: dict[str, list[dict[str, Any]]] = {}
    for name, fn in strategies:
        print(f"正在评测：{name} ...")
        summary, details = compute_metrics(name, questions, fn, top_k=5)
        summaries.append(summary)
        all_details[name] = details

    table = format_markdown_table(summaries)
    print("\n" + table + "\n")

    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        report = (
            "# 检索策略离线对照评测报告\n\n"
            f"- 评测集：`{questions_path.as_posix()}`（共 {len(questions)} 题）\n"
            f"- 向量后端：`{backend}` | 元数据过滤：`{'开启' if use_source_filter else '关闭'}`\n"
            f"- Embedding：`{settings.embedding_model}` | Reranker：`{settings.reranker_model}`\n\n"
            "## 指标对比\n\n"
            f"{table}\n"
        )
        output_md.write_text(report, encoding="utf-8")
        print(f"已保存评测报告：{output_md.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="离线对比 Vector / BM25 / Hybrid / Rerank 检索指标与耗时")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--chunks", default=str(settings.chunks_path))
    parser.add_argument("--embeddings", default=str(settings.embeddings_path))
    parser.add_argument("--backend", choices=["numpy", "chroma"], default=settings.vector_backend)
    parser.add_argument("--no-source-filter", action="store_true", help="关闭 source 元数据过滤")
    parser.add_argument("--output", default="docs/eval_retrieval_report.md", help="Markdown 报告输出路径")
    args = parser.parse_args()

    run_benchmark(
        questions_path=Path(args.questions),
        chunks_path=Path(args.chunks),
        embeddings_path=Path(args.embeddings),
        backend=args.backend,
        use_source_filter=not args.no_source_filter,
        output_md=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
