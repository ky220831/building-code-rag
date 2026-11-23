from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_chunks(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Chunk 文件不存在：{jsonl_path}")

    chunks: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_number} 行格式错误") from exc
            if not isinstance(item, dict) or "chunk_id" not in item or "text" not in item:
                raise ValueError(f"JSONL 第 {line_number} 行缺少 chunk_id 或 text")
            try:
                chunk_id = int(item["chunk_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"JSONL 第 {line_number} 行 chunk_id 非法") from exc
            if chunk_id in seen_ids:
                raise ValueError(f"chunk_id 重复：{chunk_id}")
            item["chunk_id"] = chunk_id
            item["text"] = str(item.get("text", ""))
            seen_ids.add(chunk_id)
            chunks.append(item)
    return chunks


def build_chunk_map(chunks: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    chunk_map: dict[int, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = int(chunk["chunk_id"])
        if chunk_id in chunk_map:
            raise ValueError(f"chunk_id 重复：{chunk_id}")
        chunk_map[chunk_id] = chunk
    return chunk_map


def hydrate_results(
    results: list[dict[str, Any]],
    chunk_map: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    hydrated: list[dict[str, Any]] = []
    for item in results:
        chunk_id = int(item["chunk_id"])
        if chunk_id not in chunk_map:
            raise KeyError(f"检索结果中的 chunk_id={chunk_id} 不存在于 chunk_map，拒绝错误映射")
        hydrated.append({**chunk_map[chunk_id], **item})
    return hydrated
