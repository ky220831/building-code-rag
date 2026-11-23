from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PAGE_RE = re.compile(r"^========== PAGE (\d+) ==========\s*$")


def parse_pages(text: str) -> list[dict]:
    pages: list[dict] = []
    current_page: int | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        match = PAGE_RE.match(line.strip())
        if match:
            if current_page is not None or current_lines:
                pages.append({
                    "page": current_page if current_page is not None else 0,
                    "content": "\n".join(current_lines).strip(),
                })
            current_page = int(match.group(1))
            current_lines = []
        else:
            current_lines.append(line)

    if current_page is not None or current_lines:
        pages.append({
            "page": current_page if current_page is not None else 0,
            "content": "\n".join(current_lines).strip(),
        })
    return pages


def split_paragraphs(page_content: str) -> list[str]:
    parts = re.split(r"\n\s*\n", page_content)
    return [part.strip() for part in parts if part.strip()]


def split_by_paragraphs(
    pages: list[dict], chunk_size: int = 800, chunk_overlap: int = 100
) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须满足 0 <= overlap < chunk_size")

    chunks: list[dict] = []
    chunk_id = 0
    slide_step = chunk_size - chunk_overlap

    def append_chunk(text: str, start_page: int, end_page: int) -> None:
        nonlocal chunk_id
        cleaned = text.strip()
        chunks.append({
            "chunk_id": chunk_id,
            "start_page": start_page,
            "end_page": end_page,
            "text": cleaned,
            "char_count": len(cleaned),
        })
        chunk_id += 1

    for page_info in pages:
        page_number = int(page_info["page"])
        paragraphs = split_paragraphs(str(page_info.get("content", "")))
        if not paragraphs:
            append_chunk("", page_number, page_number)
            continue

        buffer = ""
        buffer_start_page = page_number
        for paragraph in paragraphs:
            if len(paragraph) > chunk_size:
                if buffer:
                    append_chunk(buffer, buffer_start_page, page_number)
                    buffer = ""
                start = 0
                while start < len(paragraph):
                    end = min(start + chunk_size, len(paragraph))
                    append_chunk(paragraph[start:end], page_number, page_number)
                    if end >= len(paragraph):
                        break
                    start += slide_step
                continue

            candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    append_chunk(buffer, buffer_start_page, page_number)
                buffer = paragraph
                buffer_start_page = page_number

        if buffer:
            append_chunk(buffer, buffer_start_page, page_number)

    return chunks


def save_jsonl(chunks: list[dict], output_path: Path, source_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            record = {
                "source": source_name,
                "chunk_id": chunk["chunk_id"],
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
                "text": chunk["text"],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 文本分块器")
    parser.add_argument("txt_file", help="输入 TXT 路径")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    txt_path = Path(args.txt_file)
    if not txt_path.exists():
        parser.error(f"文件不存在：{txt_path}")
    if txt_path.suffix.lower() != ".txt":
        parser.error("输入文件后缀必须为 .txt")

    pages = parse_pages(txt_path.read_text(encoding="utf-8"))
    chunks = split_by_paragraphs(pages, args.chunk_size, args.overlap)
    output_path = txt_path.with_suffix(".chunks.jsonl")
    save_jsonl(chunks, output_path, source_name=txt_path.name)

    non_empty = [chunk["char_count"] for chunk in chunks if chunk["char_count"] > 0]
    print(f"输入文件：{txt_path.resolve()}")
    print(f"输出文件：{output_path.resolve()}")
    print(f"识别页面：{len(pages)} | Chunk 总数：{len(chunks)} | 非空 Chunk：{len(non_empty)}")
    if non_empty:
        print(f"最短：{min(non_empty)} | 最长：{max(non_empty)} | 平均：{sum(non_empty) / len(non_empty):.1f}")


if __name__ == "__main__":
    main()
