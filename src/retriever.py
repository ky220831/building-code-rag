from pathlib import Path
import argparse
import json


def load_chunks(jsonl_path: Path) -> list[dict]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def count_keyword_hits(text: str, keywords: list[str]) -> int:
    hits = 0
    for kw in keywords:
        if kw in text:
            hits += 1
    return hits


def search(chunks: list[dict], query: str, top_k: int = 5) -> list[dict]:
    keywords = [kw.strip() for kw in query.split() if kw.strip()]

    scored = []
    for chunk in chunks:
        text = chunk.get("text", "")
        score = count_keyword_hits(text, keywords)
        scored.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "score": score,
            "text": text
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = [r for r in scored if r["score"] > 0][:top_k]
    return results


def main():
    parser = argparse.ArgumentParser(description="关键词检索器")
    parser.add_argument("jsonl_file", help="chunks.jsonl 文件路径")
    parser.add_argument("query", help="查询关键词，用空格分隔")
    parser.add_argument("--top-k", type=int, default=5, help="返回前N条结果，默认5")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_file)
    if not jsonl_path.exists():
        print(f"错误：文件不存在 -> {jsonl_path}")
        return

    chunks = load_chunks(jsonl_path)
    print(f"已加载 {len(chunks)} 个chunk")
    print(f"查询：{args.query}")
    print(f"关键词：{[kw.strip() for kw in args.query.split() if kw.strip()]}")
    print("-" * 60)

    results = search(chunks, args.query, args.top_k)

    if not results:
        print("没有找到匹配的结果")
        return

    print(f"返回 Top {len(results)} 结果：")
    for i, r in enumerate(results, start=1):
        print(f"\n【第{i}条】得分={r['score']} | 页 {r['start_page']}~{r['end_page']} | chunk_id={r['chunk_id']}")
        preview = r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
        print(preview)
        print("-" * 40)


if __name__ == "__main__":
    main()
