import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_chunks(jsonl_path: Path) -> list[dict]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def search(chunks, embeddings, query_vec, top_k=5):
    scores = []
    for i, chunk in enumerate(chunks):
        sim = cosine_similarity(query_vec, embeddings[i])
        scores.append((sim, i))

    scores.sort(reverse=True, key=lambda x: x[0])
    results = []
    for sim, idx in scores[:top_k]:
        results.append({
            "chunk_id": chunks[idx]["chunk_id"],
            "source": chunks[idx]["source"],
            "start_page": chunks[idx]["start_page"],
            "end_page": chunks[idx]["end_page"],
            "score": float(sim),
            "text": chunks[idx]["text"]
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="向量检索器")
    parser.add_argument("query", help="查询文本")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    jsonl_path = Path("data/processed/机器学习中的数学.chunks.jsonl")
    emb_path = Path("data/processed/机器学习中的数学.embeddings.npz")

    print("加载chunks...")
    chunks = load_chunks(jsonl_path)

    print("加载向量...")
    data = np.load(emb_path)
    embeddings = data["embeddings"]

    print("加载模型...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    print(f"查询：{args.query}")
    print("正在向量化查询...")
    query_vec = model.encode(args.query)

    results = search(chunks, embeddings, query_vec, args.top_k)

    print("-" * 60)
    print(f"返回 Top {len(results)} 结果：")
    for i, r in enumerate(results, start=1):
        print(f"\n【第{i}条】相似度={r['score']:.4f} | 页 {r['start_page']}~{r['end_page']} | chunk_id={r['chunk_id']}")
        preview = r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
        print(preview)
        print("-" * 40)


if __name__ == "__main__":
    main()
