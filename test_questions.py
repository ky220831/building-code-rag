import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(f"【资料{i}】（第{chunk['start_page']}页）\n{chunk['text']}")
    context = "\n\n".join(context_parts)
    prompt = f"""你是一个专业的问答助手。请根据下面提供的资料回答用户的问题。
要求：
1. 只根据资料回答，不要编造
2. 如果资料里没有相关信息，请直接说"未找到相关内容"
3. 回答要简洁准确

【资料】
{context}

【问题】
{query}

请回答："""
    return prompt


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com/v1"
    )

    chunks = []
    with open("data/processed/机器学习中的数学.chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    embeddings = np.load("data/processed/机器学习中的数学.embeddings.npz")["embeddings"]
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    questions = [
        "什么是梯度下降？",
        "线性回归的最小二乘法怎么推导？",
        "什么是主成分分析？",
        "概率论中的贝叶斯定理是什么？",
        "矩阵乘法有什么几何意义？",
    ]

    results = []

    for q in questions:
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print(f"{'='*60}")

        query_vec = model.encode(q)

        scores = []
        for i in range(len(chunks)):
            sim = cosine_similarity(query_vec, embeddings[i])
            scores.append((sim, i))
        scores.sort(reverse=True, key=lambda x: x[0])

        top_chunks = [chunks[idx] for _, idx in scores[:3]]

        prompt = build_prompt(q, top_chunks)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = response.choices[0].message.content

        print(f"回答：{answer}")
        print(f"\n来源：")
        for i, c in enumerate(top_chunks, start=1):
            sim = cosine_similarity(query_vec, embeddings[c['chunk_id']])
            print(f"  资料{i}：第{c['start_page']}页（相似度={sim:.4f}）")

        results.append({
            "question": q,
            "answer": answer,
            "sources": [{"page": c['start_page'], "sim": float(cosine_similarity(query_vec, embeddings[c['chunk_id']]))} for c in top_chunks]
        })

    with open("data/processed/test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\n结果已保存到 data/processed/test_results.json")


if __name__ == "__main__":
    main()
