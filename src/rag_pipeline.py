import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI


def build_prompt(query: str, chunks: list[dict]) -> str:
    """把检索到的chunk拼进Prompt模板"""
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


def load_chunks(jsonl_path: Path) -> list[dict]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def retrieve(query_vec, embeddings, chunks, top_k=3):
    scores = []
    for i, chunk in enumerate(chunks):
        sim = cosine_similarity(query_vec, embeddings[i])
        scores.append((sim, i))
    scores.sort(reverse=True, key=lambda x: x[0])
    return [chunks[idx] for _, idx in scores[:top_k]]


def main():
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    if not DEEPSEEK_API_KEY:
        print("错误：请先设置环境变量 DEEPSEEK_API_KEY")
        return
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1"
    )

    jsonl_path = Path("data/processed/机器学习中的数学.chunks.jsonl")
    emb_path = Path("data/processed/机器学习中的数学.embeddings.npz")

    print("加载数据...")
    chunks = load_chunks(jsonl_path)
    embeddings = np.load(emb_path)["embeddings"]

    print("加载Embedding模型...")
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    while True:
        query = input("\n请输入问题（输入q退出）：").strip()
        if query.lower() == "q":
            break
        if not query:
            continue

        print("正在检索...")
        query_vec = embed_model.encode(query)
        top_chunks = retrieve(query_vec, embeddings, chunks, top_k=3)

        print("正在生成回答...")
        prompt = build_prompt(query, top_chunks)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        answer = response.choices[0].message.content
        print("\n=== 回答 ===")
        print(answer)
        print("\n=== 来源 ===")
        for i, c in enumerate(top_chunks, start=1):
            print(f"资料{i}：第{c['start_page']}页（相似度={cosine_similarity(query_vec, embeddings[c['chunk_id']]):.4f}）")


if __name__ == "__main__":
    main()
