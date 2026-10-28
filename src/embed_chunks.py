import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    jsonl_path = Path("data/processed/机器学习中的数学.chunks.jsonl")
    out_path = Path("data/processed/机器学习中的数学.embeddings.npz")

    print("加载chunks...")
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"共{len(chunks)}个chunk")

    print("加载Embedding模型...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    print("开始向量化...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    print(f"向量形状：{embeddings.shape}")

    np.savez(
        out_path,
        embeddings=embeddings,
        chunk_ids=np.array([c["chunk_id"] for c in chunks])
    )
    print(f"已保存到：{out_path.resolve()}")

if __name__ == "__main__":
    main()
