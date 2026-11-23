from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from sentence_transformers import SentenceTransformer

from src.common import load_chunks
from src.config import settings


def create_embeddings(chunks: list[dict], model: SentenceTransformer) -> tuple[np.ndarray, np.ndarray]:
    valid_chunks = [chunk for chunk in chunks if str(chunk.get("text", "")).strip()]
    if not valid_chunks:
        raise ValueError("没有可向量化的非空 chunk")

    texts = [chunk["text"] for chunk in valid_chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    chunk_ids = np.asarray([int(chunk["chunk_id"]) for chunk in valid_chunks], dtype=np.int64)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or len(embeddings) != len(chunk_ids):
        raise RuntimeError("Embedding 矩阵行数与 chunk_ids 数量不一致")
    return embeddings, chunk_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="将 JSONL chunks 批量向量化")
    parser.add_argument("--chunks", default=str(settings.chunks_path))
    parser.add_argument("--output", default=str(settings.embeddings_path))
    parser.add_argument("--model", default=settings.embedding_model)
    args = parser.parse_args()

    chunks = load_chunks(Path(args.chunks))
    model = SentenceTransformer(args.model)
    embeddings, chunk_ids = create_embeddings(chunks, model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, embeddings=embeddings, chunk_ids=chunk_ids)
    print(f"Embedding shape：{embeddings.shape}")
    print(f"chunk_ids shape：{chunk_ids.shape}")
    print(f"保存路径：{output_path.resolve()}")


if __name__ == "__main__":
    main()
