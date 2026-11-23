from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseVectorRetriever(ABC):
    """向量检索抽象基类，便于后续接入 Chroma / Qdrant / Milvus 等向量库。"""

    chunk_ids: np.ndarray

    @abstractmethod
    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """返回按相似度降序排列的检索结果列表，每项包含 chunk_id、vector_score、vector_rank。"""


class VectorRetriever(BaseVectorRetriever):
    """基于本地 NPZ 文件的内存向量检索实现，行向量与 chunk_ids 严格一一对应。"""

    def __init__(self, embeddings_path: Path, embedding_model: Any):
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embedding 文件不存在：{embeddings_path}")

        with np.load(embeddings_path, allow_pickle=False) as data:
            if "embeddings" not in data or "chunk_ids" not in data:
                raise ValueError("NPZ 必须同时包含 embeddings 和 chunk_ids")
            self.embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            self.chunk_ids = np.asarray(data["chunk_ids"], dtype=np.int64)

        if self.embeddings.ndim != 2:
            raise ValueError("embeddings 必须是二维矩阵")
        if self.chunk_ids.ndim != 1 or len(self.embeddings) != len(self.chunk_ids):
            raise ValueError("Embedding 行数与 chunk_ids 数量不一致")
        if len(set(self.chunk_ids.tolist())) != len(self.chunk_ids):
            raise ValueError("NPZ 中存在重复 chunk_id")
        if len(self.chunk_ids) == 0:
            raise ValueError("Embedding 文件为空")

        self.embedding_model = embedding_model
        self.embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        self.valid_rows = self.embedding_norms > 0

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not isinstance(query, str) or not query.strip():
            return []

        query_vector = np.asarray(
            self.embedding_model.encode(query, convert_to_numpy=True), dtype=np.float32
        ).reshape(-1)
        if query_vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query embedding 维度与文档 embedding 维度不一致")

        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0:
            return []

        valid_indices = np.flatnonzero(self.valid_rows)
        if len(valid_indices) == 0:
            return []
        scores = self.embeddings[valid_indices] @ query_vector
        scores = scores / (self.embedding_norms[valid_indices] * query_norm)
        order = np.argsort(-scores)[:top_k]

        results: list[dict[str, Any]] = []
        for rank, position in enumerate(order, start=1):
            row = int(valid_indices[int(position)])
            results.append({
                "chunk_id": int(self.chunk_ids[row]),
                "vector_score": float(scores[int(position)]),
                "vector_rank": rank,
            })
        return results


NumpyVectorRetriever = VectorRetriever


def create_vector_retriever(
    backend: str,
    embeddings_path: Path,
    embedding_model: Any,
) -> BaseVectorRetriever:
    """按配置创建向量检索实例，后续接入外部向量库时在此注册即可。"""
    normalized = backend.strip().lower()
    if normalized == "numpy":
        return VectorRetriever(embeddings_path, embedding_model)
    raise ValueError(f"暂不支持的向量后端：{backend}（当前可用：numpy）")
