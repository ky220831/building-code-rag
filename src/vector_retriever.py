from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseVectorRetriever(ABC):
    """向量检索抽象基类，便于在 NumPy 内存检索与外部向量库之间平滑切换。"""

    chunk_ids: np.ndarray

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 20,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """返回按相似度降序排列的检索结果列表，每项包含 chunk_id、vector_score、vector_rank。"""


class VectorRetriever(BaseVectorRetriever):
    """基于本地 NPZ 文件的内存向量检索实现，行向量与 chunk_ids 严格一一对应。"""

    def __init__(
        self,
        embeddings_path: Path,
        embedding_model: Any,
        chunk_sources: dict[int, str] | None = None,
    ):
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
        self.chunk_sources = chunk_sources or {}
        self.embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        self.valid_rows = self.embedding_norms > 0

    def search(
        self,
        query: str,
        top_k: int = 20,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
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

        row_mask = self.valid_rows.copy()
        if source_filter and source_filter.strip():
            target = source_filter.strip()
            source_mask = np.array(
                [self.chunk_sources.get(int(cid), "") == target for cid in self.chunk_ids],
                dtype=bool,
            )
            row_mask &= source_mask

        valid_indices = np.flatnonzero(row_mask)
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


class ChromaVectorRetriever(BaseVectorRetriever):
    """基于本地持久化 Chroma 的向量检索实现，支持增量 upsert 与按 source 元数据过滤。"""

    def __init__(
        self,
        embedding_model: Any,
        persist_dir: Path | None = None,
        collection_name: str = "building_codes",
        collection: Any | None = None,
    ):
        self.embedding_model = embedding_model
        if collection is not None:
            self.collection = collection
        else:
            if persist_dir is None:
                raise ValueError("使用 Chroma 后端时必须提供 persist_dir 或 collection")
            try:
                import chromadb
            except ImportError as exc:
                raise ImportError(
                    "未安装 chromadb，请先执行：python -m pip install chromadb"
                ) from exc

            persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_dir))
            self.collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        self._refresh_chunk_ids()

    def _refresh_chunk_ids(self) -> None:
        raw = self.collection.get(include=[])
        ids = raw.get("ids") or []
        self.chunk_ids = np.asarray([int(item) for item in ids], dtype=np.int64)

    def upsert_chunks(self, chunks: list[dict[str, Any]], embeddings: np.ndarray) -> int:
        """将非空 chunks 与对应向量增量写入 Chroma，重复 chunk_id 自动覆盖更新。"""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 与 embeddings 数量不一致")
        if not chunks:
            return 0

        ids = [str(int(chunk["chunk_id"])) for chunk in chunks]
        documents = [str(chunk.get("text", "")) for chunk in chunks]
        metadatas = [
            {
                "chunk_id": int(chunk["chunk_id"]),
                "source": str(chunk.get("source", "")),
                "start_page": int(chunk.get("start_page", 0)),
                "end_page": int(chunk.get("end_page", chunk.get("start_page", 0))),
            }
            for chunk in chunks
        ]
        self.collection.upsert(
            ids=ids,
            embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
            documents=documents,
            metadatas=metadatas,
        )
        self._refresh_chunk_ids()
        return len(ids)

    def search(
        self,
        query: str,
        top_k: int = 20,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not isinstance(query, str) or not query.strip():
            return []

        total = int(self.collection.count())
        if total == 0:
            return []

        query_vector = np.asarray(
            self.embedding_model.encode(query, convert_to_numpy=True), dtype=np.float32
        ).reshape(-1)
        if float(np.linalg.norm(query_vector)) == 0:
            return []

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector.tolist()],
            "n_results": min(top_k, total),
            "include": ["distances"],
        }
        if source_filter and source_filter.strip():
            query_kwargs["where"] = {"source": source_filter.strip()}

        response = self.collection.query(**query_kwargs)
        ids_batch = (response.get("ids") or [[]])[0]
        distances_batch = (response.get("distances") or [[]])[0]

        results: list[dict[str, Any]] = []
        for rank, (raw_id, distance) in enumerate(zip(ids_batch, distances_batch), start=1):
            similarity = 1.0 - float(distance)
            results.append({
                "chunk_id": int(raw_id),
                "vector_score": similarity,
                "vector_rank": rank,
            })
        return results


NumpyVectorRetriever = VectorRetriever


def create_vector_retriever(
    backend: str,
    embeddings_path: Path,
    embedding_model: Any,
    *,
    chroma_persist_dir: Path | None = None,
    chroma_collection: str = "building_codes",
    chunk_sources: dict[int, str] | None = None,
) -> BaseVectorRetriever:
    """按配置创建向量检索实例。"""
    normalized = backend.strip().lower()
    if normalized == "numpy":
        return VectorRetriever(embeddings_path, embedding_model, chunk_sources=chunk_sources)
    if normalized == "chroma":
        if chroma_persist_dir is None:
            raise ValueError("Chroma 后端缺少 chroma_persist_dir 配置")
        return ChromaVectorRetriever(
            embedding_model=embedding_model,
            persist_dir=chroma_persist_dir,
            collection_name=chroma_collection,
        )
    raise ValueError(f"暂不支持的向量后端：{backend}（当前可用：numpy, chroma）")
