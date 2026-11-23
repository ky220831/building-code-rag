import unittest

from src.common import build_chunk_map, hydrate_results
from src.rrf_fusion import reciprocal_rank_fusion


class RetrievalTests(unittest.TestCase):
    def test_rrf_uses_chunk_id_and_merges_two_rankings(self):
        vector = [
            {"chunk_id": 10, "vector_score": 0.9, "vector_rank": 1},
            {"chunk_id": 20, "vector_score": 0.8, "vector_rank": 2},
        ]
        bm25 = [
            {"chunk_id": 20, "bm25_score": 5.0, "bm25_rank": 1},
            {"chunk_id": 30, "bm25_score": 4.0, "bm25_rank": 2},
        ]
        results = reciprocal_rank_fusion(vector, bm25, k=60, top_k=3)
        self.assertEqual(results[0]["chunk_id"], 20)
        self.assertAlmostEqual(results[0]["rrf_score"], 1 / 62 + 1 / 61)

    def test_duplicate_id_in_same_list_only_counts_once(self):
        vector = [
            {"chunk_id": 7, "vector_score": 0.9},
            {"chunk_id": 7, "vector_score": 0.8},
        ]
        result = reciprocal_rank_fusion(vector, [], k=60)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["rrf_score"], 1 / 61)

    def test_hydrate_uses_id_mapping_not_list_index(self):
        chunks = [
            {"chunk_id": 2, "text": "A", "source": "a.pdf", "start_page": 1},
            {"chunk_id": 19, "text": "B", "source": "b.pdf", "start_page": 8},
        ]
        chunk_map = build_chunk_map(chunks)
        result = hydrate_results([{"chunk_id": 19, "rrf_score": 0.1}], chunk_map)
        self.assertEqual(result[0]["text"], "B")
        self.assertEqual(result[0]["start_page"], 8)

    def test_hydrate_rejects_unknown_id(self):
        with self.assertRaises(KeyError):
            hydrate_results([{"chunk_id": 99}], {1: {"chunk_id": 1}})




import tempfile
from pathlib import Path
import numpy as np
from src.vector_retriever import BaseVectorRetriever, VectorRetriever, create_vector_retriever


class _DummyEmbeddingModel:
    def encode(self, query, convert_to_numpy=True):
        return np.array([1.0, 0.0], dtype=np.float32)


class VectorRetrieverTests(unittest.TestCase):
    def test_numpy_vector_retriever_respects_chunk_id_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "vecs.npz"
            embeddings = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
            chunk_ids = np.array([101, 205], dtype=np.int64)
            np.savez_compressed(npz_path, embeddings=embeddings, chunk_ids=chunk_ids)

            retriever = create_vector_retriever("numpy", npz_path, _DummyEmbeddingModel())
            self.assertIsInstance(retriever, BaseVectorRetriever)
            self.assertIsInstance(retriever, VectorRetriever)

            hits = retriever.search("防火分区", top_k=2)
            self.assertEqual(hits[0]["chunk_id"], 205)
            self.assertEqual(hits[0]["vector_rank"], 1)
            self.assertAlmostEqual(hits[0]["vector_score"], 1.0, places=5)

    def test_unsupported_backend_raises_value_error(self):
        with self.assertRaises(ValueError):
            create_vector_retriever("unknown_db", Path("dummy.npz"), _DummyEmbeddingModel())


if __name__ == "__main__":
    unittest.main()
