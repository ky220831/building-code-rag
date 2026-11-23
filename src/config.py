from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    chunks_path: Path = Path(os.getenv("CHUNKS_PATH", "data/processed/sample_code.chunks.jsonl"))
    embeddings_path: Path = Path(os.getenv("EMBEDDINGS_PATH", "data/processed/sample_code.embeddings.npz"))
    vector_backend: str = os.getenv("VECTOR_BACKEND", "numpy").strip().lower()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    use_reranker: bool = os.getenv("USE_RERANKER", "true").strip().lower() in {"1", "true", "yes", "on"}
    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "20"))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "20"))
    fused_top_k: int = int(os.getenv("FUSED_TOP_K", "20"))
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "5"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    llm_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "30"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))


settings = Settings()
