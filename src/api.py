from __future__ import annotations

import hmac
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.config import settings
from src.rag_pipeline import RAGPipeline, build_pipeline

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, description="用户问题")
    source: str | None = Field(default=None, description="可选：限定检索的规范文件名")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query 不能为空或只包含空格")
        return cleaned


class SourceItem(BaseModel):
    source: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


def _format_sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _verify_api_key(request: Request, expected_token: str) -> None:
    """当配置了 API_AUTH_TOKEN 时，校验请求头 X-API-Key。"""
    if not expected_token:
        return
    provided = request.headers.get("X-API-Key", "").strip()
    if not provided or not hmac.compare_digest(provided, expected_token):
        raise HTTPException(status_code=401, detail="无效或缺少 X-API-Key")


def create_app(
    pipeline: RAGPipeline | None = None,
    pipeline_factory: Callable[[], RAGPipeline] = build_pipeline,
    api_auth_token: str | None = None,
) -> FastAPI:
    required_token = (
        settings.api_auth_token if api_auth_token is None else api_auth_token.strip()
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            app.state.rag_pipeline = pipeline
        else:
            try:
                app.state.rag_pipeline = pipeline_factory()
            except Exception:
                logger.exception("RAG Pipeline 初始化失败")
                raise
        yield

    app = FastAPI(title="Building Code RAG API", version="1.2.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        if getattr(request.app.state, "rag_pipeline", None) is None:
            raise HTTPException(status_code=503, detail="RAG Pipeline 尚未就绪")
        return {"status": "ok"}

    @app.get("/documents")
    def documents(request: Request) -> dict[str, list[str]]:
        _verify_api_key(request, required_token)
        rag_pipeline = getattr(request.app.state, "rag_pipeline", None)
        if rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG Pipeline 尚未就绪")
        searcher = getattr(rag_pipeline, "searcher", None)
        if searcher is None or not hasattr(searcher, "list_sources"):
            return {"documents": []}
        return {"documents": searcher.list_sources()}

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        _verify_api_key(request, required_token)
        rag_pipeline = getattr(request.app.state, "rag_pipeline", None)
        if rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG Pipeline 尚未就绪")
        try:
            result: dict[str, Any] = (
                rag_pipeline.run(body.query, source_filter=body.source)
                if body.source
                else rag_pipeline.run(body.query)
            )
            return ChatResponse(**result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("RAG 处理失败")
            raise HTTPException(status_code=502, detail="RAG 检索或生成服务异常") from exc
        except Exception as exc:
            logger.exception("Chat 接口发生未预期异常")
            raise HTTPException(status_code=500, detail="服务器内部错误") from exc

    @app.post("/chat/stream")
    def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
        _verify_api_key(request, required_token)
        rag_pipeline = getattr(request.app.state, "rag_pipeline", None)
        if rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG Pipeline 尚未就绪")

        def event_generator() -> Iterator[str]:
            try:
                stream = (
                    rag_pipeline.run_stream(body.query, source_filter=body.source)
                    if body.source
                    else rag_pipeline.run_stream(body.query)
                )
                for item in stream:
                    yield _format_sse(item["event"], item["data"])
            except ValueError as exc:
                yield _format_sse("error", {"detail": str(exc)})
            except RuntimeError:
                logger.exception("RAG 流式处理失败")
                yield _format_sse("error", {"detail": "RAG 检索或生成服务异常"})
            except Exception:
                logger.exception("Chat stream 接口发生未预期异常")
                yield _format_sse("error", {"detail": "服务器内部错误"})

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
