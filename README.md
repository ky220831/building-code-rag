# building-code-rag

面向建筑规范类文本 PDF 的检索增强问答（RAG）后端服务。把 PDF 解析、分块、向量/BM25 混合检索、RRF 融合、CrossEncoder 重排以及 LLM 生成串成了可运行的 FastAPI 接口，返回答案的同时附带出处文件名与页码。

## 流程概览

离线建库：

```text
PDF -> pdf_parser -> TXT -> text_splitter -> chunks.jsonl
                                               ├─> embed_chunks -> embeddings.npz
                                               └─> 启动时构建内存 BM25 索引
```

在线查询：

```text
POST /chat
  -> HybridSearch (Vector + BM25 -> RRF -> CrossEncoder Rerank)
  -> 过滤无效/空结果（未命中时不调用 LLM）
  -> GenerationService (组装带编号上下文 -> OpenAI 兼容 Chat API)
  -> 返回 { answer, sources }
```

## 目录结构

```text
building-code-rag/
├── data/
│   ├── raw/                  # 放原始 PDF（已加入 .gitignore）
│   └── processed/            # 放解析文本、chunks.jsonl、embeddings.npz
├── src/
│   ├── config.py             # 环境变量与检索参数配置
│   ├── common.py             # chunks 读取与 chunk_id 回填
│   ├── pdf_parser.py         # PyMuPDF 文本提取（按页）
│   ├── text_splitter.py      # 滑动窗口分块并保留起止页码
│   ├── embed_chunks.py       # 批量生成向量并保存为 npz
│   ├── vector_retriever.py   # BaseVectorRetriever 接口与 NumPy 实现
│   ├── bm25_retriever.py     # jieba 分词 + BM25 召回
│   ├── rrf_fusion.py         # Reciprocal Rank Fusion 融合
│   ├── reranker.py           # CrossEncoder 重排
│   ├── hybrid_search.py      # 混合检索入口
│   ├── generation.py         # Prompt 构造与 LLM 调用
│   ├── rag_pipeline.py       # 检索与生成编排、来源去重整理
│   └── api.py                # FastAPI 路由与生命周期管理
├── tests/                    # 单元测试（不依赖外部模型和 API）
├── .env.example
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

Windows PowerShell（也可用 `python -m venv .venv` + `pip`）：

```powershell
uv venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### 2. 配置环境变量

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填入有效的 `DEEPSEEK_API_KEY`（或其它兼容 OpenAI 协议的 `LLM_BASE_URL` / `LLM_MODEL`），并把 `CHUNKS_PATH` 和 `EMBEDDINGS_PATH` 指向你生成的语料文件。模型下载慢时可先设置镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

### 3. 构建本地知识库

将带文本层的 PDF 放到 `data/raw/` 下，依次执行：

```powershell
python -m src.pdf_parser "data/raw/your-code.pdf"
python -m src.text_splitter "data/processed/your-code.txt" --chunk-size 800 --overlap 100
python -m src.embed_chunks --chunks "data/processed/your-code.chunks.jsonl" --output "data/processed/your-code.embeddings.npz"
```

完成后更新 `.env` 中的 `CHUNKS_PATH` 和 `EMBEDDINGS_PATH`。

说明：`embeddings.npz` 同时保存 `embeddings` 和 `chunk_ids`，重新分块后需要重新跑一次 `embed_chunks`；空白 chunk 不会进入索引，因此检索回填严格按 `chunk_id` 映射，不依赖数组下标。

### 4. 调试检索与启动服务

命令行单独验证混合检索：

```powershell
python -m src.hybrid_search "防火分区的设置要求"
python -m src.hybrid_search "防火分区的设置要求" --no-reranker
```

启动 FastAPI：

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

访问 `http://127.0.0.1:8000/docs` 可直接调试接口。

## 接口说明

- `GET /health`：检查 Pipeline 初始化状态
- `GET /documents`：列出当前加载的知识库文件名
- `POST /chat`：传入 `{"query": "...", "source": "可选规范名"}`，返回答案与实际被引用的来源页码
- `POST /chat/stream`：SSE 流式问答，依次推送 `delta`、`sources`、`done`（含检索耗时与首字延迟 `ttft_ms`）事件

返回示例：

```json
{
  "answer": "根据检索资料……[资料1]",
  "sources": [
    {
      "source": "GB_50016.txt",
      "page": 12
    }
  ]
}
```

## 运行测试

单元测试使用 mock 数据，无需下载模型或配置 API Key：

```powershell
python -m unittest discover -s tests -v
```

## 离线检索评测

不消耗 LLM Token，直接对比 5 组检索策略的 Hit@1、Hit@5、MRR@5 与平均耗时，并自动生成 `docs/eval_retrieval_report.md`：

```powershell
python -m eval.eval_retrieval --questions eval/questions.json --backend chroma
```

## 关于向量存储

支持两种向量后端，通过 `.env` 中的 `VECTOR_BACKEND` 切换：

- `numpy`（默认）：读取本地 `.npz` 内存检索，零外部库依赖，中小规模规范文档延迟在毫秒级。
- `chroma`：基于 `ChromaVectorRetriever` 本地持久化存储（`CHROMA_PERSIST_DIR`），支持增量 `upsert` 和按 `source` 规范文件名过滤。

切换为 Chroma 建库与检索示例：

```powershell
python -m src.embed_chunks --chunks "data/processed/your-code.chunks.jsonl" --backend chroma
python -m src.hybrid_search "防火分区的设置要求" --backend chroma --source "your-code.txt"
```

## 当前边界与后续计划

- 已完成：文本型 PDF 解析、分块、向量 + BM25 混合召回、RRF 融合、CrossEncoder 重排、FastAPI 封装、基础单元测试。
- 待补充（后续迭代）：
  - [x] 接入 Chroma 本地持久化向量库作为可选后端，支持 `upsert` 增量更新与按规范文件名（`source`）过滤
  - [x] 新增 `POST /chat/stream` SSE 流式输出、分段耗时日志（`retrieval_ms` / `ttft_ms` / `total_ms`）、按 `[资料N]` 过滤引用来源及目录页自动跳过
  - [x] 新增离线检索评测脚本 `eval/eval_retrieval.py`（自动对比 Vector / BM25 / Hybrid / Rerank Top-10 & Top-20 的 Hit@1、Hit@5、MRR@5 与平均耗时）
  - [ ] 补充 Dockerfile 与基础接口鉴权
  - [ ] 扫描件 PDF 的 OCR 预处理
