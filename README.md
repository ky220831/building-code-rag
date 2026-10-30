# Building Code RAG

基于检索增强生成（RAG）的建筑规范问答系统。

## 项目简介

本项目实现了一个完整的 RAG（Retrieval-Augmented Generation）问答链路：
- PDF 文档解析 → 文本分块 → 向量化 → 语义检索 → Prompt 组装 → 大模型生成回答

目标是帮助建筑行业从业者快速查询规范条文，回答带来源页码，可追溯。

## 技术栈

| 模块 | 技术选型 |
|---|---|
| PDF 解析 | PyMuPDF |
| 文本分块 | 自定义段落优先 + 滑动窗口 |
| Embedding | sentence-transformers（本地运行，多语言模型） |
| 向量检索 | 余弦相似度（NumPy 实现） |
| 大模型 | DeepSeek Chat API |
| 开发语言 | Python 3.11+ |

## 项目结构

```
building-rag/
├── data/
│   ├── raw/                # 原始 PDF 文件
│   └── processed/          # 处理后的 txt / chunks.jsonl / embeddings.npz
├── src/
│   ├── pdf_parser.py       # PDF 文本提取
│   ├── text_splitter.py    # 文本分块
│   ├── retriever.py        # 关键词检索
│   ├── embed_chunks.py     # 批量向量化
│   ├── vector_retriever.py # 向量语义检索
│   └── rag_pipeline.py     # 完整 RAG 链路
├── docs/
│   └── retriever_optimization_log.md  # 检索优化记录
├── pyproject.toml
└── README.md
```

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
uv venv
.venv\Scripts\activate

# 安装依赖
uv add pymupdf sentence-transformers numpy openai
```

### 2. 环境变量

```powershell
$env:DEEPSEEK_API_KEY="你的API Key"
$env:HF_ENDPOINT="https://hf-mirror.com"  # 国内镜像，加速模型下载
```

### 3. 运行流程

```bash
# Step 1: PDF 解析（提取文本）
python src/pdf_parser.py "data/raw/xxx.pdf"

# Step 2: 文本分块（输出 chunks.jsonl）
python src/text_splitter.py "data/processed/xxx.txt"

# Step 3: 批量向量化（输出 embeddings.npz，只跑一次）
python src/embed_chunks.py

# Step 4: 启动问答
python src/rag_pipeline.py
```

## 核心设计

### 文本分块策略
- **段落优先**：优先按自然段落切分，避免切断句子语义
- **滑动窗口**：超长段落按 chunk_size=800 字符滑动切分
- **重叠设计**：overlap=100 字符，保证上下文连贯
- **页码保留**：每个 chunk 记录起止页码，回答可溯源

### 检索策略
- **关键词检索**：基于词频计数，适合精确匹配（如"第3.2.1条"）
- **向量检索**：基于余弦相似度，适合语义匹配（如"关于防火要求的内容"）
- **混合检索**：生产环境可结合两者，提高召回率

### Prompt 模板
```
你是一个专业的问答助手。请根据下面提供的资料回答用户的问题。
1. 只根据资料回答，不要编造
2. 如果资料里没有相关信息，请直接说"未找到相关内容"
3. 回答要简洁准确

【资料】
（检索到的 chunk 内容，带页码）

【问题】
（用户的问题）
```

## 测试效果

使用《机器学习中的数学》教材（324页，653个chunk）进行测试：

| 问题 | 结果 | 来源 |
|---|---|---|
| 什么是梯度下降？ | ✅ 回答准确 | 第192页 |
| 什么是主成分分析？ | ✅ 回答简洁 | 第119页 |
| 线性回归的最小二乘法怎么推导？ | ⚠️ 部分回答 | 第253页 |
| 贝叶斯定理是什么？ | ❌ 未找到 | 检索到概率章节但不匹配 |

## 已知问题（BadCase）

1. **目录页干扰**：关键词检索时，目录页（只出现一次关键词）和正文页得分相同，排序不合理
2. **召回率不足**：向量检索 Top 3 可能漏掉相关内容，导致回答"未找到"
3. **扫描件 PDF**：无文本层的扫描件无法直接提取，需要 OCR 预处理

## 后续优化方向

- [ ] 混合检索（关键词 + 向量，结果融合）
- [ ] 重排序（Rerank）：对 Top 10 结果重新打分，取 Top 3
- [ ] 接入真实建筑规范文本版 PDF
- [ ] 部署为 Web 服务（FastAPI + 前端页面）
- [ ] 评估指标（召回率、准确率）

## 开发进度

- [x] Day 1：PDF 文本解析（PyMuPDF）
- [x] Day 2：文本分块（段落优先 + 滑动窗口）
- [x] Day 3：关键词检索 + 优化（词频计数）
- [x] Day 4：Embedding 向量化 + 向量检索
- [x] Day 5：完整 RAG 链路（DeepSeek API 生成回答）
- [ ] Day 6：测试 BadCase 记录 + README 完善

## License

MIT
