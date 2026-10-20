# Building Code RAG

面向建筑规范的 RAG（检索增强生成）知识库项目。利用建环专业背景构建的求职项目。

## 项目结构

```
building-rag/
├── data/
│   ├── raw/          # 原始 PDF 规范文件（不提交）
│   └── processed/    # 解析输出（不提交）
├── src/
│   ├── pdf_parser.py      # PDF 文本解析（Day 1）
│   ├── text_splitter.py   # 文本分块器（Day 2）
│   └── test_text_splitter.py  # 单元测试
├── .gitignore
├── .python-version
├── pyproject.toml
└── uv.lock
```

## 环境要求

- Python 3.12+
- uv

## 快速开始

```powershell
# 安装依赖
uv sync

# 解析 PDF
uv run python src/pdf_parser.py data/raw/你的规范.pdf

# 文本分块
uv run python src/text_splitter.py data/processed/你的规范.txt

# 运行测试
uv run python src/test_text_splitter.py
```

## 开发进度

- [x] Day 1：PDF 文本解析（PyMuPDF）
- [x] Day 2：文本分块（段落优先 + 滑动窗口）
- [ ] Day 3：Embedding 向量化
- [ ] Day 4：向量数据库
- [ ] Day 5：检索 + 大模型问答
