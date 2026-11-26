# 检索策略离线对照评测报告

- 评测集：`eval/questions.json`（共 12 题）
- 向量后端：`chroma` | 元数据过滤：`开启`
- Embedding：`paraphrase-multilingual-MiniLM-L12-v2` | Reranker：`BAAI/bge-reranker-base`

## 指标对比

| 检索策略 | 评测题数 | Hit@1 | Hit@5 | MRR@5 | 平均耗时 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Vector Only (chroma) | 12 | 41.7% | 75.0% | 0.542 | 23.0 |
| BM25 Only | 12 | 91.7% | 100.0% | 0.958 | 0.4 |
| Hybrid (Vector+BM25+RRF, 无重排) | 12 | 33.3% | 100.0% | 0.586 | 23.5 |
| Hybrid + Rerank (FUSED_TOP_K=10) | 12 | 100.0% | 100.0% | 1.000 | 8822.2 |
| Hybrid + Rerank (FUSED_TOP_K=20) | 12 | 100.0% | 100.0% | 1.000 | 17775.6 |
