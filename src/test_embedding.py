import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer

print("正在加载Embedding模型（第一次会下载，需要等几分钟）...")

# 用多语言轻量模型，支持中文
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("模型加载完成！")

# 测试文本
texts = [
    "线性代数研究向量和矩阵运算",
    "梯度下降是一种优化算法",
    "今天天气很好"
]

print("\n正在向量化...")
embeddings = model.encode(texts)

print(f"向量维度：{embeddings.shape}")
print(f"第1个向量前5个值：{embeddings[0][:5]}")

# 计算余弦相似度
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n=== 余弦相似度测试 ===")
print(f"文本1 vs 文本2：{cosine_similarity(embeddings[0], embeddings[1]):.4f}")
print(f"文本1 vs 文本3：{cosine_similarity(embeddings[0], embeddings[2]):.4f}")
print(f"文本2 vs 文本3：{cosine_similarity(embeddings[1], embeddings[2]):.4f}")

print("\n说明：数值越接近1，说明语义越相似")
