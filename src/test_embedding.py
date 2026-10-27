from openai import OpenAI
import os

# 把你的 DeepSeek API key 填在这里，或者从环境变量读
# 申请地址：https://platform.deepseek.com/
DEEPSEEK_API_KEY = "在这里填你的key"

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

# 测试文本
test_text = "线性代数研究向量和矩阵运算"

print("测试文本：", test_text)
print("正在调用Embedding API...")

try:
    response = client.embeddings.create(
        model="deepseek-embedding",
        input=test_text
    )
    embedding = response.data[0].embedding
    print(f"成功！向量维度：{len(embedding)}")
    print(f"前5个值：{embedding[:5]}")
except Exception as e:
    print(f"调用失败：{e}")
    print("\n可能原因：DeepSeek 没有 embedding 接口")
    print("解决方案：换用其他 Embedding 服务（智谱/通义/OpenAI）")
