# Building Code RAG - 全部代码汇总

---

## 文件1：src/pdf_parser.py（PDF解析）

```python
from pathlib import Path
import argparse
import pymupdf


def parse_pdf(pdf_path: Path) -> tuple[list[str], dict]:
    stats = {
        "total_pages": 0,
        "success_pages": 0,
        "blank_pages": 0,
        "total_chars": 0
    }
    page_texts = []

    try:
        with pymupdf.open(str(pdf_path)) as doc:
            stats["total_pages"] = len(doc)

            for page_num in range(stats["total_pages"]):
                page = doc[page_num]
                text = page.get_text()
                page_texts.append(text)

                if text.strip():
                    stats["success_pages"] += 1
                    stats["total_chars"] += len(text)
                else:
                    stats["blank_pages"] += 1
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在：{pdf_path}")
    except pymupdf.FileDataError:
        raise RuntimeError(f"PDF损坏或不是合法PDF文件：{pdf_path}")
    except Exception as e:
        raise RuntimeError(f"读取PDF失败：{str(e)}")

    return page_texts, stats


def save_text(page_texts: list[str], out_path: Path):
    lines = []
    for idx, text in enumerate(page_texts, start=1):
        lines.append(f"========== PAGE {idx} ==========")
        lines.append(text)
    content = "\n".join(lines)
    out_path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="建筑规范PDF文本提取工具")
    parser.add_argument("pdf_file", help="原始PDF文件路径（data/raw/xxx.pdf）")
    args = parser.parse_args()

    pdf_in = Path(args.pdf_file)

    if pdf_in.suffix.lower() != ".pdf":
        print("错误：输入文件必须是.pdf格式")
        return

    out_file = Path("data/processed") / pdf_in.with_suffix(".txt").name
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        page_texts, stats = parse_pdf(pdf_in)
    except Exception as err:
        print(f"运行失败：{err}")
        return

    save_text(page_texts, out_file)

    print(f"输入文件：{pdf_in.resolve()}")
    print(f"输出文件：{out_file.resolve()}")
    print("-" * 50)
    print(f"总页数：{stats['total_pages']}")
    print(f"有效文本页：{stats['success_pages']}")
    print(f"空白页：{stats['blank_pages']}")
    print(f"总字符数：{stats['total_chars']}")
    print("-" * 50)

    preview_num = min(3, len(page_texts))
    for i in range(preview_num):
        print(f"【第{i+1}页预览】")
        preview_content = page_texts[i][:300] + "..." if len(page_texts[i]) > 300 else page_texts[i]
        print(preview_content)
        print("-" * 30)


if __name__ == "__main__":
    main()
```

---

## 文件2：src/text_splitter.py（文本分块）

```python
from pathlib import Path
import argparse
import re
import json

PAGE_RE = re.compile(r"^========== PAGE (\d+) ==========\s*$")


def parse_pages(txt: str) -> list[dict]:
    pages = []
    current_page = None
    current_lines = []

    for line in txt.splitlines():
        match = PAGE_RE.match(line.strip())
        if match:
            if current_page is not None or current_lines:
                pages.append({
                    "page": current_page if current_page is not None else 0,
                    "content": "\n".join(current_lines).strip()
                })
            current_page = int(match.group(1))
            current_lines = []
        else:
            current_lines.append(line)

    if current_page is not None or current_lines:
        pages.append({
            "page": current_page if current_page is not None else 0,
            "content": "\n".join(current_lines).strip()
        })
    return pages


def split_paragraphs(page_content: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", page_content)
    return [p.strip() for p in paragraphs if p.strip()]


def split_by_paragraphs(
    pages: list[dict],
    chunk_size: int,
    chunk_overlap: int
) -> list[dict]:
    if chunk_overlap >= chunk_size:
        raise ValueError("overlap必须小于chunk_size，否则产生死循环")

    chunks = []
    chunk_id = 0
    slide_step = chunk_size - chunk_overlap

    for page_info in pages:
        page_num = page_info["page"]
        para_list = split_paragraphs(page_info["content"])

        if not para_list:
            chunks.append({
                "chunk_id": chunk_id,
                "start_page": page_num,
                "end_page": page_num,
                "text": "",
                "char_count": 0
            })
            chunk_id += 1
            continue

        current_buf = ""
        buf_start_page = page_num

        for para in para_list:
            if len(para) > chunk_size:
                if current_buf:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "start_page": buf_start_page,
                        "end_page": page_num,
                        "text": current_buf.strip(),
                        "char_count": len(current_buf.strip())
                    })
                    chunk_id += 1
                    current_buf = ""

                p_start = 0
                p_len = len(para)
                while p_start < p_len:
                    p_end = p_start + chunk_size
                    sub_text = para[p_start:p_end]
                    chunks.append({
                        "chunk_id": chunk_id,
                        "start_page": page_num,
                        "end_page": page_num,
                        "text": sub_text,
                        "char_count": len(sub_text)
                    })
                    chunk_id += 1
                    if p_end >= p_len:
                        break
                    p_start += slide_step
                continue

            if len(current_buf) + len(para) + 2 <= chunk_size:
                current_buf = f"{current_buf}\n\n{para}" if current_buf else para
            else:
                chunks.append({
                    "chunk_id": chunk_id,
                    "start_page": buf_start_page,
                    "end_page": page_num,
                    "text": current_buf.strip(),
                    "char_count": len(current_buf.strip())
                })
                chunk_id += 1
                current_buf = para
                buf_start_page = page_num

        if current_buf:
            chunks.append({
                "chunk_id": chunk_id,
                "start_page": buf_start_page,
                "end_page": page_num,
                "text": current_buf.strip(),
                "char_count": len(current_buf.strip())
            })
            chunk_id += 1
    return chunks


def save_jsonl(chunks: list[dict], out_path: Path, source_name: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in chunks:
            record = {
                "source": source_name,
                "chunk_id": item["chunk_id"],
                "start_page": item["start_page"],
                "end_page": item["end_page"],
                "text": item["text"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    result = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result


def main():
    parser = argparse.ArgumentParser(description="RAG文本分块器")
    parser.add_argument("txt_file", help="输入txt路径 data/processed/xxx.txt")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    txt_path = Path(args.txt_file)
    if not txt_path.exists():
        print(f"错误：文件不存在 -> {txt_path}")
        return
    if txt_path.suffix.lower() != ".txt":
        print("错误：输入文件后缀必须为 .txt")
        return

    full_text = txt_path.read_text(encoding="utf-8")
    total_input_chars = len(full_text)

    page_list = parse_pages(full_text)
    chunk_list = split_by_paragraphs(page_list, args.chunk_size, args.overlap)

    out_jsonl = txt_path.with_suffix(".chunks.jsonl")
    save_jsonl(chunk_list, out_jsonl, source_name=txt_path.name)

    non_empty = [c for c in chunk_list if c["char_count"] > 0]
    non_empty_len = [c["char_count"] for c in non_empty]

    print(f"输入文本：{txt_path.resolve()}")
    print(f"输出JSONL：{out_jsonl.resolve()}")
    print("-" * 60)
    print(f"输入总字符：{total_input_chars}")
    print(f"识别页面总数：{len(page_list)}")
    print(f"生成chunk总数：{len(chunk_list)}")
    if non_empty_len:
        print(f"非空chunk数量：{len(non_empty_len)}")
        print(f"最短chunk长度：{min(non_empty_len)}")
        print(f"最长chunk长度：{max(non_empty_len)}")
        print(f"平均chunk长度：{sum(non_empty_len)/len(non_empty_len):.1f}")
    print("-" * 60)

    for c in chunk_list[:3]:
        preview = c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"]
        preview = preview if preview else "(空白文本)"
        print(f"chunk_id={c['chunk_id']} | page {c['start_page']}~{c['end_page']} | len={c['char_count']}")
        print(preview)
        print("-" * 40)


if __name__ == "__main__":
    main()
```

---

## 文件3：src/retriever.py（关键词检索）

```python
from pathlib import Path
import argparse
import json


def load_chunks(jsonl_path: Path) -> list[dict]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def count_keyword_hits(text: str, keywords: list[str]) -> int:
    hits = 0
    for kw in keywords:
        hits += text.count(kw)
    return hits


def search(chunks: list[dict], query: str, top_k: int = 5) -> list[dict]:
    keywords = [kw.strip() for kw in query.split() if kw.strip()]

    scored = []
    for chunk in chunks:
        text = chunk.get("text", "")
        score = count_keyword_hits(text, keywords)
        scored.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "score": score,
            "text": text
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = [r for r in scored if r["score"] > 0][:top_k]
    return results


def main():
    parser = argparse.ArgumentParser(description="关键词检索器")
    parser.add_argument("jsonl_file", help="chunks.jsonl 文件路径")
    parser.add_argument("query", help="查询关键词，用空格分隔")
    parser.add_argument("--top-k", type=int, default=5, help="返回前N条结果，默认5")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_file)
    if not jsonl_path.exists():
        print(f"错误：文件不存在 -> {jsonl_path}")
        return

    chunks = load_chunks(jsonl_path)
    print(f"已加载 {len(chunks)} 个chunk")
    print(f"查询：{args.query}")
    print(f"关键词：{[kw.strip() for kw in args.query.split() if kw.strip()]}")
    print("-" * 60)

    results = search(chunks, args.query, args.top_k)

    if not results:
        print("没有找到匹配的结果")
        return

    print(f"返回 Top {len(results)} 结果：")
    for i, r in enumerate(results, start=1):
        print(f"\n【第{i}条】得分={r['score']} | 页 {r['start_page']}~{r['end_page']} | chunk_id={r['chunk_id']}")
        preview = r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
        print(preview)
        print("-" * 40)


if __name__ == "__main__":
    main()
```

---

## 文件4：src/embed_chunks.py（向量化）

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    jsonl_path = Path("data/processed/机器学习中的数学.chunks.jsonl")
    out_path = Path("data/processed/机器学习中的数学.embeddings.npz")

    print("加载chunks...")
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"共{len(chunks)}个chunk")

    print("加载Embedding模型...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    print("开始向量化...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    print(f"向量形状：{embeddings.shape}")

    np.savez(
        out_path,
        embeddings=embeddings,
        chunk_ids=np.array([c["chunk_id"] for c in chunks])
    )
    print(f"已保存到：{out_path.resolve()}")

if __name__ == "__main__":
    main()
```

---

## 文件5：src/vector_retriever.py（向量检索）

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_chunks(jsonl_path: Path) -> list[dict]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def search(chunks, embeddings, query_vec, top_k=5):
    scores = []
    for i, chunk in enumerate(chunks):
        sim = cosine_similarity(query_vec, embeddings[i])
        scores.append((sim, i))

    scores.sort(reverse=True, key=lambda x: x[0])
    results = []
    for sim, idx in scores[:top_k]:
        results.append({
            "chunk_id": chunks[idx]["chunk_id"],
            "source": chunks[idx]["source"],
            "start_page": chunks[idx]["start_page"],
            "end_page": chunks[idx]["end_page"],
            "score": float(sim),
            "text": chunks[idx]["text"]
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="向量检索器")
    parser.add_argument("query", help="查询文本")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    jsonl_path = Path("data/processed/机器学习中的数学.chunks.jsonl")
    emb_path = Path("data/processed/机器学习中的数学.embeddings.npz")

    print("加载chunks...")
    chunks = load_chunks(jsonl_path)

    print("加载向量...")
    data = np.load(emb_path)
    embeddings = data["embeddings"]

    print("加载模型...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    print(f"查询：{args.query}")
    print("正在向量化查询...")
    query_vec = model.encode(args.query)

    results = search(chunks, embeddings, query_vec, args.top_k)

    print("-" * 60)
    print(f"返回 Top {len(results)} 结果：")
    for i, r in enumerate(results, start=1):
        print(f"\n【第{i}条】相似度={r['score']:.4f} | 页 {r['start_page']}~{r['end_page']} | chunk_id={r['chunk_id']}")
        preview = r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
        print(preview)
        print("-" * 40)


if __name__ == "__main__":
    main()
```

---

## 文件6：src/rag_pipeline.py（完整RAG链路）

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(f"【资料{i}】（第{chunk['start_page']}页）\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    prompt = f"""你是一个专业的问答助手。请根据下面提供的资料回答用户的问题。
要求：
1. 只根据资料回答，不要编造
2. 如果资料里没有相关信息，请直接说"未找到相关内容"
3. 回答要简洁准确

【资料】
{context}

【问题】
{query}

请回答："""
    return prompt


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_chunks(jsonl_path: Path) -> list[dict]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def retrieve(query_vec, embeddings, chunks, top_k=3):
    scores = []
    for i, chunk in enumerate(chunks):
        sim = cosine_similarity(query_vec, embeddings[i])
        scores.append((sim, i))
    scores.sort(reverse=True, key=lambda x: x[0])
    return [chunks[idx] for _, idx in scores[:top_k]]


def main():
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    if not DEEPSEEK_API_KEY:
        print("错误：请先设置环境变量 DEEPSEEK_API_KEY")
        return
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1"
    )

    jsonl_path = Path("data/processed/机器学习中的数学.chunks.jsonl")
    emb_path = Path("data/processed/机器学习中的数学.embeddings.npz")

    print("加载数据...")
    chunks = load_chunks(jsonl_path)
    embeddings = np.load(emb_path)["embeddings"]

    print("加载Embedding模型...")
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    while True:
        query = input("\n请输入问题（输入q退出）：").strip()
        if query.lower() == "q":
            break
        if not query:
            continue

        print("正在检索...")
        query_vec = embed_model.encode(query)
        top_chunks = retrieve(query_vec, embeddings, chunks, top_k=3)

        print("正在生成回答...")
        prompt = build_prompt(query, top_chunks)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        answer = response.choices[0].message.content
        print("\n=== 回答 ===")
        print(answer)
        print("\n=== 来源 ===")
        for i, c in enumerate(top_chunks, start=1):
            print(f"资料{i}：第{c['start_page']}页（相似度={cosine_similarity(query_vec, embeddings[c['chunk_id']]):.4f}）")


if __name__ == "__main__":
    main()
```

---

## 运行顺序

```bash
# 1. PDF解析
python src/pdf_parser.py "data/raw/xxx.pdf"

# 2. 文本分块
python src/text_splitter.py "data/processed/xxx.txt"

# 3. 向量化（只跑一次）
python src/embed_chunks.py

# 4. 向量检索
python src/vector_retriever.py "你的问题"

# 5. 完整RAG问答
python src/rag_pipeline.py
```
