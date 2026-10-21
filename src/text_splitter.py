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
