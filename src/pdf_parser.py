from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf


def parse_pdf(pdf_path: Path) -> tuple[list[str], dict[str, int]]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"文件不存在：{pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("输入文件必须是 .pdf 格式")

    stats = {"total_pages": 0, "success_pages": 0, "blank_pages": 0, "total_chars": 0}
    page_texts: list[str] = []
    try:
        with pymupdf.open(str(pdf_path)) as doc:
            stats["total_pages"] = len(doc)
            for page in doc:
                page_text = page.get_text()
                page_texts.append(page_text)
                if page_text.strip():
                    stats["success_pages"] += 1
                    stats["total_chars"] += len(page_text)
                else:
                    stats["blank_pages"] += 1
    except pymupdf.FileDataError as exc:
        raise RuntimeError(f"PDF 损坏或不是合法 PDF 文件：{pdf_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"读取 PDF 失败：{exc}") from exc
    return page_texts, stats


def save_text(page_texts: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for page_number, page_text in enumerate(page_texts, start=1):
        lines.append(f"========== PAGE {page_number} ==========")
        lines.append(page_text)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="建筑规范 PDF 文本提取工具")
    parser.add_argument("pdf_file", help="PDF 路径，例如 data/raw/xxx.pdf")
    parser.add_argument("--output", default=None, help="可选输出 TXT 路径")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_file)
    output_path = Path(args.output) if args.output else Path("data/processed") / pdf_path.with_suffix(".txt").name
    page_texts, stats = parse_pdf(pdf_path)
    save_text(page_texts, output_path)
    print(f"输入文件：{pdf_path.resolve()}")
    print(f"输出文件：{output_path.resolve()}")
    print(f"总页数：{stats['total_pages']} | 有效文本页：{stats['success_pages']} | 空白页：{stats['blank_pages']}")
    print(f"总字符数：{stats['total_chars']}")
    for index, page_text in enumerate(page_texts[:3], start=1):
        preview = page_text[:300] + ("..." if len(page_text) > 300 else "")
        print(f"\n【第 {index} 页预览】\n{preview}")


if __name__ == "__main__":
    main()
