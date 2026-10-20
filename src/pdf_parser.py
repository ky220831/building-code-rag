from pathlib import Path
import argparse
import pymupdf  # PyMuPDF

def parse_pdf(pdf_path: Path) -> tuple[list[str], dict]:
    """
    解析PDF，逐页提取文本；只读取PDF内原生文本，不做OCR识别图片
    :param pdf_path: PDF文件路径对象
    :return: 每页文本列表、统计信息字典
    """
    stats = {
        "total_pages": 0,
        "success_pages": 0,
        "blank_pages": 0,
        "total_chars": 0
    }
    page_texts = []

    # with上下文：自动打开、自动关闭PDF文件，避免资源泄露
    try:
        with pymupdf.open(str(pdf_path)) as doc:
            stats["total_pages"] = len(doc)
            # 遍历所有页面，page_num从0开始（PDF内部索引）
            for page_num in range(stats["total_pages"]):
                page = doc[page_num]
                text = page.get_text()
                page_texts.append(text)

                # 判断页面是否有有效文本
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
    """
    按页码分隔保存文本，UTF-8编码写入txt
    每页添加分隔标记：========== PAGE N ==========
    """
    lines = []
    for idx, text in enumerate(page_texts, start=1):
        lines.append(f"========== PAGE {idx} ==========")
        lines.append(text)
    content = "\n".join(lines)
    out_path.write_text(content, encoding="utf-8")


def main():
    # argparse：接收命令行传入的PDF路径参数
    parser = argparse.ArgumentParser(description="建筑规范PDF文本提取工具")
    parser.add_argument("pdf_file", help="原始PDF文件路径（data/raw/xxx.pdf）")
    args = parser.parse_args()

    pdf_in = Path(args.pdf_file)
    # 校验后缀，只允许pdf文件
    if pdf_in.suffix.lower() != ".pdf":
        print("错误：输入文件必须是.pdf格式")
        return

    # 自动拼接输出路径：data/processed/同名.txt
    out_file = Path("data/processed") / pdf_in.with_suffix(".txt").name
    # 如果processed文件夹不存在，自动创建
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        page_texts, stats = parse_pdf(pdf_in)
    except Exception as err:
        print(f"运行失败：{err}")
        return

    # 调用保存函数写入txt
    save_text(page_texts, out_file)

    # 控制台打印结果信息
    print(f"输入文件：{pdf_in.resolve()}")
    print(f"输出文件：{out_file.resolve()}")
    print("-" * 50)
    print(f"总页数：{stats['total_pages']}")
    print(f"有效文本页：{stats['success_pages']}")
    print(f"空白页：{stats['blank_pages']}")
    print(f"总字符数：{stats['total_chars']}")
    print("-" * 50)
    # 打印前3页预览，不足3页就全部展示
    preview_num = min(3, len(page_texts))
    for i in range(preview_num):
        print(f"【第{i+1}页预览】")
        preview_content = page_texts[i][:300] + "..." if len(page_texts[i])>300 else page_texts[i]
        print(preview_content)
        print("-" * 30)


if __name__ == "__main__":
    main()
