from pathlib import Path
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).parent))

from text_splitter import parse_pages, split_by_paragraphs, save_jsonl, load_jsonl


def test_short_text():
    txt = "========== PAGE 1 ==========\n这是一段很短的文本。"
    pages = parse_pages(txt)
    chunks = split_by_paragraphs(pages, chunk_size=800, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "这是一段很短的文本。"
    assert chunks[0]["start_page"] == 1
    assert chunks[0]["end_page"] == 1
    print("[通过] 短文本测试")


def test_long_paragraph():
    long_para = "这是一个超长段落。" * 200
    txt = f"========== PAGE 1 ==========\n{long_para}"
    pages = parse_pages(txt)
    chunks = split_by_paragraphs(pages, chunk_size=800, chunk_overlap=100)
    assert len(chunks) > 1
    for c in chunks:
        assert c["char_count"] <= 800
    for i, c in enumerate(chunks):
        assert c["chunk_id"] == i
    print(f"[通过] 超长段落测试：切出{len(chunks)}个chunk")


def test_multi_page():
    txt = """========== PAGE 1 ==========
第一页第一段。

第一页第二段。

========== PAGE 2 ==========
第二页第一段。

========== PAGE 3 ==========
第三页第一段。
"""
    pages = parse_pages(txt)
    assert len(pages) == 3
    chunks = split_by_paragraphs(pages, chunk_size=800, chunk_overlap=100)
    for c in chunks:
        assert c["start_page"] >= 1
        assert c["end_page"] <= 3
    print(f"[通过] 多页文本测试：3页切出{len(chunks)}个chunk")


def test_blank_page():
    txt = """========== PAGE 1 ==========

========== PAGE 2 ==========
第二页有内容。

========== PAGE 3 ==========

"""
    pages = parse_pages(txt)
    chunks = split_by_paragraphs(pages, chunk_size=800, chunk_overlap=100)
    empty_chunks = [c for c in chunks if c["char_count"] == 0]
    assert len(empty_chunks) >= 2
    print(f"[通过] 空白页测试：共{len(empty_chunks)}个空chunk")


def test_jsonl_roundtrip():
    txt = "========== PAGE 1 ==========\n测试内容A。\n\n========== PAGE 2 ==========\n测试内容B。"
    pages = parse_pages(txt)
    chunks = split_by_paragraphs(pages, chunk_size=800, chunk_overlap=100)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.chunks.jsonl"
        save_jsonl(chunks, out_path, source_name="test.txt")
        loaded = load_jsonl(out_path)
        assert len(loaded) == len(chunks)
        for item in loaded:
            assert "source" in item
            assert "chunk_id" in item
            assert "start_page" in item
            assert "end_page" in item
            assert "text" in item
        print(f"[通过] JSONL可重读测试：{len(loaded)}条记录")


def test_overlap_no_infinite_loop():
    para = "A" * 1000
    txt = f"========== PAGE 1 ==========\n{para}"
    pages = parse_pages(txt)
    chunks = split_by_paragraphs(pages, chunk_size=800, chunk_overlap=100)
    assert len(chunks) >= 2
    print(f"[通过] overlap边界测试：切出{len(chunks)}个chunk，无死循环")


if __name__ == "__main__":
    test_short_text()
    test_long_paragraph()
    test_multi_page()
    test_blank_page()
    test_jsonl_roundtrip()
    test_overlap_no_infinite_loop()
    print("\n===== 全部单元测试通过 =====")
