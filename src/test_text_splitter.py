"""
text_splitter 单元测试
覆盖场景：短文本、超长段落、多页文本、空白页、overlap边界、JSONL可重读
"""
from pathlib import Path
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).parent))

from text_splitter import parse_pages, split_by_paragraphs, save_jsonl, load_jsonl


def test_short_text():
    """场景1：短文本（远小于chunk_size）"""
    txt = "========== PAGE 1 ==========\n这是一段很短的文本。"
    pages = parse_pages(txt)
    chunks = split_by_paragraphs(pages, chunk_size=800, chunk_overlap=100)
    assert len(chunks) == 1, f"短文本应该只有1个chunk，实际{len(chunks)}"
    assert chunks[0]["text"] == "这是一段很短的文本。"
    assert chunks[0]["start_page"] == 1
    assert chunks[0]["end_page"] == 1
    print("[通过] 短文本测试")


def test_long_paragraph():
    """场景2：超长段落（超过chunk_size，验证硬切和向前推进）"""
    # 造一个2000字的超长段落
    long_para = "这是一个超长段落。" * 200  # 1800字
    txt = f"========== PAGE 1 ==========\n{long_para}"
    pages = parse_pages(txt)
    chunks = split_by_paragraphs(pages, chunk_size=800, chunk_overlap=100)
    # 验证：没有死循环，chunk数量合理
    assert len(chunks) > 1, "超长段落应该被切成多个chunk"
    # 验证：每个chunk长度不超过chunk_size
    for c in chunks:
        assert c["char_count"] <= 800, f"chunk长度{c['char_count']}超过800"
    # 验证：chunk_id连续递增
    for i, c in enumerate(chunks):
        assert c["chunk_id"] == i, f"chunk_id不连续：期望{i}，实际{c['chunk_id']}"
    print(f"[通过] 超长段落测试：切出{len(chunks)}个chunk")


def test_multi_page():
    """场景3：多页文本"""
    txt = """========== PAGE 1 ==========
第一页第一段。

第一页第二段。

========== PAGE 2 ==========
第二页第一段。

========== PAGE 3 ==========
第三页第一段。
"""
    pages = parse_pages(txt)
    assert len(pages) == 3, f"应该解析出3页，实际{len(pages)}"
    chunks = split_by_paragraphs(pages, chunk_size=800, chunk_overlap=100)
    # 验证：每个chunk都记录了正确的页码范围
    for c in chunks:
        assert c["start_page"] >= 1
        assert c["end_page"] <= 3
    print(f"[通过] 多页文本测试：3页切出{len(chunks)}个chunk")


def test_blank_page():
    """场景4：空白页面"""
    txt = """========== PAGE 1 ==========

========== PAGE 2 ==========
第二页有内容。

========== PAGE 3 ==========

"""
    pages = parse_pages(txt)
    chunks = split_by_paragraphs(pages, chunk_size=800, chunk_overlap=100)
    # 验证：空白页生成了空chunk
    empty_chunks = [c for c in chunks if c["char_count"] == 0]
    assert len(empty_chunks) >= 2, f"应该有至少2个空白chunk，实际{len(empty_chunks)}"
    print(f"[通过] 空白页测试：共{len(empty_chunks)}个空chunk")


def test_jsonl_roundtrip():
    """场景5：JSONL写入后可重新读取"""
    txt = "========== PAGE 1 ==========\n测试内容A。\n\n========== PAGE 2 ==========\n测试内容B。"
    pages = parse_pages(txt)
    chunks = split_by_paragraphs(pages, chunk_size=800, chunk_overlap=100)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.chunks.jsonl"
        save_jsonl(chunks, out_path, source_name="test.txt")
        # 重新读取
        loaded = load_jsonl(out_path)
        assert len(loaded) == len(chunks), "JSONL读写数量不一致"
        # 验证字段完整
        for item in loaded:
            assert "source" in item
            assert "chunk_id" in item
            assert "start_page" in item
            assert "end_page" in item
            assert "text" in item
        print(f"[通过] JSONL可重读测试：{len(loaded)}条记录")


def test_overlap_no_infinite_loop():
    """场景6：overlap边界，确保不会死循环"""
    # 构造一个刚好在边界的情况
    para = "A" * 1000  # 1000字，chunk_size=800, overlap=100, step=700
    txt = f"========== PAGE 1 ==========\n{para}"
    pages = parse_pages(txt)
    chunks = split_by_paragraphs(pages, chunk_size=800, chunk_overlap=100)
    # 1000字，step=700，应该切3块：[0:800], [700:1500]→[700:1000], 实际就是2块
    assert len(chunks) >= 2, "边界情况应该切出多个chunk"
    print(f"[通过] overlap边界测试：切出{len(chunks)}个chunk，无死循环")


if __name__ == "__main__":
    test_short_text()
    test_long_paragraph()
    test_multi_page()
    test_blank_page()
    test_jsonl_roundtrip()
    test_overlap_no_infinite_loop()
    print("\n===== 全部单元测试通过 =====")
