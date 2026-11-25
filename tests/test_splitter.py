import unittest

from src.text_splitter import is_toc_page, parse_pages, split_by_paragraphs


class SplitterTests(unittest.TestCase):
    def test_parse_page_markers(self):
        pages = parse_pages("========== PAGE 1 ==========\n甲\n\n========== PAGE 2 ==========\n乙")
        self.assertEqual([item["page"] for item in pages], [1, 2])
        self.assertEqual(pages[1]["content"], "乙")

    def test_long_paragraph_uses_overlap_windows(self):
        chunks = split_by_paragraphs([{"page": 1, "content": "x" * 1700}], 800, 100)
        self.assertEqual([len(item["text"]) for item in chunks], [800, 800, 300])
        self.assertEqual(chunks[1]["text"][:100], chunks[0]["text"][-100:])

    def test_blank_page_is_preserved_as_empty_chunk(self):
        chunks = split_by_paragraphs([{"page": 4, "content": "   "}], 800, 100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "")
        self.assertEqual(chunks[0]["start_page"], 4)

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            split_by_paragraphs([], chunk_size=100, chunk_overlap=100)


    def test_toc_page_is_skipped_without_shifting_page_numbers(self):
        toc = "目 次\n1 总则 .................... 1\n2 术语 .................... 3\n3 基本规定 ................ 5"
        pages = [
            {"page": 2, "content": toc},
            {"page": 5, "content": "3.1.1 室内设计计算温度不低于18℃。"},
        ]
        self.assertTrue(is_toc_page(toc))
        chunks = split_by_paragraphs(pages, 800, 100)
        self.assertEqual(chunks[0]["start_page"], 2)
        self.assertEqual(chunks[0]["text"], "")
        self.assertEqual(chunks[1]["start_page"], 5)
        self.assertIn("3.1.1", chunks[1]["text"])


if __name__ == "__main__":
    unittest.main()
