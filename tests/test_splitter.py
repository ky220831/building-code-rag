import unittest

from src.text_splitter import parse_pages, split_by_paragraphs


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


if __name__ == "__main__":
    unittest.main()
