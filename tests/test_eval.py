import unittest

from eval.eval_retrieval import compute_metrics, format_markdown_table, is_chunk_match


class EvalMetricsTests(unittest.TestCase):
    def test_is_chunk_match_checks_all_keywords_and_optional_pages(self):
        q = {"expected_keywords": ["18℃", "3.1.1"], "expected_pages": [6]}
        self.assertTrue(is_chunk_match({"text": "3.1.1 不低于18℃", "start_page": 6, "end_page": 6}, q))
        self.assertFalse(is_chunk_match({"text": "3.1.1 不低于18℃", "start_page": 2, "end_page": 2}, q))
        self.assertFalse(is_chunk_match({"text": "只有18℃", "start_page": 6, "end_page": 6}, q))

    def test_compute_metrics_hit_and_mrr(self):
        questions = [
            {"id": 1, "query": "q1", "expected_keywords": ["A"]},
            {"id": 2, "query": "q2", "expected_keywords": ["B"]},
            {"id": 3, "query": "q3", "expected_keywords": ["C"]},
        ]

        def fake_search(q):
            if q["id"] == 1:
                return [{"text": "包含 A"}]  # rank 1 -> rr = 1.0
            if q["id"] == 2:
                return [{"text": "无关"}, {"text": "包含 B"}]  # rank 2 -> rr = 0.5
            return [{"text": "无关"}]  # miss -> rr = 0.0

        summary, details = compute_metrics("Demo", questions, fake_search, top_k=5)
        self.assertAlmostEqual(summary.hit_at_1, 1 / 3)
        self.assertAlmostEqual(summary.hit_at_5, 2 / 3)
        self.assertAlmostEqual(summary.mrr_at_5, (1.0 + 0.5) / 3)
        self.assertEqual([d["first_hit_rank"] for d in details], [1, 2, None])

        md = format_markdown_table([summary])
        self.assertIn("| Demo | 3 | 33.3% | 66.7% | 0.500 |", md)


if __name__ == "__main__":
    unittest.main()
