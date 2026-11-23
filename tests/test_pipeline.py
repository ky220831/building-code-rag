import unittest

from src.rag_pipeline import RAGPipeline


class FakeSearch:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def search(self, query):
        self.calls += 1
        return self.results


class FakeGenerator:
    def __init__(self):
        self.calls = 0
        self.last_chunks = None

    def generate(self, query, chunks):
        self.calls += 1
        self.last_chunks = chunks
        return "根据资料，测试答案。[资料1]"


class PipelineTests(unittest.TestCase):
    def test_valid_results_flow_to_generation_and_sources(self):
        searcher = FakeSearch([{
            "chunk_id": 12,
            "source": "building-code.pdf",
            "start_page": 15,
            "end_page": 16,
            "text": "规范资料正文",
        }])
        generator = FakeGenerator()
        result = RAGPipeline(searcher, generator).run("防火分区如何设置？")
        self.assertEqual(result["answer"], "根据资料，测试答案。[资料1]")
        self.assertEqual(result["sources"], [{"source": "building-code.pdf", "page": 15}])
        self.assertEqual(generator.calls, 1)

    def test_empty_results_skip_llm_call(self):
        searcher = FakeSearch([])
        generator = FakeGenerator()
        result = RAGPipeline(searcher, generator).run("没有命中资料的问题")
        self.assertEqual(result, {"answer": "未找到相关内容。", "sources": []})
        self.assertEqual(generator.calls, 0)

    def test_invalid_and_duplicate_results_are_filtered(self):
        searcher = FakeSearch([
            {"chunk_id": "bad", "source": "a.pdf", "start_page": 1, "text": "错误 ID"},
            {"chunk_id": 1, "source": "a.pdf", "start_page": 1, "text": "有效资料"},
            {"chunk_id": 1, "source": "a.pdf", "start_page": 1, "text": "重复资料"},
            {"chunk_id": 2, "source": "a.pdf", "start_page": 0, "text": "非法页码"},
            {"chunk_id": 3, "source": "a.pdf", "start_page": 2, "text": "   "},
        ])
        generator = FakeGenerator()
        RAGPipeline(searcher, generator).run("测试问题")
        self.assertEqual(len(generator.last_chunks), 1)
        self.assertEqual(generator.last_chunks[0]["chunk_id"], 1)

    def test_blank_query_is_rejected_before_retrieval(self):
        searcher = FakeSearch([])
        generator = FakeGenerator()
        with self.assertRaises(ValueError):
            RAGPipeline(searcher, generator).run("   ")
        self.assertEqual(searcher.calls, 0)


if __name__ == "__main__":
    unittest.main()
