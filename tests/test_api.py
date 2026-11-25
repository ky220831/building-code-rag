import unittest
from fastapi.testclient import TestClient

from src.api import create_app
from src.rag_pipeline import RAGPipeline


class FakeSearch:
    def search(self, query):
        return [{
            "chunk_id": 1,
            "source": "code.pdf",
            "start_page": 7,
            "end_page": 7,
            "text": "测试资料",
        }]

    def list_sources(self):
        return ["code.pdf"]


class FakeGenerator:
    def generate(self, query, chunks):
        return "测试回答 [资料1]"

    def generate_stream(self, query, chunks):
        yield "测试"
        yield "回答 [资料1]"


class ApiTests(unittest.TestCase):
    def setUp(self):
        pipeline = RAGPipeline(FakeSearch(), FakeGenerator())
        self.client = TestClient(create_app(pipeline=pipeline))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_chat_returns_answer_and_sources(self):
        response = self.client.post("/chat", json={"query": "問題？"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "测试回答 [资料1]")
        self.assertEqual(response.json()["sources"], [{"source": "code.pdf", "page": 7}])

    def test_blank_query_is_rejected(self):
        response = self.client.post("/chat", json={"query": "   "})
        self.assertEqual(response.status_code, 422)

    def test_documents_endpoint(self):
        response = self.client.get("/documents")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"documents": ["code.pdf"]})


    def test_chat_stream_endpoint(self):
        response = self.client.post("/chat/stream", json={"query": "问题？"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        body = response.text
        self.assertIn("event: delta", body)
        self.assertIn("event: sources", body)
        self.assertIn("event: done", body)
        self.assertIn("code.pdf", body)


if __name__ == "__main__":
    unittest.main()
