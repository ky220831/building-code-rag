import unittest
from types import SimpleNamespace

from src.generation import GenerationService


class FakeClient:
    def __init__(self, content="测试答案"):
        self.content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.last_prompt = None

    def create(self, **kwargs):
        self.last_prompt = kwargs["messages"][0]["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class GenerationTests(unittest.TestCase):
    def test_prompt_contains_query_and_page_citation(self):
        chunk = {
            "source": "code.pdf",
            "start_page": 3,
            "end_page": 4,
            "text": "防火分区资料正文",
        }
        prompt = GenerationService.build_prompt("防火分区要求？", [chunk])
        self.assertIn("防火分区要求？", prompt)
        self.assertIn("code.pdf", prompt)
        self.assertIn("3~4", prompt)

    def test_generation_returns_non_empty_answer(self):
        client = FakeClient("  有依据的答案  ")
        answer = GenerationService(client=client).generate(
            "问题？",
            [{"source": "x.pdf", "start_page": 2, "end_page": 2, "text": "资料"}],
        )
        self.assertEqual(answer, "有依据的答案")
        self.assertIn("问题？", client.last_prompt)

    def test_empty_context_is_rejected(self):
        with self.assertRaises(ValueError):
            GenerationService.build_prompt("问题？", [])

    def test_empty_llm_response_is_rejected(self):
        client = FakeClient("   ")
        service = GenerationService(client=client)
        with self.assertRaises(RuntimeError):
            service.generate(
                "问题？",
                [{"source": "x.pdf", "start_page": 2, "end_page": 2, "text": "资料"}],
            )


if __name__ == "__main__":
    unittest.main()
