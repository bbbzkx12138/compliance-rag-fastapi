from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from app.services.llm import OpenAICompatibleProvider


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload
        self.headers = {"x-request-id": "unit-request-id"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class LLMProviderTests(unittest.TestCase):
    def test_qwen_request_uses_fast_non_thinking_profile(self) -> None:
        provider = OpenAICompatibleProvider(
            provider_name="qwen",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen3.8-max",
            api_key="test-key",
            max_tokens=700,
            enable_thinking=False,
            json_mode=True,
            input_cost_per_million=0.8,
            output_cost_per_million=2,
        )
        response = FakeHTTPResponse(
            {
                "id": "response-id",
                "model": "qwen3.8-max",
                "choices": [
                    {"message": {"content": '{"refused": true, "answer": "x"}'}}
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )
        with patch(
            "app.services.llm.urllib.request.urlopen",
            return_value=response,
        ) as mocked:
            completion = provider.complete(
                system_prompt="system",
                user_prompt="user",
            )

        request_body = json.loads(mocked.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(request_body["model"], "qwen3.8-max")
        self.assertEqual(request_body["max_tokens"], 700)
        self.assertFalse(request_body["enable_thinking"])
        self.assertEqual(
            request_body["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(completion.total_tokens, 15)
        self.assertAlmostEqual(completion.estimated_cost, 0.000018)


if __name__ == "__main__":
    unittest.main()
