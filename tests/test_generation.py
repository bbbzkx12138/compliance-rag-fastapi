from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["POLICY_RAG_MODELS_ENABLED"] = "false"
os.environ["POLICY_RAG_LLM_PROVIDER"] = "none"

from app.config import POLICIES_PATH, load_policies
from app.services.generation import RAGGenerationService
from app.services.index_manager import IndexManager
from app.services.llm import LLMProviderError, ProviderCompletion
from app.services.repository import PolicyRepository


class FakeProvider:
    enabled = True

    def __init__(
        self,
        content: str | list[str] | None = None,
        error: Exception | None = None,
    ):
        self.contents = content if isinstance(content, list) else [content or "{}"]
        self.error = error
        self.calls = 0

    def status(self):
        return {
            "provider": "fake",
            "model": "fake-rag-model",
            "enabled": True,
            "api_key_configured": True,
            "cost_currency": "CNY",
        }

    def complete(self, *, system_prompt: str, user_prompt: str):
        self.calls += 1
        if self.error is not None:
            raise self.error
        content_index = min(self.calls - 1, len(self.contents) - 1)
        return ProviderCompletion(
            content=self.contents[content_index],
            model="fake-rag-model",
            latency_ms=12.5,
            prompt_tokens=100,
            completion_tokens=30,
            total_tokens=130,
            estimated_cost=0.00017,
            cost_currency="CNY",
            request_id="fake-request",
            attempts=1,
        )


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        policies = load_policies(POLICIES_PATH)
        repository = PolicyRepository(
            Path(cls.temp_dir.name) / "generation.sqlite3"
        )
        repository.sync_builtin_documents(policies["documents"])
        cls.index_manager = IndexManager(policies, repository.list_documents())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def generate(
        self,
        provider: FakeProvider,
        question: str,
        *,
        repair_retries: int = 1,
    ):
        service = RAGGenerationService(
            self.index_manager,
            provider,
            repair_retries=repair_retries,
        )
        return asyncio.run(
            service.generate(
                question,
                scope="current",
                department="all",
                mode="hybrid",
            )
        ), service

    def test_valid_structured_answer_keeps_real_citation(self) -> None:
        payload = {
            "refused": False,
            "answer": "员工不得收受购物卡等可即时变现的财物。",
            "claims": [
                {
                    "text": "员工不得收受购物卡、预付卡和有价证券等财物。",
                    "citations": [1],
                }
            ],
            "confidence": "high",
            "refusal_reason": None,
        }
        provider = FakeProvider(json.dumps(payload, ensure_ascii=False))
        result, service = self.generate(
            provider, "客户送2000元购物卡可以收吗？"
        )
        answer = result["answer"]
        self.assertEqual(result["generation"]["mode"], "llm")
        self.assertEqual(answer["points"][0]["doc_id"], "GFT-003")
        self.assertEqual(answer["points"][0]["citation"], 1)
        self.assertEqual(result["generation"]["total_tokens"], 130)
        self.assertGreater(result["generation"]["estimated_cost"], 0)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(service.status()["metrics"]["llm_answers"], 1)

    def test_invalid_citation_falls_back_to_extractive(self) -> None:
        payload = {
            "refused": False,
            "answer": "员工可以收受购物卡。",
            "claims": [
                {
                    "text": "员工可以收受购物卡。",
                    "citations": [99],
                }
            ],
            "confidence": "high",
            "refusal_reason": None,
        }
        provider = FakeProvider(json.dumps(payload, ensure_ascii=False))
        result, service = self.generate(
            provider, "客户送2000元购物卡可以收吗？"
        )
        self.assertEqual(result["generation"]["mode"], "fallback")
        self.assertIn("不存在的证据编号", result["generation"]["fallback_reason"])
        self.assertEqual(
            service.status()["metrics"]["invalid_citation_count"], 1
        )
        self.assertEqual(result["generation"]["total_tokens"], 260)
        self.assertGreater(result["generation"]["estimated_cost"], 0)
        self.assertEqual(
            result["generation"]["citation_validation"]["code"],
            "citation_out_of_range",
        )

    def test_precheck_refusal_skips_llm_call(self) -> None:
        provider = FakeProvider("{}")
        result, service = self.generate(provider, "公司可以报销宠物保险吗？")
        self.assertTrue(result["answer"]["refused"])
        self.assertEqual(result["generation"]["mode"], "precheck_refusal")
        self.assertEqual(provider.calls, 0)
        self.assertEqual(service.status()["metrics"]["precheck_refusals"], 1)

    def test_provider_error_falls_back(self) -> None:
        provider = FakeProvider(
            error=LLMProviderError("invalid_api_key", status_code=401)
        )
        result, _ = self.generate(
            provider, "客户送2000元购物卡可以收吗？"
        )
        self.assertEqual(result["generation"]["mode"], "fallback")
        self.assertEqual(
            result["generation"]["provider_status_code"], 401
        )
        self.assertFalse(result["answer"]["refused"])

    def test_llm_refusal_is_preserved(self) -> None:
        payload = {
            "refused": True,
            "answer": "证据不足。",
            "claims": [],
            "confidence": "low",
            "refusal_reason": "现有条款不能回答该问题。",
        }
        provider = FakeProvider(json.dumps(payload, ensure_ascii=False))
        result, _ = self.generate(
            provider, "客户送2000元购物卡可以收吗？"
        )
        self.assertTrue(result["answer"]["refused"])
        self.assertEqual(result["generation"]["mode"], "llm_refusal")

    def test_repair_retry_can_fix_invalid_citation(self) -> None:
        invalid_payload = {
            "refused": False,
            "answer": "员工可以收受购物卡。",
            "claims": [
                {
                    "text": "员工可以收受购物卡。",
                    "citations": [99],
                }
            ],
            "confidence": "high",
            "refusal_reason": None,
        }
        valid_payload = {
            "refused": False,
            "answer": "员工不得收受购物卡等可即时变现的财物。",
            "claims": [
                {
                    "text": "员工不得收受购物卡、预付卡和有价证券等财物。",
                    "citations": [1],
                }
            ],
            "confidence": "high",
            "refusal_reason": None,
        }
        provider = FakeProvider(
            [
                json.dumps(invalid_payload, ensure_ascii=False),
                json.dumps(valid_payload, ensure_ascii=False),
            ]
        )
        result, service = self.generate(
            provider,
            "客户送2000元购物卡可以收吗？",
            repair_retries=1,
        )
        self.assertEqual(result["generation"]["mode"], "llm")
        self.assertEqual(result["generation"]["repair_attempts"], 1)
        self.assertEqual(result["generation"]["provider_call_count"], 2)
        self.assertEqual(result["generation"]["total_tokens"], 260)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(service.status()["metrics"]["llm_answers"], 1)

    def test_evidence_letter_citations_are_normalized(self) -> None:
        payload = {
            "refused": False,
            "answer": "员工不得收受购物卡等可即时变现的财物。",
            "claims": [
                {
                    "text": "员工不得收受购物卡、预付卡和有价证券等财物。",
                    "citations": ["E1"],
                }
            ],
            "confidence": "high",
            "refusal_reason": None,
        }
        provider = FakeProvider(json.dumps(payload, ensure_ascii=False))
        result, _ = self.generate(
            provider,
            "客户送2000元购物卡可以收吗？",
            repair_retries=0,
        )
        self.assertEqual(result["generation"]["mode"], "llm")
        self.assertEqual(result["answer"]["points"][0]["citations"], [1])


if __name__ == "__main__":
    unittest.main()
