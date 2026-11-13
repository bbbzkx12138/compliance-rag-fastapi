from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ["POLICY_RAG_MODELS_ENABLED"] = "false"
os.environ["POLICY_RAG_LLM_PROVIDER"] = "none"

from app.config import POLICIES_PATH, load_policies
from app.services.retrieval import RAGEngine


PROJECT_DIR = Path(__file__).resolve().parent.parent


class RetrievalEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        policies = load_policies(POLICIES_PATH)
        cls.engine = RAGEngine(policies)

    def test_index_metadata(self) -> None:
        metadata = self.engine.metadata()
        self.assertEqual(metadata["document_count"], 9)
        self.assertEqual(metadata["base_document_count"], 9)
        self.assertEqual(metadata["uploaded_document_count"], 0)
        self.assertEqual(metadata["chunk_count"], 36)
        self.assertEqual(metadata["evaluation_case_count"], 54)

    def test_current_scope_excludes_expired_policy(self) -> None:
        response = self.engine.search(
            "旧版差旅制度里深圳住宿标准是多少？", scope="current"
        )
        document_ids = [
            result["document"]["id"] for result in response["results"][:3]
        ]
        self.assertNotIn("TRV-001-OLD", document_ids)

    def test_all_scope_can_retrieve_history(self) -> None:
        response = self.engine.search(
            "旧版差旅制度里深圳住宿标准是多少？", scope="all"
        )
        document_ids = [
            result["document"]["id"] for result in response["results"][:3]
        ]
        self.assertIn("TRV-001-OLD", document_ids)

    def test_answer_contains_policy_citation(self) -> None:
        result = self.engine.answer_question(
            "客户送2000元购物卡可以收吗？", scope="current"
        )
        self.assertFalse(result["answer"]["refused"])
        self.assertTrue(
            any(
                point["doc_id"] == "GFT-003"
                for point in result["answer"]["points"]
            )
        )

    def test_citation_numbers_match_evidence_indexes(self) -> None:
        result = self.engine.answer_question(
            "客户送2000元购物卡可以收吗？", scope="current"
        )
        answer = result["answer"]
        for point in answer["points"]:
            evidence = answer["evidence"][point["citation"] - 1]
            self.assertEqual(point["doc_id"], evidence["document"]["id"])

    def test_unsupported_question_is_refused(self) -> None:
        result = self.engine.answer_question(
            "公司可以报销宠物保险吗？", scope="current"
        )
        self.assertTrue(result["answer"]["refused"])

    def test_version_comparison_extracts_both_thresholds(self) -> None:
        result = self.engine.answer_question(
            "差旅新规相比旧规，住宿标准有什么变化？", scope="all"
        )
        text = " ".join(point["text"] for point in result["answer"]["points"])
        self.assertIn("600元", text)
        self.assertIn("500元", text)

    def test_hybrid_evaluation(self) -> None:
        keyword = self.engine.evaluate("keyword")
        vector = self.engine.evaluate("vector")
        hybrid = self.engine.evaluate("hybrid")
        self.assertGreaterEqual(hybrid["passed"], 50)
        self.assertGreaterEqual(hybrid["passed"], keyword["passed"])
        self.assertGreater(hybrid["passed"], vector["passed"])
        self.assertGreaterEqual(hybrid["recall_at_5"], 0.95)
        self.assertGreaterEqual(hybrid["citation_accuracy"], 0.95)
        self.assertGreaterEqual(hybrid["refusal_accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
