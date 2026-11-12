from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.services.retrieval import RAGEngine


PROJECT_DIR = Path(__file__).resolve().parent.parent


class RetrievalEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        policies = json.loads(
            (PROJECT_DIR / "data" / "policies.json").read_text(encoding="utf-8")
        )
        cls.engine = RAGEngine(policies)

    def test_index_metadata(self) -> None:
        metadata = self.engine.metadata()
        self.assertEqual(metadata["document_count"], 9)
        self.assertEqual(metadata["base_document_count"], 9)
        self.assertEqual(metadata["uploaded_document_count"], 0)
        self.assertEqual(metadata["chunk_count"], 36)
        self.assertEqual(metadata["evaluation_case_count"], 12)

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
        self.assertEqual(hybrid["passed"], 12)
        self.assertGreaterEqual(hybrid["passed"], keyword["passed"])
        self.assertGreater(hybrid["passed"], vector["passed"])
        self.assertEqual(hybrid["recall_at_3"], 1.0)
        self.assertEqual(hybrid["citation_accuracy"], 1.0)
        self.assertEqual(hybrid["refusal_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
