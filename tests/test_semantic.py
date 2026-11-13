from __future__ import annotations

import os
import unittest

import numpy as np

from app.config import (
    EMBEDDING_MODEL_PATH,
    RERANK_MAX_LENGTH,
    RERANK_MODEL_PATH,
)
from app.services.semantic import SemanticModelRuntime


@unittest.skipUnless(
    os.environ.get("POLICY_RAG_RUN_MODEL_TESTS") == "1",
    "set POLICY_RAG_RUN_MODEL_TESTS=1 to run local model smoke tests",
)
class SemanticModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = SemanticModelRuntime(
            embedding_model_path=EMBEDDING_MODEL_PATH,
            reranker_model_path=RERANK_MODEL_PATH,
            device="cpu",
            query_prefix="为这个句子生成表示以用于检索相关文章：",
            embedding_batch_size=8,
            rerank_max_length=RERANK_MAX_LENGTH,
            torch_threads=4,
        )

    def test_embedding_prefers_related_clause(self) -> None:
        query = self.runtime.encode_query("酒店费最多报多少？")
        documents = self.runtime.encode_documents(
            [
                "员工住宿标准为每晚上限600元。",
                "员工不得收受供应商提供的购物卡。",
            ]
        )
        related = float(np.dot(query, documents[0]))
        unrelated = float(np.dot(query, documents[1]))
        self.assertGreater(related, unrelated)

    def test_reranker_prefers_related_clause(self) -> None:
        scores = self.runtime.rerank(
            "酒店费最多报多少？",
            [
                "员工住宿标准为每晚上限600元。",
                "员工不得收受供应商提供的购物卡。",
            ],
        )
        self.assertGreater(float(scores[0]), float(scores[1]))


if __name__ == "__main__":
    unittest.main()
