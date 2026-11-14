from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["POLICY_RAG_MODELS_ENABLED"] = "false"
os.environ["POLICY_RAG_LLM_PROVIDER"] = "none"

from app.config import (  # noqa: E402
    EVAL_MIN_CITATION_ACCURACY,
    EVAL_MIN_FAITHFULNESS,
    EVAL_MIN_RECALL_AT_5,
    EVAL_MIN_REFUSAL_ACCURACY,
    POLICIES_PATH,
    load_policies,
)
from app.services.retrieval import RAGEngine  # noqa: E402


def main() -> int:
    policies = load_policies(POLICIES_PATH)
    engine = RAGEngine(policies)
    if len(engine.evaluation_cases) < 100:
        print(
            f"FAIL: evaluation set has only {len(engine.evaluation_cases)} cases"
        )
        return 1

    keyword = engine.evaluate("keyword")
    vector = engine.evaluate("vector")
    hybrid = engine.evaluate("hybrid")
    refusal_threshold = (
        EVAL_MIN_REFUSAL_ACCURACY if hybrid["semantic_enabled"] else 0.85
    )
    checks = {
        "case_count>=100": len(engine.evaluation_cases) >= 100,
        "case_ids_unique": len(
            {case["id"] for case in engine.evaluation_cases}
        )
        == len(engine.evaluation_cases),
        "negative_cases>=10": sum(
            1 for case in engine.evaluation_cases if case["should_refuse"]
        )
        >= 10,
        "recall_at_5": hybrid["recall_at_5"] >= EVAL_MIN_RECALL_AT_5,
        "citation_accuracy": (
            hybrid["citation_accuracy"] >= EVAL_MIN_CITATION_ACCURACY
        ),
        "refusal_accuracy": (
            hybrid["refusal_accuracy"] >= refusal_threshold
        ),
        "faithfulness": hybrid["faithfulness"] >= EVAL_MIN_FAITHFULNESS,
        "hybrid_not_worse_than_keyword": hybrid["passed"] >= keyword["passed"],
        "hybrid_not_worse_than_vector": hybrid["passed"] >= vector["passed"],
    }
    print("PolicyLens quality gate")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print(
        "metrics",
        {
            "cases": len(engine.evaluation_cases),
            "keyword_passed": keyword["passed"],
            "vector_passed": vector["passed"],
            "hybrid_passed": hybrid["passed"],
            "recall_at_5": round(hybrid["recall_at_5"], 4),
            "citation_accuracy": round(hybrid["citation_accuracy"], 4),
            "refusal_accuracy": round(hybrid["refusal_accuracy"], 4),
            "refusal_threshold": refusal_threshold,
            "faithfulness": round(hybrid["faithfulness"], 4),
        },
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
