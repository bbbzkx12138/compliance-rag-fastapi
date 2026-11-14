from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import APP_VERSION
from app.services.index_manager import IndexManager


class EvaluationService:
    def __init__(
        self,
        index_manager: IndexManager,
        reports_dir: Path,
        thresholds: dict[str, float],
    ) -> None:
        self.index_manager = index_manager
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds = thresholds
        self._lock = threading.Lock()

    def run(
        self,
        modes: list[str],
        *,
        save_report: bool = True,
    ) -> dict[str, Any]:
        selected_modes = list(dict.fromkeys(modes))
        if not selected_modes:
            raise ValueError("至少选择一个评测模式")
        with self._lock:
            started_at = datetime.now(timezone.utc)
            results = {
                mode: self.index_manager.evaluate(mode) for mode in selected_modes
            }
            report_id = (
                started_at.strftime("%Y%m%dT%H%M%S")
                + "-"
                + uuid.uuid4().hex[:8]
            )
            primary_mode = "rerank" if "rerank" in results else selected_modes[-1]
            quality_gate = self._build_quality_gate(primary_mode, results[primary_mode])
            report = {
                "report_id": report_id,
                "generated_at": started_at.isoformat(),
                "app_version": APP_VERSION,
                "case_count": results[selected_modes[0]]["total"],
                "modes": selected_modes,
                "primary_mode": primary_mode,
                "thresholds": self.thresholds,
                "quality_gate": quality_gate,
                "comparison": self._build_comparison(results),
                "index": self.index_manager.metadata(),
                "results": results,
            }
            if save_report:
                json_path = self.reports_dir / f"{report_id}.json"
                markdown_path = self.reports_dir / f"{report_id}.md"
                json_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                markdown_path.write_text(
                    self.render_markdown(report),
                    encoding="utf-8",
                )
                report["report_files"] = {
                    "json": str(json_path),
                    "markdown": str(markdown_path),
                }
            return report

    def _build_quality_gate(
        self,
        primary_mode: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        checks = {
            "recall_at_5": result["recall_at_5"]
            >= self.thresholds["recall_at_5"],
            "citation_accuracy": result["citation_accuracy"]
            >= self.thresholds["citation_accuracy"],
            "refusal_accuracy": result["refusal_accuracy"]
            >= self.thresholds["refusal_accuracy"],
            "faithfulness": result["faithfulness"]
            >= self.thresholds["faithfulness"],
        }
        return {
            "passed": all(checks.values()),
            "primary_mode": primary_mode,
            "checks": checks,
        }

    @staticmethod
    def _build_comparison(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if "rerank" in results:
            candidate = "rerank"
        elif "hybrid" in results:
            candidate = "hybrid"
        else:
            return {}
        deltas = {}
        for mode, result in results.items():
            if mode == candidate:
                continue
            deltas[f"{candidate}_vs_{mode}"] = {
                metric: round(results[candidate][metric] - result[metric], 6)
                for metric in (
                    "passed",
                    "recall_at_5",
                    "mrr",
                    "citation_accuracy",
                    "refusal_accuracy",
                    "faithfulness",
                )
            }
        return {
            "candidate": candidate,
            "deltas": deltas,
        }

    @staticmethod
    def render_markdown(report: dict[str, Any]) -> str:
        lines = [
            "# PolicyLens RAG Evaluation Report",
            "",
            f"- Report ID: `{report['report_id']}`",
            f"- Generated: `{report['generated_at']}`",
            f"- App version: `{report['app_version']}`",
            f"- Cases: `{report['case_count']}`",
            f"- Primary mode: `{report['primary_mode']}`",
            f"- Quality gate: `{'PASS' if report['quality_gate']['passed'] else 'FAIL'}`",
            "",
            "## Metrics",
            "",
            "| Mode | Passed | Recall@5 | MRR | Citation | Refusal | Faithfulness | Latency ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, result in report["results"].items():
            lines.append(
                "| {mode} | {passed}/{total} | {recall:.1%} | {mrr:.3f} | "
                "{citation:.1%} | {refusal:.1%} | {faithfulness:.1%} | {latency:.1f} |".format(
                    mode=mode,
                    passed=result["passed"],
                    total=result["total"],
                    recall=result["recall_at_5"],
                    mrr=result["mrr"],
                    citation=result["citation_accuracy"],
                    refusal=result["refusal_accuracy"],
                    faithfulness=result["faithfulness"],
                    latency=result["average_latency_ms"],
                )
            )
        lines.extend(["", "## Quality Gate", ""])
        for name, passed in report["quality_gate"]["checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
        lines.extend(["", "## Failed Cases", ""])
        for mode, result in report["results"].items():
            failed_ids = [row["id"] for row in result["rows"] if not row["passed"]]
            lines.append(
                f"- `{mode}`: {', '.join(failed_ids) if failed_ids else 'none'}"
            )
        return "\n".join(lines) + "\n"

    def list_reports(self, limit: int = 20) -> list[dict[str, Any]]:
        reports = []
        for path in sorted(self.reports_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            reports.append(
                {
                    "report_id": payload.get("report_id", path.stem),
                    "generated_at": payload.get("generated_at"),
                    "case_count": payload.get("case_count"),
                    "primary_mode": payload.get("primary_mode"),
                    "quality_gate": payload.get("quality_gate", {}).get("passed"),
                    "comparison": payload.get("comparison", {}),
                    "json_path": str(path),
                    "markdown_path": str(path.with_suffix(".md")),
                }
            )
        return reports

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        path = self.reports_dir / f"{Path(report_id).name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def report_path(self, report_id: str, extension: str) -> Path | None:
        if extension not in {"json", "md"}:
            return None
        path = self.reports_dir / f"{Path(report_id).name}.{extension}"
        return path if path.exists() else None
