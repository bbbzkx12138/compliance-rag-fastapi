from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._http_requests: defaultdict[tuple[str, str, int], int] = defaultdict(int)
        self._http_duration_sum: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._http_duration_count: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._active_requests = 0
        self._retrieval_count: defaultdict[str, int] = defaultdict(int)
        self._retrieval_duration: defaultdict[str, float] = defaultdict(float)
        self._semantic_duration: defaultdict[str, float] = defaultdict(float)
        self._rerank_duration: defaultdict[str, float] = defaultdict(float)
        self._generation_count: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._generation_failures: defaultdict[str, int] = defaultdict(int)
        self._generation_tokens: defaultdict[str, int] = defaultdict(int)
        self._generation_cost: defaultdict[str, float] = defaultdict(float)

    def request_started(self) -> None:
        with self._lock:
            self._active_requests += 1

    def request_finished(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._http_requests[(method, path, status_code)] += 1
            self._http_duration_sum[(method, path)] += duration_ms
            self._http_duration_count[(method, path)] += 1

    def record_retrieval(
        self,
        *,
        mode: str,
        elapsed_ms: float,
        semantic_ms: float,
        rerank_ms: float,
    ) -> None:
        with self._lock:
            self._retrieval_count[mode] += 1
            self._retrieval_duration[mode] += elapsed_ms
            self._semantic_duration[mode] += semantic_ms
            self._rerank_duration[mode] += rerank_ms

    def record_generation(self, generation: dict[str, Any]) -> None:
        mode = str(generation.get("mode", "unknown"))
        model = str(generation.get("model") or "none")
        with self._lock:
            self._generation_count[(mode, model)] += 1
            if mode == "fallback":
                self._generation_failures[model] += 1
            self._generation_tokens[model] += int(
                generation.get("total_tokens") or 0
            )
            self._generation_cost[model] += float(
                generation.get("estimated_cost") or 0
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            http_requests = [
                {
                    "method": method,
                    "path": path,
                    "status": status,
                    "count": count,
                }
                for (method, path, status), count in sorted(
                    self._http_requests.items()
                )
            ]
            http_durations = [
                {
                    "method": method,
                    "path": path,
                    "count": self._http_duration_count[(method, path)],
                    "average_ms": self._http_duration_sum[(method, path)]
                    / self._http_duration_count[(method, path)],
                    "total_ms": self._http_duration_sum[(method, path)],
                }
                for method, path in sorted(self._http_duration_count)
            ]
            retrieval = [
                {
                    "mode": mode,
                    "count": count,
                    "average_ms": self._retrieval_duration[mode] / count,
                    "average_semantic_ms": self._semantic_duration[mode] / count,
                    "average_rerank_ms": self._rerank_duration[mode] / count,
                }
                for mode, count in sorted(self._retrieval_count.items())
            ]
            generation = [
                {
                    "mode": mode,
                    "model": model,
                    "count": count,
                }
                for (mode, model), count in sorted(self._generation_count.items())
            ]
            generation_total = sum(self._generation_count.values())
            generation_failure_total = sum(self._generation_failures.values())
            return {
                "active_requests": self._active_requests,
                "http_requests": http_requests,
                "http_durations": http_durations,
                "retrieval": retrieval,
                "generation": generation,
                "generation_failures": dict(self._generation_failures),
                "generation_failure_rate": (
                    generation_failure_total / generation_total
                    if generation_total
                    else 0.0
                ),
                "generation_tokens": dict(self._generation_tokens),
                "generation_cost": dict(self._generation_cost),
            }

    def render_prometheus(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP policylens_active_requests Current in-flight HTTP requests",
            "# TYPE policylens_active_requests gauge",
            f"policylens_active_requests {snapshot['active_requests']}",
        ]
        for item in snapshot["http_requests"]:
            lines.append(
                "policylens_http_requests_total"
                f'{{method="{self._escape(item["method"])}",'
                f'path="{self._escape(item["path"])}",'
                f'status="{item["status"]}"}} {item["count"]}'
            )
        for item in snapshot["http_durations"]:
            labels = (
                f'method="{self._escape(item["method"])}",'
                f'path="{self._escape(item["path"])}"'
            )
            lines.append(
                f"policylens_http_request_duration_ms_sum{{{labels}}} "
                f"{item['total_ms']}"
            )
            lines.append(
                f"policylens_http_request_duration_ms_count{{{labels}}} "
                f"{item['count']}"
            )
        for item in snapshot["retrieval"]:
            mode = self._escape(item["mode"])
            lines.extend(
                [
                    f'policylens_retrieval_requests_total{{mode="{mode}"}} {item["count"]}',
                    f'policylens_retrieval_duration_ms_sum{{mode="{mode}"}} {item["average_ms"] * item["count"]}',
                    f'policylens_semantic_duration_ms_sum{{mode="{mode}"}} {item["average_semantic_ms"] * item["count"]}',
                    f'policylens_rerank_duration_ms_sum{{mode="{mode}"}} {item["average_rerank_ms"] * item["count"]}',
                ]
            )
        for item in snapshot["generation"]:
            mode = self._escape(item["mode"])
            model = self._escape(item["model"])
            lines.append(
                "policylens_generation_requests_total"
                f'{{mode="{mode}",model="{model}"}} {item["count"]}'
            )
        for model, count in snapshot["generation_failures"].items():
            lines.append(
                f'policylens_generation_failures_total{{model="{self._escape(model)}"}} {count}'
            )
        lines.append(
            "policylens_generation_failure_rate "
            f"{snapshot['generation_failure_rate']}"
        )
        for model, count in snapshot["generation_tokens"].items():
            lines.append(
                f'policylens_generation_tokens_total{{model="{self._escape(model)}"}} {count}'
            )
        for model, cost in snapshot["generation_cost"].items():
            lines.append(
                f'policylens_generation_estimated_cost_total{{model="{self._escape(model)}"}} {cost}'
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _escape(value: Any) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')
