from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any


class RequestTraceStore:
    def __init__(self, max_records: int = 500) -> None:
        self.max_records = max_records
        self._lock = threading.Lock()
        self._records: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def start(
        self,
        request_id: str,
        *,
        method: str,
        path: str,
        client: str,
    ) -> None:
        with self._lock:
            self._records[request_id] = {
                "request_id": request_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "method": method,
                "path": path,
                "client": client,
                "status": None,
                "duration_ms": None,
                "operation": None,
                "error": None,
            }
            while len(self._records) > self.max_records:
                self._records.popitem(last=False)

    def finish(
        self,
        request_id: str,
        *,
        status: int,
        duration_ms: float,
        operation: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            record.update(
                {
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                    "duration_ms": round(duration_ms, 2),
                    "operation": copy.deepcopy(operation),
                    "error": error,
                }
            )

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(request_id)
            return copy.deepcopy(record) if record is not None else None

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._records.values())[-limit:]
            return [copy.deepcopy(record) for record in reversed(records)]
