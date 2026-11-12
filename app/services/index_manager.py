from __future__ import annotations

import copy
import threading
from datetime import datetime, timezone
from typing import Any

from app.services.retrieval import RAGEngine


class IndexManager:
    """Serializes reads and writes around the in-memory retrieval index."""

    def __init__(
        self,
        policies: dict[str, Any],
        documents: list[dict[str, Any]],
    ) -> None:
        self._policies = policies
        self._engine = RAGEngine(policies, documents)
        self._lock = threading.RLock()
        self._version = 1
        self._built_at = datetime.now(timezone.utc).isoformat()
        self._last_reason = "startup"

    def _record_change(self, reason: str) -> None:
        self._version += 1
        self._built_at = datetime.now(timezone.utc).isoformat()
        self._last_reason = reason

    def search(
        self,
        query: str,
        *,
        scope: str = "current",
        department: str = "all",
        mode: str = "hybrid",
        limit: int = 6,
    ) -> dict[str, Any]:
        with self._lock:
            return self._engine.search(
                query,
                scope=scope,
                department=department,
                mode=mode,
                limit=limit,
            )

    def answer_question(
        self,
        query: str,
        *,
        scope: str = "current",
        department: str = "all",
        mode: str = "hybrid",
        limit: int = 6,
    ) -> dict[str, Any]:
        with self._lock:
            return self._engine.answer_question(
                query,
                scope=scope,
                department=department,
                mode=mode,
                limit=limit,
            )

    def evaluate(self, mode: str) -> dict[str, Any]:
        with self._lock:
            return self._engine.evaluate(mode)

    def rebuild(
        self,
        documents: list[dict[str, Any]],
        *,
        reason: str = "manual_rebuild",
    ) -> dict[str, Any]:
        with self._lock:
            self._engine.rebuild(copy.deepcopy(documents))
            self._record_change(reason)
            return self.status()

    def refresh_document(
        self,
        document: dict[str, Any],
        *,
        reason: str = "document_changed",
    ) -> dict[str, Any]:
        with self._lock:
            self._engine.refresh_document(copy.deepcopy(document))
            self._record_change(reason)
            return self.status()

    def remove_document(
        self,
        document_id: str,
        *,
        reason: str = "document_deleted",
    ) -> dict[str, Any]:
        with self._lock:
            removed = self._engine.remove_document(document_id)
            if removed:
                self._record_change(reason)
            return self.status()

    @property
    def documents(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._engine.documents)

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            payload = self._engine.metadata()
            payload.update(
                {
                    "index_version": self._version,
                    "index_built_at": self._built_at,
                    "last_rebuild_reason": self._last_reason,
                }
            )
            return payload

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "meta": self.metadata(),
        }
