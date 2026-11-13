from __future__ import annotations

import ctypes
import logging
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger("policylens.semantic")


def _process_rss_mb() -> float | None:
    try:
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            return round(counters.WorkingSetSize / (1024 * 1024), 1)
    except (AttributeError, OSError):
        return None
    return None


class SemanticModelRuntime:
    """Lazy local BGE embedding and Cross-Encoder reranking."""

    def __init__(
        self,
        *,
        embedding_model_path: Path,
        reranker_model_path: Path,
        device: str = "cpu",
        query_prefix: str = "",
        embedding_batch_size: int = 32,
        rerank_max_length: int = 256,
        torch_threads: int = 4,
    ) -> None:
        self.embedding_model_path = Path(embedding_model_path)
        self.reranker_model_path = Path(reranker_model_path)
        self.device = device
        self.query_prefix = query_prefix
        self.embedding_batch_size = embedding_batch_size
        self.rerank_max_length = rerank_max_length
        self.torch_threads = torch_threads
        self._embedding_model = None
        self._reranker_model = None
        self._embedding_error: str | None = None
        self._reranker_error: str | None = None
        self._embedding_dimensions: int | None = None
        self._query_cache: dict[str, np.ndarray] = {}
        self._rerank_cache: dict[tuple[str, str], float] = {}
        self._lock = threading.RLock()

    @property
    def embedding_name(self) -> str:
        return self.embedding_model_path.name

    @property
    def reranker_name(self) -> str:
        return self.reranker_model_path.name

    def _load_embedding_model(self):
        with self._lock:
            if self._embedding_model is not None or self._embedding_error:
                return self._embedding_model
            if not self.embedding_model_path.exists():
                self._embedding_error = (
                    f"embedding model not found: {self.embedding_model_path}"
                )
                return None
            try:
                import torch
                from sentence_transformers import SentenceTransformer

                torch.set_num_threads(max(1, self.torch_threads))
                self._embedding_model = SentenceTransformer(
                    str(self.embedding_model_path), device=self.device
                )
            except Exception as error:  # pragma: no cover - model runtime path
                self._embedding_error = str(error)
                logger.exception("Failed to load embedding model")
            return self._embedding_model

    def _load_reranker_model(self):
        with self._lock:
            if self._reranker_model is not None or self._reranker_error:
                return self._reranker_model
            if not self.reranker_model_path.exists():
                self._reranker_error = (
                    f"reranker model not found: {self.reranker_model_path}"
                )
                return None
            try:
                import torch
                from sentence_transformers import CrossEncoder

                torch.set_num_threads(max(1, self.torch_threads))
                self._reranker_model = CrossEncoder(
                    str(self.reranker_model_path),
                    device=self.device,
                    max_length=self.rerank_max_length,
                )
            except Exception as error:  # pragma: no cover - model runtime path
                self._reranker_error = str(error)
                logger.exception("Failed to load reranker model")
            return self._reranker_model

    @property
    def embedding_available(self) -> bool:
        return self._load_embedding_model() is not None

    @property
    def reranker_available(self) -> bool:
        return self._load_reranker_model() is not None

    def warmup_reranker(self) -> bool:
        return self._load_reranker_model() is not None

    def encode_documents(self, texts: list[str]) -> np.ndarray | None:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        model = self._load_embedding_model()
        if model is None:
            return None
        with self._lock:
            vectors = model.encode(
                texts,
                batch_size=self.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            self._embedding_dimensions = int(vectors.shape[1])
            return vectors

    def encode_query(self, query: str) -> np.ndarray | None:
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached
        model = self._load_embedding_model()
        if model is None:
            return None
        text = f"{self.query_prefix}{query}" if self.query_prefix else query
        with self._lock:
            vector = model.encode(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]
            vector = np.asarray(vector, dtype=np.float32)
            self._embedding_dimensions = int(vector.shape[0])
            if len(self._query_cache) >= 500:
                self._query_cache.clear()
            self._query_cache[query] = vector
            return vector

    def rerank(
        self, query: str, documents: list[str], batch_size: int = 20
    ) -> np.ndarray | None:
        if not documents:
            return np.empty((0,), dtype=np.float32)
        model = self._load_reranker_model()
        if model is None:
            return None
        missing = [
            (query, document)
            for document in documents
            if (query, document) not in self._rerank_cache
        ]
        with self._lock:
            if missing:
                scores = model.predict(
                    missing,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                for pair, score in zip(
                    missing, np.asarray(scores, dtype=np.float32), strict=True
                ):
                    self._rerank_cache[pair] = float(score)
                if len(self._rerank_cache) > 5000:
                    self._rerank_cache.clear()
            return np.asarray(
                [
                    self._rerank_cache[(query, document)]
                    for document in documents
                ],
                dtype=np.float32,
            )

    @staticmethod
    def sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exponent = math.exp(value)
        return exponent / (1.0 + exponent)

    def status(self) -> dict[str, Any]:
        embedding_size_mb = self._model_size_mb(
            self.embedding_model_path, {"model.safetensors", "pytorch_model.bin"}
        )
        reranker_size_mb = self._model_size_mb(
            self.reranker_model_path, {"model.safetensors", "pytorch_model.bin"}
        )
        return {
            "embedding": {
                "name": self.embedding_name,
                "path": str(self.embedding_model_path),
                "loaded": self._embedding_model is not None,
                "available": self._embedding_error is None
                and self.embedding_model_path.exists(),
                "dimensions": self._embedding_dimensions,
                "device": self.device,
                "vram_mb": 0,
                "model_size_mb": embedding_size_mb,
                "error": self._embedding_error,
            },
            "reranker": {
                "name": self.reranker_name,
                "path": str(self.reranker_model_path),
                "loaded": self._reranker_model is not None,
                "available": self._reranker_error is None
                and self.reranker_model_path.exists(),
                "device": self.device,
                "vram_mb": 0,
                "model_size_mb": reranker_size_mb,
                "error": self._reranker_error,
            },
            "process_rss_mb": _process_rss_mb(),
            "estimated_loaded_model_mb": round(
                embedding_size_mb + reranker_size_mb, 1
            ),
            "estimated_cost_usd": 0.0,
        }

    @staticmethod
    def _model_size_mb(path: Path, filenames: set[str]) -> float:
        if not path.exists():
            return 0.0
        for filename in ("model.safetensors", "pytorch_model.bin"):
            candidate = path / filename
            if filename in filenames and candidate.exists():
                return round(candidate.stat().st_size / (1024 * 1024), 1)
        return 0.0
