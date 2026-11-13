from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


_load_dotenv(PROJECT_DIR / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else (PROJECT_DIR / path).resolve()


STATIC_DIR = PROJECT_DIR / "static"
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
POLICIES_PATH = DATA_DIR / "policies.json"
DATABASE_PATH = _env_path(
    "POLICY_RAG_DB", PROJECT_DIR / "storage" / "policy_rag.sqlite3"
)
APP_NAME = "规镜 PolicyLens"
APP_VERSION = "2.3.0"
APP_ENV = os.environ.get("POLICY_RAG_ENV", "development")
APP_HOST = os.environ.get("POLICY_RAG_HOST", "127.0.0.1")
APP_PORT = _env_int("POLICY_RAG_PORT", 8000)
APP_API_BASE_URL = os.environ.get(
    "POLICY_RAG_API_URL", f"http://{APP_HOST}:{APP_PORT}"
).rstrip("/")
APP_RELOAD = _env_bool("POLICY_RAG_RELOAD", False)
APP_LOG_LEVEL = os.environ.get("POLICY_RAG_LOG_LEVEL", "INFO").upper()
APP_LOG_JSON = _env_bool("POLICY_RAG_LOG_JSON", False)
WEB_CONCURRENCY = _env_int("WEB_CONCURRENCY", 1)
MODELS_ENABLED = _env_bool("POLICY_RAG_MODELS_ENABLED", True)
MODEL_DEVICE = os.environ.get("POLICY_RAG_MODEL_DEVICE", "cpu")
TORCH_THREADS = _env_int("POLICY_RAG_TORCH_THREADS", 4)
EMBEDDING_MODEL_PATH = _env_path(
    "POLICY_RAG_EMBEDDING_MODEL",
    MODELS_DIR / "bge-small-zh-v1.5",
)
EMBEDDING_QUERY_PREFIX = os.environ.get(
    "POLICY_RAG_EMBEDDING_QUERY_PREFIX",
    "为这个句子生成表示以用于检索相关文章：",
)
EMBEDDING_BATCH_SIZE = _env_int("POLICY_RAG_EMBEDDING_BATCH_SIZE", 32)
RERANK_ENABLED = _env_bool("POLICY_RAG_RERANK_ENABLED", True)
RERANK_MODEL_PATH = _env_path(
    "POLICY_RAG_RERANK_MODEL",
    MODELS_DIR / "bge-reranker-base",
)
RERANK_CANDIDATES = _env_int("POLICY_RAG_RERANK_CANDIDATES", 20)
RERANK_MAX_LENGTH = _env_int("POLICY_RAG_RERANK_MAX_LENGTH", 256)
LLM_PROVIDER = os.environ.get("POLICY_RAG_LLM_PROVIDER", "none").strip().lower()
LLM_MODEL = os.environ.get("POLICY_RAG_LLM_MODEL", "").strip()
LLM_BASE_URL = os.environ.get("POLICY_RAG_LLM_BASE_URL", "").strip().rstrip("/")
LLM_API_KEY = os.environ.get("POLICY_RAG_LLM_API_KEY", "").strip()
LLM_TIMEOUT_SECONDS = _env_int("POLICY_RAG_LLM_TIMEOUT_SECONDS", 90)
LLM_MAX_RETRIES = _env_int("POLICY_RAG_LLM_MAX_RETRIES", 2)
LLM_REPAIR_RETRIES = _env_int("POLICY_RAG_LLM_REPAIR_RETRIES", 1)
LLM_TEMPERATURE = float(os.environ.get("POLICY_RAG_LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = _env_int("POLICY_RAG_LLM_MAX_TOKENS", 700)
LLM_EVIDENCE_LIMIT = _env_int("POLICY_RAG_LLM_EVIDENCE_LIMIT", 4)
LLM_ENABLE_THINKING = _env_bool("POLICY_RAG_LLM_ENABLE_THINKING", False)
LLM_JSON_MODE = _env_bool("POLICY_RAG_LLM_JSON_MODE", True)
LLM_INPUT_COST_PER_MILLION = float(
    os.environ.get("POLICY_RAG_LLM_INPUT_COST_PER_MILLION", "0")
)
LLM_OUTPUT_COST_PER_MILLION = float(
    os.environ.get("POLICY_RAG_LLM_OUTPUT_COST_PER_MILLION", "0")
)
LLM_COST_CURRENCY = os.environ.get("POLICY_RAG_LLM_COST_CURRENCY", "CNY")


def load_policies(path: Path = POLICIES_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        policies = json.load(handle)
    extended_cases_path = path.with_name("evaluation_cases_extended.json")
    if extended_cases_path.exists():
        extended_cases = json.loads(
            extended_cases_path.read_text(encoding="utf-8")
        )
        existing_ids = {
            item["id"] for item in policies.get("evaluation_cases", [])
        }
        policies.setdefault("evaluation_cases", []).extend(
            item for item in extended_cases if item["id"] not in existing_ids
        )
    return policies
