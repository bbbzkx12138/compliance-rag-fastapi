from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_DIR / "static"
DATA_DIR = PROJECT_DIR / "data"
POLICIES_PATH = DATA_DIR / "policies.json"
DATABASE_PATH = Path(
    os.environ.get("POLICY_RAG_DB", PROJECT_DIR / "storage" / "policy_rag.sqlite3")
)
APP_NAME = "规镜 PolicyLens"
APP_VERSION = "2.0.0"
