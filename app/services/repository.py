from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2


class PolicyRepository:
    """SQLite source of truth for built-in and imported policy documents."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        short_title TEXT NOT NULL,
                        version TEXT NOT NULL,
                        status TEXT NOT NULL,
                        effective_date TEXT NOT NULL,
                        expiry_date TEXT,
                        department TEXT NOT NULL,
                        category TEXT NOT NULL,
                        owner TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        tags_json TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        is_builtin INTEGER NOT NULL DEFAULT 0,
                        content_hash TEXT,
                        seeded_at TEXT
                    );

                    CREATE TABLE IF NOT EXISTS sections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT NOT NULL,
                        position INTEGER NOT NULL,
                        page INTEGER NOT NULL,
                        heading TEXT NOT NULL,
                        text TEXT NOT NULL,
                        FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS app_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_sections_document
                    ON sections(document_id, position);

                    """
                )
                self._ensure_column(
                    connection,
                    "documents",
                    "is_builtin",
                    "INTEGER NOT NULL DEFAULT 0",
                )
                self._ensure_column(connection, "documents", "content_hash", "TEXT")
                self._ensure_column(connection, "documents", "seeded_at", "TEXT")
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_documents_builtin
                    ON documents(is_builtin)
                    """
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO schema_migrations(version, applied_at)
                    VALUES (?, ?)
                    """,
                    (SCHEMA_VERSION, self._now()),
                )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _content_hash(document: dict[str, Any]) -> str:
        payload = {
            key: value
            for key, value in document.items()
            if key not in {"content_hash", "seeded_at", "is_builtin"}
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def schema_version(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT MAX(version) AS version FROM schema_migrations"
            ).fetchone()
            return int(row["version"] or 0)

    def list_documents(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM documents
                ORDER BY is_builtin DESC, effective_date DESC, id
                """
            ).fetchall()
            return [self._hydrate_document(connection, row) for row in rows]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if row is None:
                return None
            return self._hydrate_document(connection, row)

    def is_builtin(self, document_id: str) -> bool:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT is_builtin FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            return bool(row and row["is_builtin"])

    def save_document(
        self,
        document: dict[str, Any],
        *,
        is_builtin: bool = False,
    ) -> None:
        content_hash = self._content_hash(document)
        seeded_at = self._now() if is_builtin else None
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO documents (
                        id, title, short_title, version, status, effective_date,
                        expiry_date, department, category, owner, updated_at,
                        source_type, tags_json, summary, is_builtin,
                        content_hash, seeded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title = excluded.title,
                        short_title = excluded.short_title,
                        version = excluded.version,
                        status = excluded.status,
                        effective_date = excluded.effective_date,
                        expiry_date = excluded.expiry_date,
                        department = excluded.department,
                        category = excluded.category,
                        owner = excluded.owner,
                        updated_at = excluded.updated_at,
                        source_type = excluded.source_type,
                        tags_json = excluded.tags_json,
                        summary = excluded.summary,
                        is_builtin = excluded.is_builtin,
                        content_hash = excluded.content_hash,
                        seeded_at = excluded.seeded_at
                    """,
                    (
                        document["id"],
                        document["title"],
                        document["short_title"],
                        document["version"],
                        document["status"],
                        document["effective_date"],
                        document.get("expiry_date"),
                        document["department"],
                        document["category"],
                        document["owner"],
                        document["updated_at"],
                        document["source_type"],
                        json.dumps(document.get("tags", []), ensure_ascii=False),
                        document["summary"],
                        1 if is_builtin else 0,
                        content_hash,
                        seeded_at,
                    ),
                )
                connection.execute(
                    "DELETE FROM sections WHERE document_id = ?", (document["id"],)
                )
                connection.executemany(
                    """
                    INSERT INTO sections (
                        document_id, position, page, heading, text
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            document["id"],
                            index,
                            section["page"],
                            section["heading"],
                            section["text"],
                        )
                        for index, section in enumerate(document["sections"])
                    ],
                )

    def sync_builtin_documents(
        self, documents: list[dict[str, Any]]
    ) -> dict[str, Any]:
        existing = {}
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, content_hash, is_builtin
                FROM documents
                WHERE is_builtin = 1
                """
            ).fetchall()
            existing = {row["id"]: row["content_hash"] for row in rows}

        new_ids = {document["id"] for document in documents}
        removed = sorted(set(existing) - new_ids)
        changed: list[str] = []
        unchanged: list[str] = []

        with closing(self._connect()) as connection:
            with connection:
                if removed:
                    placeholders = ",".join("?" for _ in removed)
                    connection.execute(
                        f"""
                        DELETE FROM documents
                        WHERE id IN ({placeholders}) AND is_builtin = 1
                        """,
                        removed,
                    )

        for document in documents:
            document_hash = self._content_hash(document)
            if existing.get(document["id"]) == document_hash:
                unchanged.append(document["id"])
                continue
            self.save_document(document, is_builtin=True)
            changed.append(document["id"])

        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO app_metadata(key, value, updated_at)
                    VALUES ('builtin_synced_at', ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    (self._now(), self._now()),
                )

        return {
            "builtin_count": len(documents),
            "changed_ids": changed,
            "unchanged_ids": unchanged,
            "removed_ids": removed,
        }

    def delete_uploaded_document(self, document_id: str) -> bool:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    DELETE FROM documents
                    WHERE id = ? AND is_builtin = 0
                    """,
                    (document_id,),
                )
                return cursor.rowcount > 0

    def stats(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS document_count,
                    SUM(CASE WHEN is_builtin = 1 THEN 1 ELSE 0 END) AS builtin_count,
                    SUM(CASE WHEN is_builtin = 0 THEN 1 ELSE 0 END) AS uploaded_count,
                    MAX(seeded_at) AS last_seeded_at
                FROM documents
                """
            ).fetchone()
            section_count = connection.execute(
                "SELECT COUNT(*) AS count FROM sections"
            ).fetchone()["count"]
            return {
                "schema_version": self.schema_version(),
                "document_count": int(row["document_count"] or 0),
                "builtin_count": int(row["builtin_count"] or 0),
                "uploaded_count": int(row["uploaded_count"] or 0),
                "section_count": int(section_count or 0),
                "last_seeded_at": row["last_seeded_at"],
            }

    def _hydrate_document(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any]:
        sections = connection.execute(
            """
            SELECT page, heading, text
            FROM sections
            WHERE document_id = ?
            ORDER BY position
            """,
            (row["id"],),
        ).fetchall()
        return {
            "id": row["id"],
            "title": row["title"],
            "short_title": row["short_title"],
            "version": row["version"],
            "status": row["status"],
            "effective_date": row["effective_date"],
            "expiry_date": row["expiry_date"],
            "department": row["department"],
            "category": row["category"],
            "owner": row["owner"],
            "updated_at": row["updated_at"],
            "source_type": row["source_type"],
            "tags": json.loads(row["tags_json"]),
            "summary": row["summary"],
            "is_builtin": bool(row["is_builtin"]),
            "content_hash": row["content_hash"],
            "seeded_at": row["seeded_at"],
            "sections": [
                {
                    "page": section["page"],
                    "heading": section["heading"],
                    "text": section["text"],
                }
                for section in sections
            ],
        }
