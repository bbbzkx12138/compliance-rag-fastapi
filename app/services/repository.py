from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


class PolicyRepository:
    """SQLite persistence for documents imported through the API."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.executescript(
                    """
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
                    summary TEXT NOT NULL
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

                CREATE INDEX IF NOT EXISTS idx_sections_document
                ON sections(document_id, position);
                """
                )

    def list_documents(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY effective_date DESC, id"
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

    def save_document(self, document: dict[str, Any]) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                INSERT INTO documents (
                    id, title, short_title, version, status, effective_date,
                    expiry_date, department, category, owner, updated_at,
                    source_type, tags_json, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    ),
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

    def delete_document(self, document_id: str) -> bool:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM documents WHERE id = ?", (document_id,)
                )
                return cursor.rowcount > 0

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
            "sections": [
                {
                    "page": section["page"],
                    "heading": section["heading"],
                    "text": section["text"],
                }
                for section in sections
            ],
        }
