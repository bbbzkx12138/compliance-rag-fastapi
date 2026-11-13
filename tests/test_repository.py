from __future__ import annotations

import json
import os
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

os.environ["POLICY_RAG_MODELS_ENABLED"] = "false"

from app.services.index_manager import IndexManager
from app.services.repository import PolicyRepository


PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_test_policies() -> dict:
    return json.loads(
        (PROJECT_DIR / "data" / "policies.json").read_text(encoding="utf-8")
    )


def uploaded_document() -> dict:
    return {
        "id": "UPL-TEST-001",
        "title": "客户投诉处理办法",
        "short_title": "客户投诉处理办法",
        "version": "V1.0",
        "status": "effective",
        "effective_date": "2026-11-12",
        "expiry_date": None,
        "department": "客户服务部",
        "category": "用户导入",
        "owner": "测试用户",
        "updated_at": "2026-11-12",
        "source_type": "导入文档",
        "tags": ["客户", "投诉", "受理"],
        "summary": "客户投诉应在2个工作日内完成首次受理。",
        "is_builtin": False,
        "sections": [
            {
                "page": 1,
                "heading": "第一条 受理时限",
                "text": "客户投诉应在收到投诉后的2个工作日内完成首次受理。",
            }
        ],
    }


class RepositoryTests(unittest.TestCase):
    def test_builtin_seed_is_idempotent(self) -> None:
        policies = load_test_policies()
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "seed.sqlite3"
            repository = PolicyRepository(database_path)

            first = repository.sync_builtin_documents(policies["documents"])
            second = repository.sync_builtin_documents(policies["documents"])

            self.assertEqual(first["builtin_count"], 9)
            self.assertEqual(len(first["changed_ids"]), 9)
            self.assertEqual(second["changed_ids"], [])
            self.assertEqual(repository.stats()["builtin_count"], 9)
            self.assertEqual(repository.stats()["document_count"], 9)
            self.assertTrue(repository.is_builtin("TRV-001"))

            changed_policies = json.loads(json.dumps(policies["documents"]))
            changed_policies[0]["summary"] = "更新后的差旅制度摘要"
            third = repository.sync_builtin_documents(changed_policies)
            self.assertEqual(third["changed_ids"], ["TRV-001"])
            self.assertEqual(
                repository.get_document("TRV-001")["summary"],
                "更新后的差旅制度摘要",
            )

    def test_legacy_database_is_migrated_and_upload_preserved(self) -> None:
        policies = load_test_policies()
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "legacy.sqlite3"
            connection = sqlite3.connect(database_path)
            try:
                connection.executescript(
                    """
                    CREATE TABLE documents (
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
                    CREATE TABLE sections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT NOT NULL,
                        position INTEGER NOT NULL,
                        page INTEGER NOT NULL,
                        heading TEXT NOT NULL,
                        text TEXT NOT NULL
                    );
                    """
                )
                document = uploaded_document()
                connection.execute(
                    """
                    INSERT INTO documents VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        document["id"],
                        document["title"],
                        document["short_title"],
                        document["version"],
                        document["status"],
                        document["effective_date"],
                        document["expiry_date"],
                        document["department"],
                        document["category"],
                        document["owner"],
                        document["updated_at"],
                        document["source_type"],
                        json.dumps(document["tags"], ensure_ascii=False),
                        document["summary"],
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO sections (
                        document_id, position, page, heading, text
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        document["id"],
                        0,
                        document["sections"][0]["page"],
                        document["sections"][0]["heading"],
                        document["sections"][0]["text"],
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            repository = PolicyRepository(database_path)
            repository.sync_builtin_documents(policies["documents"])
            stats = repository.stats()
            preserved = repository.get_document("UPL-TEST-001")

            self.assertEqual(stats["schema_version"], 3)
            self.assertEqual(stats["builtin_count"], 9)
            self.assertEqual(stats["uploaded_count"], 1)
            self.assertEqual(stats["document_count"], 10)
            self.assertIsNotNone(preserved)
            self.assertFalse(preserved["is_builtin"])

    def test_database_survives_repository_restart(self) -> None:
        policies = load_test_policies()
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "restart.sqlite3"
            repository = PolicyRepository(database_path)
            repository.sync_builtin_documents(policies["documents"])
            repository.save_document(uploaded_document(), is_builtin=False)

            restarted = PolicyRepository(database_path)
            self.assertEqual(restarted.stats()["document_count"], 10)
            self.assertIsNotNone(restarted.get_document("UPL-TEST-001"))
            self.assertTrue(restarted.delete_uploaded_document("UPL-TEST-001"))
            self.assertEqual(restarted.stats()["document_count"], 9)

    def test_embedding_persistence_and_cascade_delete(self) -> None:
        policies = load_test_policies()
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "vectors.sqlite3"
            repository = PolicyRepository(database_path)
            repository.sync_builtin_documents(policies["documents"])
            document = uploaded_document()
            repository.save_document(document, is_builtin=False)
            repository.save_embeddings(
                "test-model",
                [
                    {
                        "chunk_key": "UPL-TEST-001-1-1",
                        "document_id": document["id"],
                        "dimensions": 4,
                        "text_hash": "hash",
                        "vector": b"\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@",
                    }
                ],
            )
            stored = repository.load_embeddings("test-model")
            self.assertIn("UPL-TEST-001-1-1", stored)
            self.assertEqual(stored["UPL-TEST-001-1-1"]["dimensions"], 4)
            self.assertTrue(repository.delete_uploaded_document(document["id"]))
            self.assertEqual(repository.load_embeddings("test-model"), {})

    def test_sqlite_vec_search(self) -> None:
        policies = load_test_policies()
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "sqlite-vec.sqlite3"
            repository = PolicyRepository(database_path)
            if not repository.vector_index_available:
                self.skipTest("sqlite-vec extension is unavailable")
            repository.sync_builtin_documents(policies["documents"])
            vector = struct.pack("512f", *([0.0] * 511 + [1.0]))
            repository.save_embeddings(
                "test-512",
                [
                    {
                        "chunk_key": "TRV-001-1-1",
                        "document_id": "TRV-001",
                        "dimensions": 512,
                        "text_hash": "hash",
                        "vector": vector,
                    }
                ],
            )
            hits = repository.search_similar_vectors(
                "test-512",
                [0.0] * 511 + [1.0],
                limit=5,
            )
            self.assertIn("TRV-001-1-1", hits)
            self.assertGreater(hits["TRV-001-1-1"], 0.99)

    def test_index_manager_supports_refresh_and_removal(self) -> None:
        policies = load_test_policies()
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "index.sqlite3"
            repository = PolicyRepository(database_path)
            repository.sync_builtin_documents(policies["documents"])
            manager = IndexManager(policies, repository.list_documents())

            self.assertEqual(manager.metadata()["document_count"], 9)
            self.assertEqual(manager.metadata()["index_version"], 1)

            document = uploaded_document()
            manager.refresh_document(document, reason="test_create")
            self.assertEqual(manager.metadata()["document_count"], 10)
            self.assertEqual(manager.metadata()["index_version"], 2)
            self.assertEqual(manager.metadata()["last_rebuild_reason"], "test_create")

            manager.remove_document(document["id"], reason="test_delete")
            self.assertEqual(manager.metadata()["document_count"], 9)
            self.assertEqual(manager.metadata()["index_version"], 3)

            manager.rebuild(repository.list_documents(), reason="test_full")
            self.assertEqual(manager.metadata()["document_count"], 9)
            self.assertEqual(manager.metadata()["index_version"], 4)


if __name__ == "__main__":
    unittest.main()
