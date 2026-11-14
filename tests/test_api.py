from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn

os.environ["POLICY_RAG_MODELS_ENABLED"] = "false"
os.environ["POLICY_RAG_LLM_PROVIDER"] = "none"

from app.main import create_app


class ApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(cls.temp_dir.name) / "test-policy-rag.sqlite3"
        cls.reports_dir = Path(cls.temp_dir.name) / "evaluation_reports"
        cls.app = create_app(
            database_path=database_path,
            evaluation_reports_dir=cls.reports_dir,
        )
        config = uvicorn.Config(
            cls.app,
            host="127.0.0.1",
            port=0,
            log_level="critical",
        )
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        for _ in range(100):
            if cls.server.started:
                break
            time.sleep(0.03)
        if not cls.server.started:
            raise RuntimeError("uvicorn test server failed to start")
        port = cls.server.servers[0].sockets[0].getsockname()[1]
        cls.base_url = f"http://127.0.0.1:{port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.should_exit = True
        cls.thread.join(timeout=5)
        cls.temp_dir.cleanup()

    def request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        include_headers: bool = False,
    ):
        data = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/json",
                **(
                    {"Content-Type": "application/json; charset=utf-8"}
                    if data
                    else {}
                ),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                body = response.read().decode("utf-8")
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    parsed = json.loads(body)
                    if include_headers:
                        return response.status, parsed, response.headers
                    return response.status, parsed
                if include_headers:
                    return response.status, body, response.headers
                return response.status, body
        except urllib.error.HTTPError as error:
            with error:
                body = error.read().decode("utf-8")
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    parsed = body
                if include_headers:
                    return error.code, parsed, error.headers
                return error.code, parsed

    def test_static_page(self) -> None:
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("规镜", body)

    def test_health(self) -> None:
        status, body = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "healthy")
        self.assertEqual(body["schema_version"], 4)
        self.assertEqual(body["meta"]["document_count"], 9)
        self.assertEqual(body["meta"]["chunk_count"], 36)
        self.assertEqual(body["database"]["builtin_count"], 9)
        self.assertEqual(body["database"]["uploaded_count"], 0)
        self.assertFalse(body["models"]["enabled"])

    def test_chat_and_refusal(self) -> None:
        status, body = self.request(
            "POST",
            "/api/chat",
            {
                "question": "发现疑似数据泄露后多久要报告？",
                "scope": "current",
                "mode": "hybrid",
            },
        )
        self.assertEqual(status, 200)
        self.assertFalse(body["answer"]["refused"])
        self.assertTrue(
            any(
                point["doc_id"] == "SEC-004"
                for point in body["answer"]["points"]
            )
        )

        status, refusal = self.request(
            "POST",
            "/api/chat",
            {"question": "公司可以报销宠物保险吗？"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(refusal["answer"]["refused"])

    def test_validation_error(self) -> None:
        status, body, headers = self.request(
            "POST",
            "/api/chat",
            {"question": "", "mode": "invalid"},
            include_headers=True,
        )
        self.assertEqual(status, 422)
        self.assertFalse(body["ok"])
        request_id = headers.get("X-Request-Id")
        self.assertTrue(request_id)
        status, trace = self.request(
            "GET", f"/api/requests/{request_id}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(trace["request"]["status"], 422)
        self.assertIn("ValidationError", trace["request"]["error"])

    def test_evaluation(self) -> None:
        status, body = self.request(
            "POST", "/api/evaluate", {"modes": ["hybrid"]}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["case_count"], 120)
        self.assertGreaterEqual(body["results"]["hybrid"]["passed"], 100)

    def test_import_query_and_delete_persistent_document(self) -> None:
        status, imported = self.request(
            "POST",
            "/api/documents",
            {
                "title": "客户投诉处理办法",
                "version": "V1.0",
                "department": "客户服务部",
                "effective_date": "2026-11-12",
                "content": (
                    "# 第一条 受理时限\n"
                    "客户投诉应在收到投诉后的2个工作日内完成首次受理。\n\n"
                    "# 第二条 升级机制\n"
                    "涉及数据安全的投诉应立即升级至合规部。"
                ),
            },
        )
        self.assertEqual(status, 201)
        document_id = imported["document"]["id"]
        self.assertTrue(document_id.startswith("UPL-"))
        self.assertEqual(imported["meta"]["document_count"], 10)
        self.assertEqual(imported["meta"]["chunk_count"], 38)
        self.assertEqual(imported["meta"]["index_version"], 2)
        self.assertEqual(imported["meta"]["last_rebuild_reason"], "document_created")

        status, chat = self.request(
            "POST",
            "/api/chat",
            {
                "question": "客户投诉几个工作日内完成首次受理？",
                "scope": "current",
                "department": "客户服务部",
            },
        )
        self.assertEqual(status, 200)
        self.assertFalse(chat["answer"]["refused"])
        self.assertTrue(
            any(
                "2个工作日" in point["text"]
                for point in chat["answer"]["points"]
            )
        )

        status, deleted = self.request(
            "DELETE", f"/api/documents/{document_id}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(deleted["meta"]["document_count"], 9)
        self.assertEqual(deleted["meta"]["index_version"], 3)

    def test_builtin_document_cannot_be_deleted(self) -> None:
        status, body = self.request("DELETE", "/api/documents/TRV-001")
        self.assertEqual(status, 403)
        self.assertFalse(body["ok"])

    def test_index_status_and_manual_rebuild(self) -> None:
        status, before = self.request("GET", "/api/index/status")
        self.assertEqual(status, 200)
        self.assertEqual(before["database"]["builtin_count"], 9)
        self.assertGreaterEqual(before["meta"]["index_version"], 1)

        status, rebuilt = self.request("POST", "/api/index/rebuild")
        self.assertEqual(status, 200)
        self.assertEqual(rebuilt["meta"]["document_count"], 9)
        self.assertEqual(rebuilt["meta"]["last_rebuild_reason"], "manual_rebuild")
        self.assertEqual(rebuilt["meta"]["index_version"], before["meta"]["index_version"] + 1)

    def test_model_status_endpoint(self) -> None:
        status, body = self.request("GET", "/api/models/status")
        self.assertEqual(status, 200)
        self.assertFalse(body["enabled"])

    def test_metrics_and_prometheus_endpoints(self) -> None:
        status, body = self.request("GET", "/api/metrics")
        self.assertEqual(status, 200)
        self.assertIn("http", body)
        self.assertIn("generation_service", body)

        status, text = self.request("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertIn("policylens_active_requests", text)

    def test_request_trace_and_feedback(self) -> None:
        status, chat = self.request(
            "POST",
            "/api/chat",
            {
                "question": "客户送2000元购物卡可以收吗？",
                "scope": "current",
                "mode": "hybrid",
            },
        )
        self.assertEqual(status, 200)
        request_id = chat["request_id"]
        status, trace = self.request("GET", f"/api/requests/{request_id}")
        self.assertEqual(status, 200)
        self.assertEqual(trace["request"]["request_id"], request_id)
        self.assertIsNotNone(trace["request"]["operation"])

        status, feedback = self.request(
            "POST",
            "/api/feedback",
            {
                "request_id": request_id,
                "helpful": True,
                "rating": 5,
                "comment": "答案和引用清楚",
            },
        )
        self.assertEqual(status, 201)
        self.assertGreaterEqual(feedback["stats"]["total"], 1)

    def test_evaluation_report_is_saved_and_listed(self) -> None:
        status, report = self.request(
            "POST",
            "/api/evaluate",
            {"modes": ["keyword", "hybrid"], "save_report": True},
        )
        self.assertEqual(status, 200)
        self.assertIn("report_id", report)
        self.assertTrue(Path(report["report_files"]["json"]).exists())
        self.assertTrue(Path(report["report_files"]["markdown"]).exists())

        status, listing = self.request("GET", "/api/evaluation/reports")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(listing["reports"]), 1)

        status, markdown = self.request(
            "GET",
            f"/api/evaluation/reports/{report['report_id']}/markdown",
        )
        self.assertEqual(status, 200)
        self.assertIn("# PolicyLens RAG Evaluation Report", markdown)


if __name__ == "__main__":
    unittest.main()
