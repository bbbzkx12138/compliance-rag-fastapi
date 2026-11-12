from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    APP_NAME,
    APP_VERSION,
    DATABASE_PATH,
    POLICIES_PATH,
    STATIC_DIR,
)
from app.schemas import (
    ChatRequest,
    DocumentCreateRequest,
    EvaluationRequest,
)
from app.services.repository import PolicyRepository
from app.services.retrieval import RAGEngine, unique_tokens


def load_policies(path: Path = POLICIES_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_content_sections(content: str) -> list[dict[str, Any]]:
    clean = re.sub(r"\r\n?", "\n", content).strip()
    if not clean:
        raise HTTPException(status_code=400, detail="文档内容不能为空")

    sections: list[dict[str, Any]] = []
    current_heading = "导入内容"
    for block in re.split(r"\n{2,}", clean):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        heading_match = re.match(r"^#{1,4}\s+(.+)$", lines[0])
        clause_match = re.match(
            r"^(第[一二三四五六七八九十百\d]+[章节条款].*)$", lines[0]
        )
        if heading_match or clause_match:
            current_heading = (
                heading_match.group(1) if heading_match else clause_match.group(1)
            ).strip()
            lines.pop(0)
        text = " ".join(lines).strip()
        if not text:
            continue
        sections.append(
            {
                "page": len(sections) + 1,
                "heading": current_heading,
                "text": text,
            }
        )
        current_heading = "导入内容"

    if not sections:
        raise HTTPException(status_code=400, detail="没有解析出可检索的文档条款")
    if len(sections) > 200:
        raise HTTPException(status_code=400, detail="单个文档最多允许 200 个条款切片")
    return sections


def summarize_document(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": document["id"],
        "title": document["title"],
        "short_title": document["short_title"],
        "version": document["version"],
        "status": document["status"],
        "status_label": {
            "effective": "现行有效",
            "expired": "已失效",
            "draft": "草稿",
        }.get(document["status"], document["status"]),
        "effective_date": document["effective_date"],
        "expiry_date": document.get("expiry_date"),
        "department": document["department"],
        "category": document["category"],
        "owner": document["owner"],
        "updated_at": document["updated_at"],
        "source_type": document["source_type"],
        "tags": document.get("tags", []),
        "summary": document["summary"],
        "section_count": len(document["sections"]),
    }


def build_uploaded_document(payload: DocumentCreateRequest) -> dict[str, Any]:
    sections = parse_content_sections(payload.content)
    document_id = (
        f"UPL-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6].upper()}"
    )
    tags = unique_tokens(
        f"{payload.title} " + " ".join(section["text"] for section in sections)
    )[:24]
    return {
        "id": document_id,
        "title": payload.title,
        "short_title": payload.title,
        "version": payload.version,
        "status": "effective",
        "effective_date": payload.effective_date.isoformat(),
        "expiry_date": None,
        "department": payload.department,
        "category": "用户导入",
        "owner": "API 用户",
        "updated_at": datetime.now(timezone.utc).date().isoformat(),
        "source_type": "导入文档",
        "tags": tags,
        "summary": sections[0]["text"][:100],
        "sections": sections,
    }


def create_app(
    *,
    database_path: Path | None = None,
    policies_path: Path | None = None,
) -> FastAPI:
    resolved_database_path = Path(database_path or DATABASE_PATH)
    policies = load_policies(Path(policies_path or POLICIES_PATH))
    repository = PolicyRepository(resolved_database_path)
    engine = RAGEngine(policies, repository.list_documents())

    application = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description="基于 FastAPI 的可追溯合规与制度 RAG 服务",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    application.state.repository = repository
    application.state.rag = engine
    application.state.started_at = datetime.now(timezone.utc)

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = str(uuid.uuid4())
        started_at = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Process-Time-Ms"] = (
            f"{(time.perf_counter() - started_at) * 1000:.2f}"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ):
        first_error = error.errors()[0] if error.errors() else {}
        location = ".".join(str(item) for item in first_error.get("loc", []))
        message = first_error.get("msg", "请求参数校验失败")
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "error": {
                    "code": 422,
                    "message": f"{location}: {message}".strip(": "),
                    "details": error.errors(),
                },
            },
        )

    @application.exception_handler(HTTPException)
    async def http_error_handler(request: Request, error: HTTPException):
        return JSONResponse(
            status_code=error.status_code,
            content={
                "ok": False,
                "error": {
                    "code": error.status_code,
                    "message": str(error.detail),
                    "details": None,
                },
            },
        )

    def current_engine() -> RAGEngine:
        return application.state.rag

    def current_repository() -> PolicyRepository:
        return application.state.repository

    def health_payload() -> dict[str, Any]:
        elapsed = datetime.now(timezone.utc) - application.state.started_at
        return {
            "ok": True,
            "status": "healthy",
            "app": APP_NAME,
            "version": APP_VERSION,
            "uptime_seconds": int(elapsed.total_seconds()),
            "storage": "sqlite",
            "meta": current_engine().metadata(),
        }

    @application.get("/api", tags=["system"])
    async def api_index():
        return {
            "ok": True,
            "name": f"{APP_NAME} API",
            "version": APP_VERSION,
            "docs": "/api/docs",
            "endpoints": {
                "health": "GET /api/health",
                "documents": "GET /api/documents",
                "document": "GET /api/documents/{document_id}",
                "chat": "POST /api/chat",
                "evaluate": "POST /api/evaluate",
                "import_document": "POST /api/documents",
                "delete_document": "DELETE /api/documents/{document_id}",
            },
        }

    @application.get("/api/health", tags=["system"])
    async def health():
        return health_payload()

    @application.get("/api/documents", tags=["documents"])
    async def list_documents(
        q: Annotated[str, Query(max_length=200)] = "",
        status: Annotated[str | None, Query(pattern="^(effective|expired|draft)$")] = None,
    ):
        documents = current_engine().documents
        if status:
            documents = [item for item in documents if item["status"] == status]
        if q.strip():
            query = re.sub(r"\s+", "", q.lower())
            documents = [
                item
                for item in documents
                if query
                in re.sub(
                    r"\s+",
                    "",
                    " ".join(
                        [
                            item["title"],
                            item.get("short_title", ""),
                            item["version"],
                            item["department"],
                            item["category"],
                            " ".join(item.get("tags", [])),
                        ]
                    ).lower(),
                )
            ]
        return {
            "ok": True,
            "meta": current_engine().metadata(),
            "documents": [summarize_document(item) for item in documents],
        }

    @application.get("/api/documents/{document_id}", tags=["documents"])
    async def get_document(document_id: str):
        document = next(
            (
                item
                for item in current_engine().documents
                if item["id"] == document_id
            ),
            None,
        )
        if document is None:
            raise HTTPException(status_code=404, detail="制度文档不存在")
        return {"ok": True, "document": document}

    @application.post("/api/chat", tags=["rag"])
    async def chat(payload: ChatRequest):
        answer_result = current_engine().answer_question(
            payload.question,
            scope=payload.scope,
            department=payload.department,
            mode=payload.mode,
            limit=6,
        )
        search = answer_result["search"]
        return {
            "ok": True,
            "question": payload.question,
            "answer": answer_result["answer"],
            "retrieval": {
                "mode": search["mode"],
                "scope": search["scope"],
                "department": search["department"],
                "elapsed_ms": search["elapsed_ms"],
                "expanded_terms": search["expanded_terms"],
                "unseen_terms": search["unseen_terms"],
                "candidate_count": len(search["results"]),
            },
            "meta": current_engine().metadata(),
        }

    @application.post("/api/evaluate", tags=["rag"])
    async def evaluate(payload: EvaluationRequest):
        modes = list(dict.fromkeys(payload.modes))
        results = {mode: current_engine().evaluate(mode) for mode in modes}
        return {
            "ok": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "case_count": len(current_engine().evaluation_cases),
            "results": results,
        }

    @application.post("/api/documents", status_code=201, tags=["documents"])
    async def create_document(payload: DocumentCreateRequest):
        document = build_uploaded_document(payload)
        current_repository().save_document(document)
        current_engine().set_uploaded_documents(
            current_repository().list_documents()
        )
        return {
            "ok": True,
            "document": document,
            "meta": current_engine().metadata(),
        }

    @application.delete("/api/documents/{document_id}", tags=["documents"])
    async def delete_document(document_id: str):
        if not document_id.startswith("UPL-"):
            raise HTTPException(status_code=403, detail="内置制度文档不能通过 API 删除")
        deleted = current_repository().delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="上传文档不存在")
        current_engine().set_uploaded_documents(
            current_repository().list_documents()
        )
        return {
            "ok": True,
            "deleted_id": document_id,
            "meta": current_engine().metadata(),
        }

    @application.get("/", include_in_schema=False)
    async def index_page():
        return FileResponse(STATIC_DIR / "index.html")

    application.mount(
        "/static", StaticFiles(directory=STATIC_DIR), name="static"
    )
    return application


app = create_app()
