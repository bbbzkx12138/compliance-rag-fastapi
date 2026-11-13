from __future__ import annotations

import logging
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
    APP_ENV,
    APP_LOG_JSON,
    APP_LOG_LEVEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_QUERY_PREFIX,
    MODEL_DEVICE,
    APP_NAME,
    APP_VERSION,
    DATABASE_PATH,
    MODELS_ENABLED,
    POLICIES_PATH,
    RERANK_CANDIDATES,
    RERANK_ENABLED,
    RERANK_MAX_LENGTH,
    RERANK_MODEL_PATH,
    LLM_REPAIR_RETRIES,
    LLM_EVIDENCE_LIMIT,
    STATIC_DIR,
    TORCH_THREADS,
    WEB_CONCURRENCY,
    load_policies,
)
from app.logging_config import configure_logging
from app.schemas import (
    ChatRequest,
    DocumentCreateRequest,
    EvaluationRequest,
)
from app.services.index_manager import IndexManager
from app.services.generation import RAGGenerationService
from app.services.llm import OpenAICompatibleProvider, create_llm_provider
from app.services.repository import PolicyRepository
from app.services.retrieval import unique_tokens
from app.services.semantic import SemanticModelRuntime


logger = logging.getLogger("policylens.http")

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
        "is_builtin": bool(document.get("is_builtin")),
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
        "is_builtin": False,
        "sections": sections,
    }


def create_app(
    *,
    database_path: Path | None = None,
    policies_path: Path | None = None,
    llm_provider: OpenAICompatibleProvider | None = None,
) -> FastAPI:
    configure_logging(APP_LOG_LEVEL, json_output=APP_LOG_JSON)
    resolved_database_path = Path(database_path or DATABASE_PATH)
    policies = load_policies(Path(policies_path or POLICIES_PATH))
    repository = PolicyRepository(resolved_database_path)
    seed_result = repository.sync_builtin_documents(policies["documents"])
    documents = repository.list_documents()
    semantic_runtime = (
        SemanticModelRuntime(
            embedding_model_path=EMBEDDING_MODEL_PATH,
            reranker_model_path=RERANK_MODEL_PATH,
            device=MODEL_DEVICE,
            query_prefix=EMBEDDING_QUERY_PREFIX,
            embedding_batch_size=EMBEDDING_BATCH_SIZE,
            rerank_max_length=RERANK_MAX_LENGTH,
            torch_threads=TORCH_THREADS,
        )
        if MODELS_ENABLED
        else None
    )
    index_manager = IndexManager(
        policies,
        documents,
        semantic_runtime=semantic_runtime,
        vector_repository=repository if semantic_runtime is not None else None,
        rerank_enabled=MODELS_ENABLED and RERANK_ENABLED,
        rerank_candidates=RERANK_CANDIDATES,
    )
    if semantic_runtime is not None and RERANK_ENABLED:
        semantic_runtime.warmup_reranker()
    resolved_llm_provider = (
        llm_provider if llm_provider is not None else create_llm_provider()
    )
    generation_service = RAGGenerationService(
        index_manager,
        resolved_llm_provider,
        repair_retries=LLM_REPAIR_RETRIES,
        evidence_limit=LLM_EVIDENCE_LIMIT,
    )

    if WEB_CONCURRENCY > 1:
        logger.warning(
            "检测到 WEB_CONCURRENCY=%s；当前索引为进程内存索引，"
            "建议使用单 worker 或共享索引服务",
            WEB_CONCURRENCY,
        )

    application = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description="基于 FastAPI 的可追溯合规与制度 RAG 服务",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    application.state.repository = repository
    application.state.index_manager = index_manager
    application.state.seed_result = seed_result
    application.state.semantic_runtime = semantic_runtime
    application.state.llm_provider = resolved_llm_provider
    application.state.generation_service = generation_service
    application.state.schema_version = repository.schema_version()
    application.state.started_at = datetime.now(timezone.utc)

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        started_at = time.perf_counter()
        response = None
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "%s %s -> %s",
                request.method,
                request.url.path,
                status_code,
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 2),
                    "client": request.client.host if request.client else "-",
                    "index_version": application.state.index_manager.metadata()[
                        "index_version"
                    ],
                },
            )
            if response is not None:
                response.headers["X-Request-Id"] = request_id
                response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["Referrer-Policy"] = "same-origin"
                if request.url.path.startswith("/api"):
                    response.headers["Cache-Control"] = "no-store"

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
            headers={
                "X-Request-Id": getattr(request.state, "request_id", "-"),
                "X-Content-Type-Options": "nosniff",
            },
        )

    @application.exception_handler(HTTPException)
    async def http_error_handler(request: Request, error: HTTPException):
        logger.warning(
            "HTTP %s: %s",
            error.status_code,
            error.detail,
            extra={
                "request_id": getattr(request.state, "request_id", "-"),
                "error_code": error.status_code,
            },
        )
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
            headers={
                "X-Request-Id": getattr(request.state, "request_id", "-"),
                "X-Content-Type-Options": "nosniff",
            },
        )

    @application.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, error: Exception):
        logger.exception(
            "Unhandled application error",
            extra={
                "request_id": getattr(request.state, "request_id", "-"),
                "error_code": 500,
            },
        )
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {
                    "code": 500,
                    "message": "服务器处理请求时发生错误",
                    "details": None,
                },
            },
            headers={
                "X-Request-Id": getattr(request.state, "request_id", "-"),
                "X-Content-Type-Options": "nosniff",
            },
        )

    def current_index() -> IndexManager:
        return application.state.index_manager

    def current_repository() -> PolicyRepository:
        return application.state.repository

    def health_payload() -> dict[str, Any]:
        elapsed = datetime.now(timezone.utc) - application.state.started_at
        return {
            "ok": True,
            "status": "healthy",
            "app": APP_NAME,
            "version": APP_VERSION,
            "environment": APP_ENV,
            "uptime_seconds": int(elapsed.total_seconds()),
            "storage": "sqlite",
            "schema_version": application.state.schema_version,
            "seed": application.state.seed_result,
            "database": current_repository().stats(),
            "meta": current_index().metadata(),
            "models": (
                semantic_runtime.status()
                if semantic_runtime is not None
                else {"enabled": False}
            ),
            "generation": generation_service.status(),
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
                "index_status": "GET /api/index/status",
                "index_rebuild": "POST /api/index/rebuild",
                "model_status": "GET /api/models/status",
                "generation_status": "GET /api/generation/status",
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
        documents = current_index().documents
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
            "meta": current_index().metadata(),
            "documents": [summarize_document(item) for item in documents],
        }

    @application.get("/api/documents/{document_id}", tags=["documents"])
    async def get_document(document_id: str):
        document = next(
            (
                item
                for item in current_index().documents
                if item["id"] == document_id
            ),
            None,
        )
        if document is None:
            raise HTTPException(status_code=404, detail="制度文档不存在")
        return {"ok": True, "document": document}

    @application.post("/api/chat", tags=["rag"])
    async def chat(payload: ChatRequest, request: Request):
        generation_result = await generation_service.generate(
            payload.question,
            scope=payload.scope,
            department=payload.department,
            mode=payload.mode,
            trace_id=request.state.request_id,
        )
        answer_result = generation_result
        search = answer_result["search"]
        return {
            "ok": True,
            "question": payload.question,
            "answer": answer_result["answer"],
            "generation": generation_result["generation"],
            "retrieval": {
                "mode": search["mode"],
                "scope": search["scope"],
                "department": search["department"],
                "elapsed_ms": search["elapsed_ms"],
                "expanded_terms": search["expanded_terms"],
                "unseen_terms": search["unseen_terms"],
                "candidate_count": len(search["results"]),
                "query_rewrites": search["query_rewrites"],
                "semantic_enabled": search["semantic_enabled"],
                "rerank_applied": search["rerank_applied"],
                "semantic_ms": search["semantic_ms"],
                "rerank_ms": search["rerank_ms"],
                "vector_index_backend": search["vector_index_backend"],
            },
            "meta": current_index().metadata(),
        }

    @application.post("/api/evaluate", tags=["rag"])
    async def evaluate(payload: EvaluationRequest):
        modes = list(dict.fromkeys(payload.modes))
        results = {mode: current_index().evaluate(mode) for mode in modes}
        return {
            "ok": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "case_count": current_index().metadata()["evaluation_case_count"],
            "results": results,
            "models": (
                semantic_runtime.status()
                if semantic_runtime is not None
                else {"enabled": False}
            ),
        }

    @application.post("/api/documents", status_code=201, tags=["documents"])
    async def create_document(payload: DocumentCreateRequest):
        document = build_uploaded_document(payload)
        current_repository().save_document(document, is_builtin=False)
        index_status = current_index().refresh_document(
            document, reason="document_created"
        )
        return {
            "ok": True,
            "document": document,
            "meta": index_status["meta"],
        }

    @application.delete("/api/documents/{document_id}", tags=["documents"])
    async def delete_document(document_id: str):
        existing = current_repository().get_document(document_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="制度文档不存在")
        if existing.get("is_builtin"):
            raise HTTPException(status_code=403, detail="内置制度文档不能通过 API 删除")
        deleted = current_repository().delete_uploaded_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="上传文档不存在")
        index_status = current_index().remove_document(
            document_id, reason="document_deleted"
        )
        return {
            "ok": True,
            "deleted_id": document_id,
            "meta": index_status["meta"],
        }

    @application.get("/api/index/status", tags=["index"])
    async def index_status():
        return {
            **current_index().status(),
            "database": current_repository().stats(),
        }

    @application.get("/api/models/status", tags=["index"])
    async def model_status():
        if semantic_runtime is None:
            return {"ok": True, "enabled": False, "models": None}
        return {
            "ok": True,
            "enabled": True,
            "models": semantic_runtime.status(),
            "meta": current_index().metadata(),
        }

    @application.get("/api/generation/status", tags=["generation"])
    async def generation_status():
        return {
            "ok": True,
            **generation_service.status(),
        }

    @application.post("/api/index/rebuild", tags=["index"])
    async def rebuild_index():
        seed = current_repository().sync_builtin_documents(policies["documents"])
        current_index().rebuild(
            current_repository().list_documents(),
            reason="manual_rebuild",
        )
        return {
            **current_index().status(),
            "seed": seed,
            "database": current_repository().stats(),
        }

    @application.get("/", include_in_schema=False)
    async def index_page():
        return FileResponse(STATIC_DIR / "index.html")

    application.mount(
        "/static", StaticFiles(directory=STATIC_DIR), name="static"
    )
    return application


app = create_app()
