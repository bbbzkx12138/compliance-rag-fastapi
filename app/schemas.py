from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


SearchScope = Literal["current", "all"]
RetrievalMode = Literal["keyword", "vector", "hybrid", "rerank"]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    scope: SearchScope = "current"
    department: str = Field(default="all", max_length=80)
    mode: RetrievalMode = "rerank"

    @field_validator("question", "department")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class EvaluationRequest(BaseModel):
    modes: list[RetrievalMode] = Field(
        default_factory=lambda: ["keyword", "vector", "hybrid", "rerank"]
    )
    save_report: bool = True


class DocumentCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    version: str = Field(default="V1.0", min_length=1, max_length=24)
    department: str = Field(default="合规部", min_length=1, max_length=80)
    effective_date: date
    content: str = Field(min_length=1, max_length=300_000)

    @field_validator("title", "version", "department", "content")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("字段不能只包含空白字符")
        return value


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=80)
    helpful: bool
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)

    @field_validator("comment")
    @classmethod
    def strip_comment(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None
