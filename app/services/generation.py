from __future__ import annotations

import asyncio
import copy
import logging
import re
import threading
import time
from typing import Any

from pydantic import ValidationError

from app.services.index_manager import IndexManager
from app.services.llm import (
    LLMAnswerPayload,
    LLMProviderError,
    OpenAICompatibleProvider,
    ProviderCompletion,
    build_repair_prompts,
    build_rag_prompts,
    parse_llm_payload,
)
from app.services.retrieval import tokenize


NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")
logger = logging.getLogger("policylens.generation")


class GenerationValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "validation_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class RAGGenerationService:
    def __init__(
        self,
        index_manager: IndexManager,
        provider: OpenAICompatibleProvider | None,
        repair_retries: int = 1,
        evidence_limit: int = 4,
    ) -> None:
        self.index_manager = index_manager
        self.provider = provider
        self.repair_retries = max(0, repair_retries)
        self.evidence_limit = max(1, evidence_limit)
        self._lock = threading.Lock()
        self._metrics = {
            "total_requests": 0,
            "llm_answers": 0,
            "llm_refusals": 0,
            "fallbacks": 0,
            "precheck_refusals": 0,
            "invalid_citation_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            metrics = copy.deepcopy(self._metrics)
        provider_status = (
            self.provider.status()
            if self.provider is not None
            else {
                "provider": "none",
                "model": None,
                "enabled": False,
                "api_key_configured": False,
            }
        )
        return {
            "provider": provider_status,
            "metrics": metrics,
            "fallback_mode": "extractive",
            "optimization": {
                "repair_retries": self.repair_retries,
                "evidence_limit": self.evidence_limit,
            },
        }

    async def generate(
        self,
        question: str,
        *,
        scope: str,
        department: str,
        mode: str,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        answer_result = await asyncio.to_thread(
            self.index_manager.answer_question,
            question,
            scope=scope,
            department=department,
            mode=mode,
            limit=6,
        )
        baseline_answer = answer_result["answer"]
        search_response = answer_result["search"]
        evidence = search_response["results"][: self.evidence_limit]
        generation = self._base_generation()
        generation["retrieval_ms"] = search_response["elapsed_ms"]

        if self.provider is None or not self.provider.enabled:
            return self._finalize(
                answer_result,
                generation,
                mode="extractive",
                started_at=started_at,
                baseline_answer=baseline_answer,
                trace_id=trace_id,
            )

        if baseline_answer["refused"]:
            self._increment("total_requests")
            self._increment("precheck_refusals")
            generation["mode"] = "precheck_refusal"
            generation["fallback_reason"] = baseline_answer["note"]
            return self._finalize(
                answer_result,
                generation,
                mode="precheck_refusal",
                started_at=started_at,
                baseline_answer=baseline_answer,
                trace_id=trace_id,
            )

        system_prompt, user_prompt = build_rag_prompts(question, evidence)
        self._increment("total_requests")
        completion: ProviderCompletion | None = None
        last_error: Exception | None = None
        last_output = ""

        for repair_attempt in range(self.repair_retries + 1):
            try:
                completion = await asyncio.to_thread(
                    self.provider.complete,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except Exception as error:
                last_error = error
                break

            # Usage is billed even if the returned content fails validation.
            self._record_completion(completion)
            self._accumulate_completion(generation, completion)
            last_output = completion.content
            generation["provider_call_count"] = repair_attempt + 1

            try:
                payload = parse_llm_payload(completion.content)
                validated = self._validate_payload(payload, evidence)
            except Exception as error:
                last_error = error
                generation["citation_validation"] = self._validation_details(
                    error, evidence
                )
                generation["repair_attempts"] = repair_attempt
                generation["llm_output_preview"] = completion.content[:4000]
                if repair_attempt < self.repair_retries:
                    system_prompt, user_prompt = build_repair_prompts(
                        question,
                        evidence,
                        completion.content,
                        self._describe_error(error),
                    )
                    continue
                break

            generation["citation_validation"] = {
                "status": "passed",
                "claim_count": len(validated["claims"]),
                "citation_ids": validated.get("all_citations", []),
            }
            generation["repair_attempts"] = repair_attempt
            if payload.refused:
                answer = self._build_refusal_answer(
                    payload,
                    evidence,
                    baseline_answer,
                    completion,
                )
                self._increment("llm_refusals")
                generation["mode"] = "llm_refusal"
                generation["citation_count"] = 0
                answer["generation"] = generation
                answer["trace"]["generation"] = generation
                self._log_generation(generation, trace_id)
                return {
                    "search": search_response,
                    "answer": answer,
                    "generation": {
                        **generation,
                        "elapsed_ms": (time.perf_counter() - started_at) * 1000,
                    },
                }

            answer = self._build_generated_answer(
                payload,
                validated,
                evidence,
                baseline_answer,
                completion,
            )
            self._increment("llm_answers")
            generation["mode"] = "llm"
            generation["citation_count"] = sum(
                len(claim["citations"]) for claim in validated["claims"]
            )
            answer["generation"] = generation
            answer["trace"]["generation"] = generation
            self._log_generation(generation, trace_id)
            return {
                "search": search_response,
                "answer": answer,
                "generation": {
                    **generation,
                    "elapsed_ms": (time.perf_counter() - started_at) * 1000,
                },
            }

        self._increment("fallbacks")
        if isinstance(last_error, GenerationValidationError):
            self._increment("invalid_citation_count")
        generation["mode"] = "fallback"
        generation["fallback_reason"] = self._describe_error(
            last_error or RuntimeError("LLM 未返回可用结果")
        )
        if last_output:
            generation["llm_output_preview"] = last_output[:4000]
        if isinstance(last_error, LLMProviderError):
            generation["provider_status_code"] = last_error.status_code
            generation["provider_request_id"] = last_error.request_id
        return self._finalize(
            answer_result,
            generation,
            mode="fallback",
            started_at=started_at,
            baseline_answer=baseline_answer,
            trace_id=trace_id,
        )

    def _base_generation(self) -> dict[str, Any]:
        provider_status = (
            self.provider.status()
            if self.provider is not None
            else {"provider": "none", "model": None, "enabled": False}
        )
        return {
            "provider": provider_status.get("provider"),
            "model": provider_status.get("model"),
            "enabled": provider_status.get("enabled", False),
            "mode": "extractive",
            "fallback_used": False,
            "fallback_reason": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "cost_currency": provider_status.get("cost_currency", "CNY"),
            "provider_latency_ms": 0.0,
            "attempts": 0,
            "request_id": None,
            "citation_count": 0,
            "provider_call_count": 0,
            "repair_attempts": 0,
            "citation_validation": None,
            "llm_output_preview": None,
        }

    def _finalize(
        self,
        answer_result: dict[str, Any],
        generation: dict[str, Any],
        *,
        mode: str,
        started_at: float,
        baseline_answer: dict[str, Any],
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        answer = copy.deepcopy(baseline_answer)
        generation["mode"] = mode
        generation["fallback_used"] = mode in {"fallback", "extractive"}
        generation["elapsed_ms"] = (time.perf_counter() - started_at) * 1000
        answer["generation"] = generation
        answer["trace"]["generation"] = generation
        self._log_generation(generation, trace_id)
        return {
            "search": answer_result["search"],
            "answer": answer,
            "generation": generation,
        }

    def _validate_payload(
        self,
        payload: LLMAnswerPayload,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if payload.refused:
            return {"claims": []}
        evidence_count = len(evidence)
        validated_claims = []
        for claim in payload.claims:
            invalid = [
                citation
                for citation in claim.citations
                if citation < 1 or citation > evidence_count
            ]
            if invalid:
                raise GenerationValidationError(
                    f"LLM 引用了不存在的证据编号: {invalid}",
                    code="citation_out_of_range",
                    details={
                        "invalid_citations": invalid,
                        "evidence_count": evidence_count,
                        "claim": claim.text,
                    },
                )
            cited_text = "\n".join(
                evidence[citation - 1]["section"]["text"]
                for citation in claim.citations
            )
            support = self._support_details(claim.text, cited_text)
            if not support["supported"]:
                raise GenerationValidationError(
                    f"结论缺少可验证的原文支持: {claim.text[:80]}",
                    code="claim_not_supported",
                    details={
                        "claim": claim.text,
                        "citations": claim.citations,
                        **support,
                    },
                )
            validated_claims.append(
                {
                    "text": claim.text,
                    "citations": claim.citations,
                }
            )

        all_cited_text = "\n".join(
            evidence[citation - 1]["section"]["text"]
            for citation in {
                citation
                for claim in validated_claims
                for citation in claim["citations"]
            }
        )
        summary_support = self._support_details(
            payload.answer,
            all_cited_text,
            threshold=0.06,
        )
        if not summary_support["supported"]:
            raise GenerationValidationError(
                "LLM 总结包含引用条款之外的内容",
                code="summary_not_supported",
                details=summary_support,
            )
        return {
            "claims": validated_claims,
            "all_citations": sorted(
                {
                    citation
                    for claim in validated_claims
                    for citation in claim["citations"]
                }
            ),
        }

    @staticmethod
    def _support_details(
        claim: str,
        evidence_text: str,
        *,
        threshold: float = 0.08,
    ) -> dict[str, Any]:
        claim_tokens = set(tokenize(claim))
        evidence_tokens = set(tokenize(evidence_text))
        if not claim_tokens or not evidence_tokens:
            return {
                "supported": False,
                "overlap": 0.0,
                "threshold": threshold,
                "missing_numbers": sorted(NUMBER_PATTERN.findall(claim)),
            }
        overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
        claim_numbers = set(NUMBER_PATTERN.findall(claim))
        evidence_numbers = set(NUMBER_PATTERN.findall(evidence_text))
        missing_numbers = sorted(claim_numbers - evidence_numbers)
        supported = overlap >= threshold and not missing_numbers
        return {
            "supported": supported,
            "overlap": round(overlap, 4),
            "threshold": threshold,
            "missing_numbers": missing_numbers,
        }

    def _build_generated_answer(
        self,
        payload: LLMAnswerPayload,
        validated: dict[str, Any],
        evidence: list[dict[str, Any]],
        baseline_answer: dict[str, Any],
        completion: ProviderCompletion,
    ) -> dict[str, Any]:
        trace = copy.deepcopy(baseline_answer["trace"])
        points = []
        for index, claim in enumerate(validated["claims"], 1):
            primary_citation = claim["citations"][0]
            result = evidence[primary_citation - 1]
            document = result["document"]
            section = result["section"]
            points.append(
                {
                    "text": claim["text"],
                    "doc_id": document["id"],
                    "title": document["title"],
                    "version": document["version"],
                    "section": section["heading"],
                    "page": section["page"],
                    "status": document["status"],
                    "citation": primary_citation,
                    "citations": claim["citations"],
                }
            )
        return {
            "refused": False,
            "confidence": payload.confidence,
            "title": "基于制度证据的生成式回答",
            "lead": payload.answer.strip(),
            "points": points,
            "note": (
                f"回答由 {completion.model} 基于检索条款组织，"
                "全部结论已通过引用存在性检查。"
            ),
            "evidence": evidence,
            "trace": trace,
        }

    def _build_refusal_answer(
        self,
        payload: LLMAnswerPayload,
        evidence: list[dict[str, Any]],
        baseline_answer: dict[str, Any],
        completion: ProviderCompletion,
    ) -> dict[str, Any]:
        reason = payload.refusal_reason or payload.answer or "证据不足，模型拒绝回答。"
        return {
            "refused": True,
            "confidence": "insufficient",
            "title": "模型判断当前证据不足",
            "lead": "未生成确定性结论。",
            "points": [],
            "note": reason,
            "evidence": evidence[:2],
            "trace": copy.deepcopy(baseline_answer["trace"]),
            "model": completion.model,
        }

    @staticmethod
    def _completion_generation(
        completion: ProviderCompletion,
    ) -> dict[str, Any]:
        return {
            "provider_latency_ms": completion.latency_ms,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "total_tokens": completion.total_tokens,
            "estimated_cost": round(completion.estimated_cost, 8),
            "cost_currency": completion.cost_currency,
            "provider_model": completion.model,
            "request_id": completion.request_id,
            "attempts": completion.attempts,
            "model": completion.model,
        }

    def _accumulate_completion(
        self,
        generation: dict[str, Any],
        completion: ProviderCompletion,
    ) -> None:
        generation["provider_latency_ms"] = round(
            generation.get("provider_latency_ms", 0.0) + completion.latency_ms,
            2,
        )
        generation["prompt_tokens"] = (
            generation.get("prompt_tokens", 0) + completion.prompt_tokens
        )
        generation["completion_tokens"] = (
            generation.get("completion_tokens", 0)
            + completion.completion_tokens
        )
        generation["total_tokens"] = (
            generation.get("total_tokens", 0) + completion.total_tokens
        )
        generation["estimated_cost"] = round(
            generation.get("estimated_cost", 0.0)
            + completion.estimated_cost,
            8,
        )
        generation["cost_currency"] = completion.cost_currency
        generation["provider_model"] = completion.model
        generation["model"] = completion.model
        generation["request_id"] = completion.request_id
        generation["attempts"] = (
            generation.get("attempts", 0) + completion.attempts
        )

    @staticmethod
    def _validation_details(
        error: Exception,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if isinstance(error, GenerationValidationError):
            return {
                "status": "failed",
                "code": error.code,
                "message": str(error),
                "evidence_count": len(evidence),
                "details": error.details,
            }
        if isinstance(error, ValidationError):
            errors = []
            for item in error.errors()[:8]:
                errors.append(
                    {
                        "location": ".".join(
                            str(part) for part in item.get("loc", [])
                        ),
                        "message": item.get("msg", ""),
                        "type": item.get("type", ""),
                    }
                )
            return {
                "status": "failed",
                "code": "schema_validation_error",
                "message": "LLM JSON 字段结构不符合要求",
                "evidence_count": len(evidence),
                "details": errors,
            }
        return {
            "status": "failed",
            "code": "json_parse_error",
            "message": str(error),
            "evidence_count": len(evidence),
            "details": {},
        }

    def _record_completion(self, completion: ProviderCompletion) -> None:
        with self._lock:
            self._metrics["prompt_tokens"] += completion.prompt_tokens
            self._metrics["completion_tokens"] += completion.completion_tokens
            self._metrics["total_tokens"] += completion.total_tokens
            self._metrics["estimated_cost"] += completion.estimated_cost

    def _increment(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._metrics[key] += amount

    @staticmethod
    def _log_generation(
        generation: dict[str, Any],
        trace_id: str | None,
    ) -> None:
        logger.info(
            "LLM generation completed: mode=%s model=%s tokens=%s cost=%s",
            generation.get("mode"),
            generation.get("model"),
            generation.get("total_tokens", 0),
            generation.get("estimated_cost", 0),
            extra={
                "request_id": trace_id or generation.get("request_id") or "-",
                "generation_mode": generation.get("mode"),
                "model": generation.get("model"),
                "prompt_tokens": generation.get("prompt_tokens", 0),
                "completion_tokens": generation.get("completion_tokens", 0),
                "total_tokens": generation.get("total_tokens", 0),
                "estimated_cost": generation.get("estimated_cost", 0),
            },
        )

    @staticmethod
    def _describe_error(error: Exception) -> str:
        if isinstance(error, LLMProviderError):
            status = f"HTTP {error.status_code}: " if error.status_code else ""
            return f"{status}{error}"
        return str(error) or error.__class__.__name__
