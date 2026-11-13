from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class LLMProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.request_id = request_id


class LLMClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1200)
    citations: list[int] = Field(min_length=1, max_length=6)


class LLMAnswerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refused: bool
    answer: str = Field(max_length=4000)
    claims: list[LLMClaim] = Field(default_factory=list, max_length=8)
    confidence: Literal["high", "medium", "low"]
    refusal_reason: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_payload(self):
        if not self.refused and not self.claims:
            raise ValueError("非拒答回答必须包含至少一个带引用的结论")
        if not self.refused and not self.answer.strip():
            raise ValueError("非拒答回答不能为空")
        if self.refused and not (self.refusal_reason or self.answer).strip():
            raise ValueError("拒答必须提供原因")
        return self


@dataclass(frozen=True)
class ProviderCompletion:
    content: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    cost_currency: str
    request_id: str | None
    attempts: int


SYSTEM_PROMPT = """你是企业合规制度问答助手。必须严格遵守以下规则：
1. 只能使用用户消息中提供的制度证据回答，不得使用模型常识、外部知识或猜测。
2. 每个关键结论必须引用至少一个真实证据编号。
3. 只能输出一个 JSON 对象，不得输出 Markdown、代码围栏或额外说明。
4. JSON 必须符合：
{
  "refused": boolean,
  "answer": string,
  "claims": [{"text": string, "citations": [integer]}],
  "confidence": "high" | "medium" | "low",
  "refusal_reason": string | null
}
5. 如果证据不足、相互冲突或不能直接支持问题，必须 refused=true，claims=[]，并说明 refusal_reason。
6. 证据标记 [E1] 对应 citations 中的整数 1，[E2] 对应整数 2，以此类推；禁止返回字符串、文档 ID 或零基编号。
7. 涉及金额、时限、比例和日期时，必须逐字保留证据中的数字和单位。
8. 每条 claim 只能陈述其引用证据直接支持的内容；跨条款结论必须同时引用所有相关条款。
9. 回答必须简洁：claims 最多 3 条，每条不超过 120 个汉字，禁止重复证据全文。
"""


def build_rag_prompts(
    question: str,
    evidence: list[dict[str, Any]],
) -> tuple[str, str]:
    evidence_blocks = []
    for index, item in enumerate(evidence, 1):
        document = item["document"]
        section = item["section"]
        evidence_blocks.append(
            "\n".join(
                [
                    f"[E{index}]",
                    f"制度：{document['title']} {document['version']}",
                    f"状态：{document['status']}",
                    f"部门：{document['department']}",
                    f"条款：{section['heading']}",
                    f"原文：{section['text']}",
                ]
            )
        )
    user_prompt = (
        f"用户问题：{question}\n\n"
        "可用制度证据：\n\n"
        + "\n\n".join(evidence_blocks)
        + "\n\n请严格返回指定 JSON。"
    )
    return SYSTEM_PROMPT, user_prompt


def build_repair_prompts(
    question: str,
    evidence: list[dict[str, Any]],
    previous_output: str,
    validation_error: str,
) -> tuple[str, str]:
    system_prompt, user_prompt = build_rag_prompts(question, evidence)
    repair_instruction = (
        "\n\n上一次回答未通过机器校验。请修正后重新输出完整 JSON：\n"
        f"校验错误：{validation_error}\n"
        "只能引用证据中存在的整数编号；每条 claim 必须带 citations，"
        "且不得加入引用原文无法直接支持的内容。\n"
        "上一次输出：\n"
        f"{previous_output[:6000]}"
    )
    return system_prompt, user_prompt + repair_instruction


def normalize_citation_values(parsed: dict[str, Any]) -> dict[str, Any]:
    claims = parsed.get("claims")
    if not isinstance(claims, list):
        return parsed
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        citations = claim.get("citations")
        if not isinstance(citations, list):
            continue
        normalized = []
        for citation in citations:
            if isinstance(citation, int):
                value = citation
            elif isinstance(citation, str) and re.fullmatch(
                r"[Ee]?\d+", citation.strip()
            ):
                value = int(citation.strip().lstrip("Ee"))
            else:
                normalized.append(citation)
                continue
            if value == 0:
                value = 1
            if value not in normalized:
                normalized.append(value)
        claim["citations"] = normalized
    return parsed


def parse_llm_payload(content: str) -> LLMAnswerPayload:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM 输出中没有 JSON 对象")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError(f"LLM JSON 解析失败: {error.msg}") from error
    return LLMAnswerPayload.model_validate(normalize_citation_values(parsed))


class OpenAICompatibleProvider:
    """OpenAI, Qwen DashScope, Ollama, or any compatible chat endpoint."""

    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: int = 30,
        max_retries: int = 2,
        temperature: float = 0,
        max_tokens: int = 1200,
        enable_thinking: bool = False,
        json_mode: bool = True,
        input_cost_per_million: float = 0,
        output_cost_per_million: float = 0,
        cost_currency: str = "CNY",
    ) -> None:
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.json_mode = json_mode
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.cost_currency = cost_currency
        self._lock = threading.Lock()
        self._success_count = 0
        self._failure_count = 0
        self._consecutive_failures = 0
        self._last_error: str | None = None
        self._disabled_until = 0.0
        self._last_request_id: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model) and (
            self.provider_name == "local" or bool(self._api_key)
        )

    def status(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model,
            "base_url": self.base_url,
            "enabled": self.enabled,
            "api_key_configured": bool(self._api_key),
            "json_mode": self.json_mode,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_tokens": self.max_tokens,
            "enable_thinking": self.enable_thinking,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "consecutive_failures": self._consecutive_failures,
            "last_error": self._last_error,
            "last_request_id": self._last_request_id,
            "circuit_open": time.monotonic() < self._disabled_until,
            "cost_currency": self.cost_currency,
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
        }

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ProviderCompletion:
        if not self.enabled:
            raise LLMProviderError("LLM Provider 未配置完整或缺少 API Key")
        if time.monotonic() < self._disabled_until:
            raise LLMProviderError("LLM Provider 暂时熔断，正在使用抽取式降级")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.provider_name == "qwen":
            # Qwen3 thinking mode can add large latency for structured extraction.
            payload["enable_thinking"] = self.enable_thinking

        started_at = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    **(
                        {"Authorization": f"Bearer {self._api_key}"}
                        if self._api_key
                        else {}
                    ),
                },
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    raw = response.read().decode("utf-8")
                    body = json.loads(raw)
                    request_id = (
                        response.headers.get("x-request-id")
                        or body.get("id")
                    )
                    content = body["choices"][0]["message"]["content"]
                    usage = body.get("usage") or {}
                    prompt_tokens = int(
                        usage.get("prompt_tokens")
                        or usage.get("input_tokens")
                        or 0
                    )
                    completion_tokens = int(
                        usage.get("completion_tokens")
                        or usage.get("output_tokens")
                        or 0
                    )
                    total_tokens = int(
                        usage.get("total_tokens")
                        or prompt_tokens + completion_tokens
                    )
                    self._record_success(request_id)
                    return ProviderCompletion(
                        content=content,
                        model=body.get("model", self.model),
                        latency_ms=(time.perf_counter() - started_at) * 1000,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        estimated_cost=(
                            prompt_tokens / 1_000_000 * self.input_cost_per_million
                            + completion_tokens
                            / 1_000_000
                            * self.output_cost_per_million
                        ),
                        cost_currency=self.cost_currency,
                        request_id=request_id,
                        attempts=attempt,
                    )
            except urllib.error.HTTPError as error:
                raw_error = error.read().decode("utf-8", errors="replace")
                try:
                    error_payload = json.loads(raw_error)
                except json.JSONDecodeError:
                    error_payload = {}
                message = (
                    error_payload.get("error", {}).get("message")
                    or raw_error[:300]
                    or str(error)
                )
                request_id = error.headers.get("x-request-id")
                wrapped = LLMProviderError(
                    message,
                    status_code=error.code,
                    retryable=error.code in RETRYABLE_STATUS_CODES,
                    request_id=request_id,
                )
                last_error = wrapped
                if (
                    error.code == 400
                    and "enable_thinking" in payload
                    and "enable_thinking" in message.lower()
                ):
                    payload.pop("enable_thinking", None)
                    if attempt <= self.max_retries:
                        continue
                if not wrapped.retryable:
                    self._record_failure(wrapped)
                    raise wrapped
            except (urllib.error.URLError, socket.timeout, TimeoutError) as error:
                last_error = LLMProviderError(
                    f"LLM 网络请求失败: {error}",
                    retryable=True,
                )

            if attempt <= self.max_retries:
                time.sleep(min(4.0, 0.5 * (2 ** (attempt - 1))))

        if last_error is None:
            last_error = LLMProviderError("LLM Provider 请求失败")
        self._record_failure(last_error)
        raise last_error

    def _record_success(self, request_id: str | None) -> None:
        with self._lock:
            self._success_count += 1
            self._consecutive_failures = 0
            self._last_error = None
            self._last_request_id = request_id

    def _record_failure(self, error: Exception) -> None:
        with self._lock:
            self._failure_count += 1
            self._consecutive_failures += 1
            self._last_error = str(error)
            if isinstance(error, LLMProviderError):
                self._last_request_id = error.request_id
            if self._consecutive_failures >= 3:
                self._disabled_until = time.monotonic() + 30


def create_llm_provider() -> OpenAICompatibleProvider | None:
    provider_name = os.environ.get("POLICY_RAG_LLM_PROVIDER", "none").lower()
    if provider_name == "none":
        return None

    defaults = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4.1-mini",
            "key_env": "OPENAI_API_KEY",
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-plus",
            "key_env": "DASHSCOPE_API_KEY",
        },
        "local": {
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "qwen2.5:7b",
            "key_env": "",
        },
    }
    if provider_name not in defaults:
        raise ValueError(f"不支持的 LLM Provider: {provider_name}")
    config = defaults[provider_name]
    api_key = (
        os.environ.get("POLICY_RAG_LLM_API_KEY", "").strip()
        or (
            os.environ.get(config["key_env"], "").strip()
            if config["key_env"]
            else ""
        )
    )
    return OpenAICompatibleProvider(
        provider_name=provider_name,
        base_url=os.environ.get(
            "POLICY_RAG_LLM_BASE_URL", config["base_url"]
        ).strip(),
        model=os.environ.get("POLICY_RAG_LLM_MODEL", config["model"]).strip(),
        api_key=api_key,
        timeout_seconds=int(
            os.environ.get("POLICY_RAG_LLM_TIMEOUT_SECONDS", "30")
        ),
        max_retries=int(os.environ.get("POLICY_RAG_LLM_MAX_RETRIES", "2")),
        temperature=float(os.environ.get("POLICY_RAG_LLM_TEMPERATURE", "0")),
        max_tokens=int(os.environ.get("POLICY_RAG_LLM_MAX_TOKENS", "700")),
        enable_thinking=os.environ.get(
            "POLICY_RAG_LLM_ENABLE_THINKING", "false"
        ).lower()
        in {"1", "true", "yes", "on"},
        json_mode=os.environ.get("POLICY_RAG_LLM_JSON_MODE", "true").lower()
        in {"1", "true", "yes", "on"},
        input_cost_per_million=float(
            os.environ.get("POLICY_RAG_LLM_INPUT_COST_PER_MILLION", "0")
        ),
        output_cost_per_million=float(
            os.environ.get("POLICY_RAG_LLM_OUTPUT_COST_PER_MILLION", "0")
        ),
        cost_currency=os.environ.get("POLICY_RAG_LLM_COST_CURRENCY", "CNY"),
    )
