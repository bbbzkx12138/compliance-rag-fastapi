from __future__ import annotations

import copy
import math
import re
import time
from collections import Counter
from typing import Any, Iterable


STOP_TOKENS = {
    "公司",
    "员工",
    "可以",
    "能否",
    "是否",
    "需要",
    "什么",
    "哪些",
    "多少",
    "怎么",
    "如何",
    "多久",
    "一下",
    "请问",
    "相关",
    "规定",
    "制度",
    "要求",
    "我们",
    "这个",
    "那个",
    "进行",
    "应该",
    "必须",
    "吗",
    "呢",
    "的",
    "了",
    "和",
    "与",
    "或",
    "及",
    "在",
    "对",
    "向",
    "从",
    "把",
    "被",
    "有",
    "是",
}

QUERY_FILLERS = (
    "请问",
    "帮我查",
    "想知道",
    "是否可以",
    "能不能",
    "可不可以",
    "有没有",
    "根据公司",
    "按照制度",
    "相关要求",
)

STATUS_LABELS = {
    "effective": "现行有效",
    "expired": "已失效",
    "draft": "草稿",
}

PUNCTUATION_PATTERN = re.compile(r"[，。！？、；：（）【】《》,.!?;:()\[\]<>]")
HAN_PATTERN = re.compile(r"[\u3400-\u9fff]+")
LATIN_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")
SENTENCE_PATTERN = re.compile(r"[^。！？；]+[。！？；]?")
NUMERIC_QUERY_PATTERN = re.compile(r"(多少|上限|标准|金额|时限|多久|比例|变化|差异)")


def normalize(text: str) -> str:
    value = str(text or "").lower()
    value = value.replace("“", "").replace("”", "").replace('"', "")
    value = value.replace("'", "").replace("`", "")
    value = PUNCTUATION_PATTERN.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize(text)
    tokens = LATIN_PATTERN.findall(normalized)
    for sequence in HAN_PATTERN.findall(normalized):
        if len(sequence) == 1:
            tokens.append(sequence)
            continue
        for index in range(len(sequence) - 1):
            tokens.append(sequence[index : index + 2])
        for index in range(len(sequence) - 2):
            tokens.append(sequence[index : index + 3])
    return [token for token in tokens if token and token not in STOP_TOKENS]


def unique_tokens(text: str) -> list[str]:
    return list(dict.fromkeys(tokenize(text)))


def _build_synonym_map(source: dict[str, list[str]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for key, values in source.items():
        mapping.setdefault(key, [])
        mapping[key].extend(value for value in values if value not in mapping[key])
        for value in values:
            mapping.setdefault(value, [])
            if key not in mapping[value]:
                mapping[value].append(key)
    return mapping


def _vector_norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _overlap_ratio(needles: Iterable[str], haystack: set[str]) -> float:
    needle_list = list(needles)
    if not needle_list:
        return 0.0
    return sum(1 for token in needle_list if token in haystack) / len(needle_list)


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.findall(text or "")
        if sentence.strip()
    ]


class RAGEngine:
    """In-memory hybrid retrieval, extractive answers, and evaluation."""

    def __init__(
        self,
        policies: dict[str, Any],
        uploaded_documents: list[dict[str, Any]] | None = None,
    ) -> None:
        self.base_documents = copy.deepcopy(policies["documents"])
        self.uploaded_documents = copy.deepcopy(uploaded_documents or [])
        self.synonyms = policies.get("synonyms", {})
        self.synonym_map = _build_synonym_map(self.synonyms)
        self.evaluation_cases = policies.get("evaluation_cases", [])
        self.corpus_meta = policies.get("corpus_meta", {})
        self.documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.vocabulary: set[str] = set()
        self.idf: dict[str, float] = {}
        self.average_length = 1.0
        self._build_index()

    def set_uploaded_documents(self, documents: list[dict[str, Any]]) -> None:
        self.uploaded_documents = copy.deepcopy(documents)
        self._build_index()

    def _build_index(self) -> None:
        self.documents = [*copy.deepcopy(self.base_documents), *self.uploaded_documents]
        chunks: list[dict[str, Any]] = []
        document_frequency: Counter[str] = Counter()

        for document in self.documents:
            for section_index, section in enumerate(document["sections"]):
                searchable_text = " ".join(
                    [
                        document["title"],
                        document.get("short_title", ""),
                        document.get("category", ""),
                        " ".join(document.get("tags", [])),
                        section["heading"],
                        section["text"],
                    ]
                )
                tokens = tokenize(searchable_text)
                term_frequencies = Counter(tokens)
                chunks.append(
                    {
                        "id": f"{document['id']}-{section_index + 1}",
                        "document": document,
                        "section": section,
                        "section_index": section_index,
                        "searchable_text": searchable_text,
                        "tokens": tokens,
                        "term_frequencies": term_frequencies,
                        "heading_tokens": set(unique_tokens(section["heading"])),
                        "title_tokens": set(
                            unique_tokens(
                                " ".join(
                                    [
                                        document["title"],
                                        document.get("short_title", ""),
                                        " ".join(document.get("tags", [])),
                                    ]
                                )
                            )
                        ),
                    }
                )
                document_frequency.update(set(tokens))

        self.chunks = chunks
        self.vocabulary = set(document_frequency)
        total_length = sum(len(chunk["tokens"]) for chunk in chunks)
        self.average_length = total_length / max(len(chunks), 1)
        self.idf = {
            token: math.log(
                1 + (len(chunks) - frequency + 0.5) / (frequency + 0.5)
            )
            for token, frequency in document_frequency.items()
        }

        for chunk in chunks:
            vector: dict[str, float] = {}
            for token, frequency in chunk["term_frequencies"].items():
                term_frequency = frequency / max(len(chunk["tokens"]), 1)
                vector[token] = term_frequency * self.idf.get(token, 0.0)
            chunk["vector"] = vector
            chunk["vector_norm"] = _vector_norm(vector)

    def _expand_query(self, query: str) -> dict[str, Any]:
        cleaned = query
        for filler in QUERY_FILLERS:
            cleaned = cleaned.replace(filler, " ")
        additions: list[str] = []
        for term, expansions in self.synonym_map.items():
            if term in cleaned:
                for expansion in expansions:
                    if expansion not in additions:
                        additions.append(expansion)
        original_tokens = unique_tokens(cleaned)
        expanded_tokens = unique_tokens(" ".join([cleaned, *additions]))
        return {
            "cleaned": cleaned.strip(),
            "original_tokens": original_tokens,
            "tokens": expanded_tokens,
            "additions": additions,
        }

    def _bm25(
        self,
        chunk: dict[str, Any],
        query_tokens: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        score = 0.0
        length_ratio = len(chunk["tokens"]) / max(self.average_length, 1.0)
        for token in query_tokens:
            frequency = chunk["term_frequencies"].get(token, 0)
            if not frequency:
                continue
            denominator = frequency + k1 * (1 - b + b * length_ratio)
            score += self.idf.get(token, 0.0) * (
                frequency * (k1 + 1) / max(denominator, 0.001)
            )
        return score

    def _query_vector(self, tokens: list[str]) -> tuple[dict[str, float], float]:
        frequencies = Counter(tokens)
        vector = {
            token: (frequency / max(len(tokens), 1)) * self.idf.get(token, 0.0)
            for token, frequency in frequencies.items()
        }
        return vector, _vector_norm(vector)

    @staticmethod
    def _cosine(
        left: dict[str, float],
        left_norm: float,
        right: dict[str, float],
        right_norm: float,
    ) -> float:
        if not left_norm or not right_norm:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
        dot_product = sum(
            value * right.get(token, 0.0) for token, value in left.items()
        )
        return dot_product / (left_norm * right_norm)

    def search(
        self,
        query: str,
        *,
        scope: str = "current",
        department: str = "all",
        mode: str = "hybrid",
        limit: int = 6,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        expanded = self._expand_query(query)
        query_tokens = (
            expanded["tokens"] if mode == "hybrid" else expanded["original_tokens"]
        )
        unique_query_tokens = set(query_tokens)
        query_vector, query_norm = self._query_vector(query_tokens)
        candidates = [
            chunk
            for chunk in self.chunks
            if (scope != "current" or chunk["document"]["status"] == "effective")
            and (scope != "all" or chunk["document"]["status"] != "draft")
            and (
                department == "all"
                or chunk["document"]["department"] == department
            )
        ]

        raw_results: list[dict[str, Any]] = []
        for chunk in candidates:
            bm25 = self._bm25(chunk, query_tokens)
            cosine = self._cosine(
                query_vector,
                query_norm,
                chunk["vector"],
                chunk["vector_norm"],
            )
            coverage = _overlap_ratio(unique_query_tokens, set(chunk["tokens"]))
            heading_coverage = _overlap_ratio(
                expanded["original_tokens"], chunk["heading_tokens"]
            )
            title_coverage = _overlap_ratio(
                expanded["original_tokens"], chunk["title_tokens"]
            )
            phrase = normalize(expanded["cleaned"]) in normalize(
                chunk["searchable_text"]
            )
            raw_results.append(
                {
                    "document": chunk["document"],
                    "section": chunk["section"],
                    "bm25": bm25,
                    "cosine": cosine,
                    "coverage": coverage,
                    "heading_coverage": heading_coverage,
                    "title_coverage": title_coverage,
                    "phrase": phrase,
                }
            )

        max_bm25 = max((result["bm25"] for result in raw_results), default=0.001)
        for result in raw_results:
            bm25_normalized = result["bm25"] / max(max_bm25, 0.001)
            if mode == "keyword":
                base = bm25_normalized
            elif mode == "vector":
                base = result["cosine"]
            else:
                base = bm25_normalized * 0.56 + result["cosine"] * 0.34
            field_boost = (
                result["heading_coverage"] * 0.09
                + result["title_coverage"] * 0.05
                + (0.08 if result["phrase"] else 0.0)
                if mode == "hybrid"
                else 0.0
            )
            coverage_factor = 0.58 + result["coverage"] * 0.42
            status = result["document"]["status"]
            authority_boost = 0.025 if status == "effective" else -0.045
            result["score"] = (
                base * coverage_factor + field_boost + authority_boost
            )
            result["scores"] = {
                "bm25": result["bm25"],
                "bm25_normalized": bm25_normalized,
                "cosine": result["cosine"],
                "coverage": result["coverage"],
                "field_boost": field_boost,
                "final": result["score"],
            }

        results = sorted(
            (
                result
                for result in raw_results
                if result["bm25"] > 0 or result["cosine"] > 0
            ),
            key=lambda item: item["score"],
            reverse=True,
        )[:limit]

        original_tokens = expanded["original_tokens"]
        known_query_terms = [
            token
            for token in query_tokens
            if len(token) > 1 and token in self.vocabulary
        ]
        covered_characters = {
            character for term in known_query_terms for character in term
        }
        unseen_terms = [
            token
            for token in original_tokens
            if token not in self.vocabulary
            and token not in STOP_TOKENS
            and len(token) > 1
            and all(character not in covered_characters for character in token)
        ]
        unknown_ratio = len(unseen_terms) / max(len(original_tokens), 1)

        return {
            "query": expanded["cleaned"],
            "mode": mode,
            "scope": scope,
            "department": department,
            "expanded_terms": expanded["additions"],
            "query_tokens": original_tokens,
            "unseen_terms": unseen_terms,
            "unknown_ratio": unknown_ratio,
            "results": results,
            "elapsed_ms": (time.perf_counter() - started_at) * 1000,
            "thresholds": {
                "minimum_score": 0.26,
                "minimum_coverage": 0.2,
                "maximum_unknown_ratio": 0.34,
            },
        }

    def _assess_evidence(self, search_response: dict[str, Any]) -> dict[str, Any]:
        results = search_response["results"]
        if not results:
            return {
                "refuse": True,
                "reason": "没有检索到与问题相关且满足权限范围的制度条款。",
                "level": "none",
            }
        top = results[0]
        thresholds = search_response["thresholds"]
        if (
            search_response["unknown_ratio"] > thresholds["maximum_unknown_ratio"]
            and top["coverage"] < 0.42
        ):
            terms = "、".join(search_response["unseen_terms"][:3])
            return {
                "refuse": True,
                "reason": f"问题中的“{terms}”未在知识库中找到可靠依据。",
                "level": "none",
            }
        if (
            top["score"] < thresholds["minimum_score"]
            or top["coverage"] < thresholds["minimum_coverage"]
        ):
            return {
                "refuse": True,
                "reason": "现有条款与问题的匹配程度不足，继续回答可能产生误导。",
                "level": "low",
            }
        if top["score"] >= 0.55 and top["coverage"] >= 0.34:
            return {"refuse": False, "reason": "", "level": "high"}
        if top["score"] >= 0.36 and top["coverage"] >= 0.2:
            return {"refuse": False, "reason": "", "level": "medium"}
        return {
            "refuse": True,
            "reason": "只找到部分相关条款，但不足以形成可靠结论。",
            "level": "low",
        }

    @staticmethod
    def _sentence_score(
        sentence: str, query_tokens: list[str], query_text: str
    ) -> float:
        sentence_tokens = set(tokenize(sentence))
        matched = [token for token in unique_tokens(sentence) if token in query_tokens]
        score = len(matched) * 2.0
        for token in query_tokens:
            if token in sentence_tokens:
                score += 0.35
        if 18 <= len(sentence) <= 150:
            score += 1.2
        if NUMERIC_QUERY_PATTERN.search(query_text) and re.search(r"\d", sentence):
            score += 3.5
        return score

    def _select_sentences(
        self,
        result: dict[str, Any],
        query_tokens: list[str],
        query_text: str,
    ) -> list[dict[str, Any]]:
        sentences = split_sentences(result["section"]["text"])
        scored = [
            {
                "sentence": sentence,
                "score": self._sentence_score(sentence, query_tokens, query_text),
            }
            for sentence in sentences
        ]
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:2]

    def create_answer(self, search_response: dict[str, Any]) -> dict[str, Any]:
        assessment = self._assess_evidence(search_response)
        top_results = search_response["results"][:4]
        trace = {
            "mode": search_response["mode"],
            "scope": search_response["scope"],
            "department": search_response["department"],
            "expanded_terms": search_response["expanded_terms"],
            "unseen_terms": search_response["unseen_terms"],
            "elapsed_ms": search_response["elapsed_ms"],
            "thresholds": search_response["thresholds"],
            "candidates": [
                {
                    "title": result["document"]["title"],
                    "version": result["document"]["version"],
                    "score": result["score"],
                    "coverage": result["coverage"],
                    "status": result["document"]["status"],
                }
                for result in top_results
            ],
        }

        if assessment["refuse"]:
            return {
                "refused": True,
                "confidence": "insufficient",
                "title": "当前知识库中没有足够依据",
                "lead": "为避免提供未经制度支持的结论，本次不作确定性回答。",
                "points": [],
                "note": assessment["reason"],
                "evidence": top_results[:2],
                "trace": trace,
            }

        if not top_results:
            return {
                "refused": True,
                "confidence": "insufficient",
                "title": "当前知识库中没有足够依据",
                "lead": "本次没有取得可引用的制度条款。",
                "points": [],
                "note": "请补充更具体的制度名称或业务场景。",
                "evidence": [],
                "trace": trace,
            }

        top_score = top_results[0]["score"]
        minimum_coverage = max(0.12, top_results[0]["coverage"] * 0.55)
        direct_results = [
            result
            for index, result in enumerate(top_results)
            if index == 0
            or (
                result["score"] >= top_score * 0.65
                and result["coverage"] >= minimum_coverage
            )
        ]

        points: list[dict[str, Any]] = []
        used_sentences: set[str] = set()
        for result in direct_results:
            for candidate in self._select_sentences(
                result, search_response["query_tokens"], search_response["query"]
            ):
                if len(points) >= 4:
                    break
                if candidate["sentence"] in used_sentences:
                    continue
                if candidate["score"] < 1.8 and points:
                    continue
                used_sentences.add(candidate["sentence"])
                document = result["document"]
                section = result["section"]
                points.append(
                    {
                        "text": candidate["sentence"],
                        "doc_id": document["id"],
                        "title": document["title"],
                        "version": document["version"],
                        "section": section["heading"],
                        "page": section["page"],
                        "status": document["status"],
                        "citation": len(points) + 1,
                    }
                )

        if not points:
            result = top_results[0]
            document = result["document"]
            section = result["section"]
            points.append(
                {
                    "text": section["text"],
                    "doc_id": document["id"],
                    "title": document["title"],
                    "version": document["version"],
                    "section": section["heading"],
                    "page": section["page"],
                    "status": document["status"],
                    "citation": 1,
                }
            )

        statuses = {result["document"]["status"] for result in top_results}
        note = "回答由检索到的制度原文抽取生成，请以引用条款的完整上下文为准。"
        if "expired" in statuses and "effective" in statuses:
            note = "本次同时命中现行与历史版本，请结合版本和生效日期使用结论。"
        elif "expired" in statuses:
            note = "当前展示的是历史版本，仅适用于追溯，不应作为现行操作依据。"

        return {
            "refused": False,
            "confidence": assessment["level"],
            "title": "基于制度条款的回答",
            "lead": (
                "已在知识库中找到直接相关的制度依据。"
                if assessment["level"] == "high"
                else "已找到可能相关的条款，建议结合完整原文确认。"
            ),
            "points": points,
            "note": note,
            "evidence": top_results,
            "trace": trace,
        }

    def answer_question(
        self,
        query: str,
        *,
        scope: str = "current",
        department: str = "all",
        mode: str = "hybrid",
        limit: int = 6,
    ) -> dict[str, Any]:
        search_response = self.search(
            query, scope=scope, department=department, mode=mode, limit=limit
        )
        return {
            "search": search_response,
            "answer": self.create_answer(search_response),
        }

    def evaluate(self, mode: str) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        retrieval_cases = 0
        recall_sum = 0.0
        reciprocal_rank_sum = 0.0
        citation_cases = 0
        citation_hits = 0.0
        refusal_cases = 0
        refusal_hits = 0
        latency_sum = 0.0

        for test_case in self.evaluation_cases:
            response = self.search(
                test_case["query"],
                mode=mode,
                scope=test_case.get("scope", "current"),
                department=test_case.get("department", "all"),
                limit=6,
            )
            answer = self.create_answer(response)
            top_three_docs = list(
                dict.fromkeys(
                    result["document"]["id"] for result in response["results"][:3]
                )
            )
            cited_text = " ".join(point["text"] for point in answer["points"])
            found_terms = [
                term for term in test_case["expected_terms"] if term in cited_text
            ]
            expected_docs = test_case["expected_doc_ids"]
            relevant_rank = next(
                (
                    index
                    for index, result in enumerate(response["results"])
                    if result["document"]["id"] in expected_docs
                ),
                -1,
            )
            retrieval_hit = any(doc_id in top_three_docs for doc_id in expected_docs)

            if not test_case["should_refuse"]:
                retrieval_cases += 1
                recall_sum += 1 if retrieval_hit else 0
                reciprocal_rank_sum += (
                    1 / (relevant_rank + 1) if relevant_rank >= 0 else 0
                )
                if test_case["expected_terms"]:
                    citation_cases += 1
                    citation_hits += len(found_terms) / len(
                        test_case["expected_terms"]
                    )
            else:
                refusal_cases += 1
                refusal_hits += 1 if answer["refused"] else 0

            latency_sum += response["elapsed_ms"]
            passed = (
                answer["refused"]
                if test_case["should_refuse"]
                else retrieval_hit
                and len(found_terms) == len(test_case["expected_terms"])
            )
            rows.append(
                {
                    "id": test_case["id"],
                    "query": test_case["query"],
                    "expected_docs": expected_docs,
                    "retrieved_docs": top_three_docs,
                    "retrieval_hit": retrieval_hit,
                    "citation_terms": found_terms,
                    "expected_terms": test_case["expected_terms"],
                    "refused": answer["refused"],
                    "should_refuse": test_case["should_refuse"],
                    "passed": passed,
                    "latency_ms": response["elapsed_ms"],
                }
            )

        return {
            "mode": mode,
            "total": len(self.evaluation_cases),
            "recall_at_3": recall_sum / retrieval_cases if retrieval_cases else 0.0,
            "mrr": (
                reciprocal_rank_sum / retrieval_cases if retrieval_cases else 0.0
            ),
            "citation_accuracy": (
                citation_hits / citation_cases if citation_cases else 0.0
            ),
            "refusal_accuracy": (
                refusal_hits / refusal_cases if refusal_cases else 0.0
            ),
            "average_latency_ms": latency_sum / max(len(self.evaluation_cases), 1),
            "passed": sum(1 for row in rows if row["passed"]),
            "rows": rows,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "corpus_name": self.corpus_meta.get("name", "PolicyLens"),
            "document_count": len(self.documents),
            "base_document_count": len(self.base_documents),
            "uploaded_document_count": len(self.uploaded_documents),
            "chunk_count": len(self.chunks),
            "evaluation_case_count": len(self.evaluation_cases),
        }
