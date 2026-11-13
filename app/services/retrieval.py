from __future__ import annotations

import copy
import hashlib
import logging
import math
import re
import time
from collections import Counter
from typing import Any, Iterable

import jieba
import numpy as np

from app.services.chunking import split_section


jieba.setLogLevel(logging.WARNING)

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
    "相比",
    "对比",
    "变化",
    "差异",
    "怎么办",
    "一般",
    "几天",
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

QUERY_REWRITE_RULES = (
    (re.compile(r"非现金礼品"), "非现金礼品 登记 退回", "非现金礼品→登记退回"),
    (re.compile(r"核心数据"), "核心数据 个人设备", "核心数据→个人设备"),
    (re.compile(r"个人网盘"), "个人网盘 个人设备", "网盘→个人设备"),
    (re.compile(r"资料传到海外|资料传往海外|传到海外"), "个人信息 数据出境", "海外传输→数据出境"),
    (re.compile(r"居家办公"), "远程办公", "居家办公→远程办公"),
    (re.compile(r"家庭网络"), "家庭私有网络", "家庭网络→家庭私有网络"),
    (re.compile(r"盖章"), "合同用印", "盖章→合同用印"),
    (re.compile(r"最多报销"), "上限 报销", "最多报销→报销上限"),
    (re.compile(r"购物卡|礼品卡|预付卡"), "购物卡 礼品 收受", "购物卡→礼品收受"),
    (re.compile(r"匿名举报"), "匿名举报 保密 身份", "匿名举报→身份保密"),
    (re.compile(r"新规|现行规定|现行制度"), "现行制度", "新规→现行制度"),
    (re.compile(r"旧规|老规定|上一版"), "旧版制度", "旧规→旧版制度"),
    (re.compile(r"相比|对比|有什么变化|差异"), "版本 对比", "变化→版本对比"),
    (re.compile(r"酒店费|住宿费|房费"), "住宿标准", "住宿费用→住宿标准"),
    (re.compile(r"打车费|打车费用"), "市内交通 报销", "打车→市内交通"),
    (re.compile(r"资料出境|信息出国"), "个人信息 数据出境", "资料出境→个人信息出境"),
    (re.compile(r"匿名投诉"), "匿名 举报", "投诉→举报"),
    (re.compile(r"设备丢了|电脑丢失"), "设备丢失 报告", "设备丢失→事件报告"),
    (re.compile(r"合同盖章"), "合同 用印", "盖章→合同用印"),
    (re.compile(r"供应商请客|请客吃饭"), "商务招待 供应商", "请客→商务招待"),
)


def normalize(text: str) -> str:
    value = str(text or "").lower()
    value = value.replace("“", "").replace("”", "").replace('"', "")
    value = value.replace("'", "").replace("`", "")
    value = PUNCTUATION_PATTERN.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize(text)
    tokens = [
        token.strip().lower()
        for token in jieba.lcut(normalized, cut_all=False, HMM=True)
        if token.strip()
    ]
    tokens.extend(LATIN_PATTERN.findall(normalized))
    return [
        token
        for token in tokens
        if token
        and token not in STOP_TOKENS
        and not re.fullmatch(r"[\s\W_]+", token)
    ]


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


def rewrite_query(query: str) -> tuple[str, list[str]]:
    rewritten = query
    applied: list[str] = []
    for pattern, replacement, label in QUERY_REWRITE_RULES:
        if pattern.search(rewritten):
            rewritten = pattern.sub(replacement, rewritten)
            applied.append(label)
    return rewritten, applied


class RAGEngine:
    """In-memory hybrid retrieval, extractive answers, and evaluation."""

    def __init__(
        self,
        policies: dict[str, Any],
        documents: list[dict[str, Any]] | None = None,
        semantic_runtime: Any | None = None,
        vector_repository: Any | None = None,
        rerank_enabled: bool = False,
        rerank_candidates: int = 20,
    ) -> None:
        if documents is None:
            self.documents = copy.deepcopy(policies["documents"])
            for document in self.documents:
                document.setdefault("is_builtin", True)
        else:
            self.documents = copy.deepcopy(documents)
        self.synonyms = policies.get("synonyms", {})
        self.synonym_map = _build_synonym_map(self.synonyms)
        self.evaluation_cases = policies.get("evaluation_cases", [])
        self.corpus_meta = policies.get("corpus_meta", {})
        self.semantic_runtime = semantic_runtime
        self.vector_repository = vector_repository
        self.rerank_enabled = rerank_enabled
        self.rerank_candidates = max(1, rerank_candidates)
        self.chunks: list[dict[str, Any]] = []
        self.vocabulary: set[str] = set()
        self.idf: dict[str, float] = {}
        self.average_length = 1.0
        self.semantic_ready = False
        self.vector_index_backend = "disabled"
        self.last_embedding_build_ms = 0.0
        self._register_domain_terms()
        self._build_index()

    def _register_domain_terms(self) -> None:
        terms = set(self.synonym_map)
        for document in self.documents:
            terms.update(document.get("tags", []))
            terms.add(document.get("title", ""))
            terms.add(document.get("short_title", ""))
        for term in terms:
            if term and len(term) >= 2:
                jieba.add_word(str(term), freq=2000)

    def set_documents(self, documents: list[dict[str, Any]]) -> None:
        self.documents = copy.deepcopy(documents)
        self._build_index()

    def _chunks_for_document(
        self, document: dict[str, Any]
    ) -> list[dict[str, Any]]:
        document_chunks: list[dict[str, Any]] = []
        for section_index, section in enumerate(document["sections"]):
            for part in split_section(section):
                searchable_text = " ".join(
                    [
                        document["title"],
                        document.get("short_title", ""),
                        document.get("category", ""),
                        " ".join(document.get("tags", [])),
                        part["heading"],
                        part["text"],
                    ]
                )
                tokens = tokenize(searchable_text)
                part_index = part.get("part_index", 0)
                chunk_key = f"{document['id']}-{section_index + 1}-{part_index + 1}"
                embedding_text = "\n".join(
                    [
                        document["title"],
                        part["heading"],
                        part["text"],
                    ]
                )
                document_chunks.append(
                    {
                        "id": chunk_key,
                        "document": document,
                        "section": part,
                        "section_index": section_index,
                        "part_index": part_index,
                        "searchable_text": searchable_text,
                        "embedding_text": embedding_text,
                        "text_hash": hashlib.sha256(
                            embedding_text.encode("utf-8")
                        ).hexdigest(),
                        "tokens": tokens,
                        "term_frequencies": Counter(tokens),
                        "heading_tokens": set(unique_tokens(part["heading"])),
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
        return document_chunks

    def _sync_semantic_index(self) -> None:
        self.semantic_ready = False
        if self.semantic_runtime is None or self.vector_repository is None:
            return
        started_at = time.perf_counter()
        model_name = self.semantic_runtime.embedding_name
        stored = self.vector_repository.load_embeddings(model_name)
        missing: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []

        for chunk in self.chunks:
            cached = stored.get(chunk["id"])
            if (
                cached
                and cached["text_hash"] == chunk["text_hash"]
                and cached["dimensions"]
            ):
                vector = np.frombuffer(cached["vector"], dtype=np.float32)
                if vector.size == cached["dimensions"]:
                    chunk["semantic_vector"] = vector
                    continue
            missing.append(chunk)

        if missing:
            vectors = self.semantic_runtime.encode_documents(
                [chunk["embedding_text"] for chunk in missing]
            )
            if vectors is not None:
                for chunk, vector in zip(missing, vectors, strict=True):
                    chunk["semantic_vector"] = np.asarray(
                        vector, dtype=np.float32
                    )
                    records.append(
                        {
                            "chunk_key": chunk["id"],
                            "document_id": chunk["document"]["id"],
                            "dimensions": int(len(vector)),
                            "text_hash": chunk["text_hash"],
                            "vector": np.asarray(vector, dtype=np.float32).tobytes(),
                        }
                    )
                self.vector_repository.save_embeddings(model_name, records)

        self.vector_repository.prune_embeddings(
            model_name, {chunk["id"] for chunk in self.chunks}
        )
        if hasattr(self.vector_repository, "sync_vector_index"):
            self.vector_repository.sync_vector_index(model_name)
        if getattr(self.vector_repository, "vector_index_available", False):
            self.vector_index_backend = "sqlite-vec"
        else:
            self.vector_index_backend = "python-memory"
        self.semantic_ready = all(
            "semantic_vector" in chunk for chunk in self.chunks
        )
        self.last_embedding_build_ms = (time.perf_counter() - started_at) * 1000

    def _recalculate_index(self) -> None:
        document_frequency: Counter[str] = Counter()

        for chunk in self.chunks:
            document_frequency.update(set(chunk["tokens"]))

        self.vocabulary = set(document_frequency)
        total_length = sum(len(chunk["tokens"]) for chunk in self.chunks)
        self.average_length = total_length / max(len(self.chunks), 1)
        self.idf = {
            token: math.log(
                1
                + (len(self.chunks) - frequency + 0.5)
                / (frequency + 0.5)
            )
            for token, frequency in document_frequency.items()
        }

        for chunk in self.chunks:
            vector: dict[str, float] = {}
            for token, frequency in chunk["term_frequencies"].items():
                term_frequency = frequency / max(len(chunk["tokens"]), 1)
                vector[token] = term_frequency * self.idf.get(token, 0.0)
            chunk["vector"] = vector
            chunk["vector_norm"] = _vector_norm(vector)

    def _build_index(self) -> None:
        self.chunks = [
            chunk
            for document in self.documents
            for chunk in self._chunks_for_document(document)
        ]
        self._recalculate_index()
        self._sync_semantic_index()

    def rebuild(self, documents: list[dict[str, Any]] | None = None) -> None:
        if documents is not None:
            self.documents = copy.deepcopy(documents)
        self._build_index()

    def refresh_document(self, document: dict[str, Any]) -> None:
        document_copy = copy.deepcopy(document)
        document_id = document_copy["id"]
        self.documents = [
            item for item in self.documents if item["id"] != document_id
        ]
        self.documents.append(document_copy)
        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk["document"]["id"] != document_id
        ]
        self.chunks.extend(self._chunks_for_document(document_copy))
        self._recalculate_index()
        self._sync_semantic_index()

    def remove_document(self, document_id: str) -> bool:
        previous_count = len(self.documents)
        self.documents = [
            item for item in self.documents if item["id"] != document_id
        ]
        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk["document"]["id"] != document_id
        ]
        if len(self.documents) == previous_count:
            return False
        self._recalculate_index()
        self._sync_semantic_index()
        return True

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
        if mode == "hybrid":
            rewritten_query, query_rewrites = rewrite_query(query)
        else:
            rewritten_query, query_rewrites = query, []
        expanded = self._expand_query(rewritten_query)
        query_tokens = (
            expanded["tokens"] if mode == "hybrid" else expanded["original_tokens"]
        )
        unique_query_tokens = set(query_tokens)
        query_vector, query_norm = self._query_vector(query_tokens)
        semantic_started_at = time.perf_counter()
        semantic_query = None
        vector_hits: dict[str, float] = {}
        if (
            self.semantic_ready
            and self.semantic_runtime is not None
            and mode in {"vector", "hybrid"}
        ):
            semantic_query = self.semantic_runtime.encode_query(
                expanded["cleaned"]
            )
            if (
                semantic_query is not None
                and self.vector_repository is not None
                and hasattr(self.vector_repository, "search_similar_vectors")
            ):
                vector_hits = self.vector_repository.search_similar_vectors(
                    self.semantic_runtime.embedding_name,
                    semantic_query,
                    limit=max(50, len(self.chunks)),
                )
        semantic_ms = (time.perf_counter() - semantic_started_at) * 1000
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
            semantic_cosine = 0.0
            if (
                semantic_query is not None
                and chunk.get("semantic_vector") is not None
            ):
                if vector_hits:
                    semantic_cosine = vector_hits.get(chunk["id"], 0.0)
                else:
                    semantic_cosine = float(
                        np.dot(semantic_query, chunk["semantic_vector"])
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
                    "semantic_cosine": semantic_cosine,
                }
            )

        max_bm25 = max((result["bm25"] for result in raw_results), default=0.001)
        for result in raw_results:
            bm25_normalized = result["bm25"] / max(max_bm25, 0.001)
            if mode == "keyword":
                base = bm25_normalized
            elif mode == "vector":
                base = (
                    max(0.0, min(1.0, result["semantic_cosine"]))
                    if semantic_query is not None
                    else result["cosine"]
                )
            else:
                if semantic_query is not None:
                    semantic_normalized = max(
                        0.0, min(1.0, result["semantic_cosine"])
                    )
                    base = (
                        bm25_normalized * 0.44
                        + semantic_normalized * 0.36
                        + result["cosine"] * 0.08
                    )
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
                "semantic_cosine": result["semantic_cosine"],
                "coverage": result["coverage"],
                "field_boost": field_boost,
                "final": result["score"],
            }

        ranked_results = sorted(
            (
                result
                for result in raw_results
                if result["bm25"] > 0
                or result["cosine"] > 0
                or result["semantic_cosine"] > 0
            ),
            key=lambda item: item["score"],
            reverse=True,
        )

        rerank_started_at = time.perf_counter()
        reranked = False
        if (
            mode == "hybrid"
            and self.rerank_enabled
            and self.semantic_runtime is not None
            and ranked_results
        ):
            rerank_candidates = ranked_results[: self.rerank_candidates]
            rerank_scores = self.semantic_runtime.rerank(
                expanded["cleaned"],
                [
                    result["section"]["text"]
                    for result in rerank_candidates
                ],
                batch_size=min(20, len(rerank_candidates)),
            )
            if rerank_scores is not None and len(rerank_scores):
                raw_scores = [float(score) for score in rerank_scores]
                minimum = min(raw_scores)
                maximum = max(raw_scores)
                base_maximum = max(
                    result["score"] for result in rerank_candidates
                )
                for result, raw_score in zip(
                    rerank_candidates, raw_scores, strict=True
                ):
                    normalized = (
                        0.5
                        if math.isclose(minimum, maximum)
                        else (raw_score - minimum) / (maximum - minimum)
                    )
                    base_normalized = result["score"] / max(base_maximum, 0.001)
                    result["rerank_score"] = raw_score
                    result["score"] = base_normalized * 0.62 + normalized * 0.38
                    result["scores"]["rerank"] = raw_score
                    result["scores"]["rerank_normalized"] = normalized
                    result["scores"]["final"] = result["score"]
                rerank_candidates.sort(key=lambda item: item["score"], reverse=True)
                ranked_results = [
                    *rerank_candidates,
                    *ranked_results[self.rerank_candidates :],
                ]
                reranked = True
        rerank_ms = (time.perf_counter() - rerank_started_at) * 1000
        results = ranked_results[:limit]

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
            "original_query": query,
            "rewritten_query": rewritten_query,
            "query_rewrites": query_rewrites,
            "mode": mode,
            "scope": scope,
            "department": department,
            "expanded_terms": expanded["additions"],
            "query_tokens": original_tokens,
            "unseen_terms": unseen_terms,
            "unknown_ratio": unknown_ratio,
            "results": results,
            "semantic_enabled": semantic_query is not None,
            "rerank_applied": reranked,
            "embedding_model": (
                self.semantic_runtime.embedding_name
                if semantic_query is not None
                else None
            ),
            "reranker_model": (
                self.semantic_runtime.reranker_name if reranked else None
            ),
            "embedding_build_ms": self.last_embedding_build_ms,
            "vector_index_backend": self.vector_index_backend,
            "semantic_ms": semantic_ms,
            "rerank_ms": rerank_ms,
            "elapsed_ms": (time.perf_counter() - started_at) * 1000,
            "thresholds": {
                "minimum_score": 0.4,
                "minimum_coverage": 0.2,
                "maximum_unknown_ratio": 0.3,
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
        rerank_score = top.get("rerank_score")
        if (
            search_response["rerank_applied"]
            and rerank_score is not None
            and rerank_score < 0.25
        ):
            return {
                "refuse": True,
                "reason": "Cross-Encoder 判断现有条款与问题不直接匹配。",
                "level": "low",
            }
        if (
            search_response["unknown_ratio"]
            > (
                0.6
                if search_response["rerank_applied"]
                else thresholds["maximum_unknown_ratio"]
            )
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
        if (
            re.search(r"多久|时限|完成|反馈|报告", query_text)
            and re.search(r"\d+\s*(小时|个工作日|工作日)", sentence)
        ):
            score += 2.5
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
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:3]

    def create_answer(self, search_response: dict[str, Any]) -> dict[str, Any]:
        assessment = self._assess_evidence(search_response)
        top_results = search_response["results"][:4]
        trace = {
            "mode": search_response["mode"],
            "scope": search_response["scope"],
            "department": search_response["department"],
            "query_rewrites": search_response["query_rewrites"],
            "expanded_terms": search_response["expanded_terms"],
            "unseen_terms": search_response["unseen_terms"],
            "elapsed_ms": search_response["elapsed_ms"],
            "semantic_ms": search_response["semantic_ms"],
            "rerank_ms": search_response["rerank_ms"],
            "semantic_enabled": search_response["semantic_enabled"],
            "rerank_applied": search_response["rerank_applied"],
            "embedding_model": search_response["embedding_model"],
            "reranker_model": search_response["reranker_model"],
            "vector_index_backend": search_response["vector_index_backend"],
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
        is_version_comparison = any(
            "版本对比" in item
            for item in search_response["query_rewrites"]
        )
        eligible_results = [
            result
            for index, result in enumerate(top_results)
            if index == 0
            or (
                result["score"] >= top_score * 0.55
                and result["coverage"] >= minimum_coverage
            )
        ]
        if is_version_comparison:
            seen_documents: set[str] = set()
            direct_results = []
            for result in eligible_results:
                document_id = result["document"]["id"]
                if document_id in seen_documents:
                    continue
                seen_documents.add(document_id)
                direct_results.append(result)
        else:
            direct_results = eligible_results

        points: list[dict[str, Any]] = []
        used_sentences: set[str] = set()
        for result in direct_results:
            candidates = self._select_sentences(
                result, search_response["query_tokens"], search_response["query"]
            )
            if is_version_comparison:
                candidates = candidates[:1]
            for candidate in candidates:
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
        recall3_sum = 0.0
        recall5_sum = 0.0
        reciprocal_rank_sum = 0.0
        citation_cases = 0
        citation_hits = 0.0
        refusal_cases = 0
        refusal_hits = 0
        latency_sum = 0.0
        semantic_latency_sum = 0.0
        rerank_latency_sum = 0.0
        embedding_latency_sum = 0.0

        for test_case in self.evaluation_cases:
            response = self.search(
                test_case["query"],
                mode=mode,
                scope=test_case.get("scope", "current"),
                department=test_case.get("department", "all"),
                limit=6,
            )
            answer = self.create_answer(response)
            top_five_docs = list(
                dict.fromkeys(
                    result["document"]["id"] for result in response["results"][:5]
                )
            )
            top_three_docs = set(
                result["document"]["id"] for result in response["results"][:3]
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
            retrieval_hit = any(doc_id in top_five_docs for doc_id in expected_docs)
            retrieval_hit_at_3 = any(
                doc_id in top_three_docs for doc_id in expected_docs
            )

            if not test_case["should_refuse"]:
                retrieval_cases += 1
                recall3_sum += 1 if retrieval_hit_at_3 else 0
                recall5_sum += 1 if retrieval_hit else 0
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
            semantic_latency_sum += response["semantic_ms"]
            rerank_latency_sum += response["rerank_ms"]
            embedding_latency_sum += response["embedding_build_ms"]
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
                    "retrieved_docs": top_five_docs,
                    "retrieval_hit": retrieval_hit,
                    "retrieval_hit_at_3": retrieval_hit_at_3,
                    "citation_terms": found_terms,
                    "expected_terms": test_case["expected_terms"],
                    "refused": answer["refused"],
                    "should_refuse": test_case["should_refuse"],
                    "passed": passed,
                    "latency_ms": response["elapsed_ms"],
                    "semantic_ms": response["semantic_ms"],
                    "rerank_ms": response["rerank_ms"],
                }
            )

        return {
            "mode": mode,
            "total": len(self.evaluation_cases),
            "recall_at_3": recall3_sum / retrieval_cases if retrieval_cases else 0.0,
            "recall_at_5": recall5_sum / retrieval_cases if retrieval_cases else 0.0,
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
            "average_semantic_ms": semantic_latency_sum
            / max(len(self.evaluation_cases), 1),
            "average_rerank_ms": rerank_latency_sum
            / max(len(self.evaluation_cases), 1),
            "embedding_build_ms": embedding_latency_sum
            / max(len(self.evaluation_cases), 1),
            "estimated_cost_usd": 0.0,
            "semantic_enabled": self.semantic_ready,
            "rerank_enabled": self.rerank_enabled,
            "passed": sum(1 for row in rows if row["passed"]),
            "rows": rows,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "corpus_name": self.corpus_meta.get("name", "PolicyLens"),
            "document_count": len(self.documents),
            "base_document_count": sum(
                1 for document in self.documents if document.get("is_builtin")
            ),
            "uploaded_document_count": sum(
                1 for document in self.documents if not document.get("is_builtin")
            ),
            "chunk_count": len(self.chunks),
            "evaluation_case_count": len(self.evaluation_cases),
            "semantic_enabled": self.semantic_ready,
            "rerank_enabled": self.rerank_enabled,
            "embedding_model": (
                self.semantic_runtime.embedding_name
                if self.semantic_runtime is not None
                else None
            ),
            "reranker_model": (
                self.semantic_runtime.reranker_name
                if self.semantic_runtime is not None and self.rerank_enabled
                else None
            ),
            "embedding_build_ms": self.last_embedding_build_ms,
            "vector_index_backend": self.vector_index_backend,
        }
