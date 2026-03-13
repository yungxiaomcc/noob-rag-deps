from __future__ import annotations

import time
from collections.abc import Sequence

import httpx
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor

from noob_rag_deps.config import RerankingConfig
from noob_rag_deps.logging import get_logger


class NoobReranker(BaseDocumentCompressor):
    """
    基于 TEI 部署的 rerank 模型（如 BAAI/bge-reranker-large）
    对接 TEI HTTP API（POST /rerank）。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 60.0,
        truncate: bool = True,
        max_tokens_per_request: int = 1024,
        max_texts_per_request: int = 16,
        *,
        config: RerankingConfig | None = None,
        top_n: int | None = None,
    ) -> None:
        if config is not None:
            base_url = config.tei_base_url
            timeout = config.timeout
            truncate = config.truncate
            max_tokens_per_request = config.max_tokens_per_request
            max_texts_per_request = config.max_texts_per_request
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._truncate = truncate
        self._max_tokens_per_request = max_tokens_per_request
        self._max_texts_per_request = max_texts_per_request
        self._top_n = top_n
        self._rerank_url = f"{self._base_url}/rerank"

    def _make_batches(self, texts: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for text in texts:
            tokens = max(len(text) // 4, 1)
            if (
                current
                and (
                    current_tokens + tokens > self._max_tokens_per_request
                    or len(current) >= self._max_texts_per_request
                )
            ):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(text)
            current_tokens += tokens
        if current:
            batches.append(current)
        return batches

    def _rerank_batch(self, query: str, batch: list[str]) -> list[tuple[int, float]]:
        """单批调用 TEI /rerank，返回该批内 (local_index, score) 列表。"""
        if not batch:
            return []
        payload: dict = {
            "query": query,
            "texts": batch,
            "truncate": self._truncate,
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(self._rerank_url, json=payload)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, list):
            raise ValueError(
                f"TEI /rerank returned unexpected format: expected list, got {type(data)}"
            )
        result: list[tuple[int, float]] = []
        for item in data:
            if not isinstance(item, dict) or "index" not in item or "score" not in item:
                raise ValueError(
                    f"TEI /rerank item unexpected format: expected dict with 'index' and 'score', got {type(item)}"
                )
            result.append((int(item["index"]), float(item["score"])))
        return result

    def _rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        """调用 TEI /rerank，返回 (index, score) 列表，已按 score 降序。"""
        if not texts:
            return []
        batches = self._make_batches(texts)
        merged: list[tuple[int, float]] = []
        offset = 0
        for batch in batches:
            local_ranked = self._rerank_batch(query, batch)
            for idx, score in local_ranked:
                merged.append((offset + idx, score))
            offset += len(batch)
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: object = None,
    ) -> Sequence[Document]:
        docs = list(documents)
        if not docs:
            return []
        texts = [doc.page_content for doc in docs]
        ranked = self._rerank(query, texts)
        ordered = [docs[idx] for idx, _ in ranked]
        if self._top_n is not None:
            ordered = ordered[: self._top_n]
        return ordered

    async def _arerank_batch(
        self, query: str, batch: list[str]
    ) -> list[tuple[int, float]]:
        """单批异步调用 TEI /rerank，返回该批内 (local_index, score) 列表。"""
        if not batch:
            return []
        logger = get_logger("noob_rag_deps.reranking")
        item_count = len(batch)
        t0 = time.perf_counter()
        try:
            payload: dict = {
                "query": query,
                "texts": batch,
                "truncate": self._truncate,
            }
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._rerank_url, json=payload)
                response.raise_for_status()
                data = response.json()
            if not isinstance(data, list):
                raise ValueError(
                    f"TEI /rerank returned unexpected format: expected list, got {type(data)}"
                )
            result: list[tuple[int, float]] = []
            for item in data:
                if not isinstance(item, dict) or "index" not in item or "score" not in item:
                    raise ValueError(
                        f"TEI /rerank item unexpected format: expected dict with 'index' and 'score', got {type(item)}"
                    )
                result.append((int(item["index"]), float(item["score"])))
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.debug(
                "rerank_done",
                item_count=item_count,
                latency_ms=round(latency_ms, 2),
            )
            return result
        except Exception as e:
            logger.warning(
                "rerank_error",
                item_count=item_count,
                error=str(e),
            )
            raise

    async def _arerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        """异步调用 TEI /rerank，返回 (index, score) 列表，已按 score 降序。"""
        if not texts:
            return []
        batches = self._make_batches(texts)
        merged: list[tuple[int, float]] = []
        offset = 0
        for batch in batches:
            local_ranked = await self._arerank_batch(query, batch)
            for idx, score in local_ranked:
                merged.append((offset + idx, score))
            offset += len(batch)
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: object = None,
    ) -> Sequence[Document]:
        docs = list(documents)
        if not docs:
            return []
        texts = [doc.page_content for doc in docs]
        ranked = await self._arerank(query, texts)
        ordered = [docs[idx] for idx, _ in ranked]
        if self._top_n is not None:
            ordered = ordered[: self._top_n]
        return ordered

    async def arerank_with_scores(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> list[tuple[Document, float]]:
        """重排并返回 (Document, rerank_score) 列表，按 rerank_score 降序。"""
        docs = list(documents)
        if not docs:
            return []
        texts = [doc.page_content for doc in docs]
        ranked = await self._arerank(query, texts)
        result: list[tuple[Document, float]] = [
            (docs[idx], score) for idx, score in ranked
        ]
        if self._top_n is not None:
            result = result[: self._top_n]
        return result
