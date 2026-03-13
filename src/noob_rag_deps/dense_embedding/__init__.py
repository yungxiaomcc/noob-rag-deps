from __future__ import annotations

import time

import httpx
from langchain_core.embeddings import Embeddings

from noob_rag_deps.config import DenseEmbeddingConfig
from noob_rag_deps.logging import get_logger


class NoobDenseEmbedding(Embeddings):
    """
    基于 TEI 部署的 dense embedding 模型。
    对接 TEI HTTP API（POST /embed），实现 LangChain Embeddings 接口。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 60.0,
        truncate: bool = True,
        max_tokens_per_request: int = 1024,
        max_texts_per_request: int = 16,
        *,
        config: DenseEmbeddingConfig | None = None,
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
        self._embed_url = f"{self._base_url}/embed"

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

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        logger = get_logger("noob_rag_deps.dense_embedding")
        payload: dict = {"inputs": batch if len(batch) > 1 else batch[0], "truncate": self._truncate}
        logger.debug("dense_embed_request", payload=payload)
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(self._embed_url, json=payload)
            logger.debug(
                "dense_embed_response",
                status_code=response.status_code,
                ok=response.is_success,
            )
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, list):
            raise ValueError(f"TEI /embed returned unexpected format: expected list, got {type(data)}")
        return [[float(x) for x in vec] for vec in data]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        batches = self._make_batches(texts)
        result: list[list[float]] = []
        for batch in batches:
            result.extend(self._embed_batch(batch))
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    async def _aembed_batch(self, batch: list[str]) -> list[list[float]]:
        logger = get_logger("noob_rag_deps.dense_embedding")
        item_count = len(batch)
        text_len = sum(len(t) for t in batch)
        t0 = time.perf_counter()
        try:
            payload: dict = {"inputs": batch if len(batch) > 1 else batch[0], "truncate": self._truncate}
            logger.debug("dense_embed_request", payload=payload)
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._embed_url, json=payload)
                logger.debug(
                    "dense_embed_response",
                    status_code=response.status_code,
                    ok=response.is_success,
                )
                response.raise_for_status()
                data = response.json()
            if not isinstance(data, list):
                raise ValueError(f"TEI /embed returned unexpected format: expected list, got {type(data)}")
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.debug(
                "dense_embed_done",
                item_count=item_count,
                text_len=text_len,
                latency_ms=round(latency_ms, 2),
            )
            return [[float(x) for x in vec] for vec in data]
        except Exception as e:
            logger.warning(
                "dense_embed_error",
                item_count=item_count,
                text_len=text_len,
                error=str(e),
            )
            raise

    async def _aembed(self, texts: list[str]) -> list[list[float]]:
        batches = self._make_batches(texts)
        result: list[list[float]] = []
        for batch in batches:
            result.extend(await self._aembed_batch(batch))
        return result

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._aembed(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return (await self._aembed([text]))[0]
