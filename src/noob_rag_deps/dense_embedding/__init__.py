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
        *,
        config: DenseEmbeddingConfig | None = None,
    ) -> None:
        if config is not None:
            base_url = config.tei_base_url
            timeout = config.timeout
            truncate = config.truncate
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._truncate = truncate
        self._embed_url = f"{self._base_url}/embed"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload: dict = {"inputs": texts if len(texts) > 1 else texts[0], "truncate": self._truncate}
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(self._embed_url, json=payload)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, list):
            raise ValueError(f"TEI /embed returned unexpected format: expected list, got {type(data)}")
        return [[float(x) for x in vec] for vec in data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    async def _aembed(self, texts: list[str]) -> list[list[float]]:
        logger = get_logger("noob_rag_deps.dense_embedding")
        item_count = len(texts)
        text_len = sum(len(t) for t in texts)
        t0 = time.perf_counter()
        try:
            payload: dict = {"inputs": texts if len(texts) > 1 else texts[0], "truncate": self._truncate}
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._embed_url, json=payload)
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

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._aembed(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return (await self._aembed([text]))[0]
