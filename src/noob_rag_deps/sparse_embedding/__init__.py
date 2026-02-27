from __future__ import annotations

import time

import httpx
from langchain_milvus.utils.sparse import BaseSparseEmbedding

from noob_rag_deps.config import SparseEmbeddingConfig
from noob_rag_deps.logging import get_logger


def _sparse_item_to_dict(item: list[dict]) -> dict[int, float]:
    """Convert TEI sparse embedding item [{"index": int, "value": float}, ...] to Dict[int, float]."""
    result: dict[int, float] = {}
    for entry in item:
        if not isinstance(entry, dict) or "index" not in entry or "value" not in entry:
            raise ValueError(
                f"TEI /embed_sparse item entry unexpected format: expected dict with 'index' and 'value', got {type(entry)}"
            )
        result[int(entry["index"])] = float(entry["value"])
    return result


class NoobSparseEmbedding(BaseSparseEmbedding):
    """
    基于 TEI 部署的 sparse embedding 模型（如 SPLADE）。
    对接 TEI HTTP API（POST /embed_sparse），实现 BaseSparseEmbedding 接口。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 60.0,
        truncate: bool = True,
        *,
        config: SparseEmbeddingConfig | None = None,
    ) -> None:
        if config is not None:
            base_url = config.tei_base_url
            timeout = config.timeout
            truncate = config.truncate
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._truncate = truncate
        self._embed_url = f"{self._base_url}/embed_sparse"

    def _embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        payload: dict = {
            "inputs": texts if len(texts) > 1 else texts[0],
            "truncate": self._truncate,
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(self._embed_url, json=payload)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, list):
            raise ValueError(
                f"TEI /embed_sparse returned unexpected format: expected list, got {type(data)}"
            )
        return [_sparse_item_to_dict(item) for item in data]

    def embed_documents(self, texts: list[str]) -> list[dict[int, float]]:
        if not texts:
            return []
        return self._embed_sparse(texts)

    def embed_query(self, query: str) -> dict[int, float]:
        return self._embed_sparse([query])[0]

    async def _aembed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        logger = get_logger("noob_rag_deps.sparse_embedding")
        item_count = len(texts)
        text_len = sum(len(t) for t in texts)
        t0 = time.perf_counter()
        try:
            payload: dict = {
                "inputs": texts if len(texts) > 1 else texts[0],
                "truncate": self._truncate,
            }
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._embed_url, json=payload)
                response.raise_for_status()
                data = response.json()
            if not isinstance(data, list):
                raise ValueError(
                    f"TEI /embed_sparse returned unexpected format: expected list, got {type(data)}"
                )
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.debug(
                "sparse_embed_done",
                item_count=item_count,
                text_len=text_len,
                latency_ms=round(latency_ms, 2),
            )
            return [_sparse_item_to_dict(item) for item in data]
        except Exception as e:
            logger.warning(
                "sparse_embed_error",
                item_count=item_count,
                text_len=text_len,
                error=str(e),
            )
            raise

    async def aembed_documents(self, texts: list[str]) -> list[dict[int, float]]:
        if not texts:
            return []
        return await self._aembed_sparse(texts)

    async def aembed_query(self, query: str) -> dict[int, float]:
        return (await self._aembed_sparse([query]))[0]
