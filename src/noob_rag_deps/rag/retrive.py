from __future__ import annotations

import time
from typing import Any

import structlog
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from pymilvus import AnnSearchRequest, WeightedRanker

from noob_rag_deps.config import conf
from noob_rag_deps.logging import get_logger
from noob_rag_deps.dense_embedding import NoobDenseEmbedding
from noob_rag_deps.reranking import NoobReranker
from noob_rag_deps.rag.insert import _validate_tenant_id
from noob_rag_deps.rag.vdb_client import has_collection, init_vdb
from noob_rag_deps.rag.vdb_schemas import (
    FIELD_DOC_ID,
    FIELD_DENSE_VECTOR,
    FIELD_FILTER_1,
    FIELD_FILTER_2,
    FIELD_FILTER_3,
    FIELD_IS_DELETE,
    FIELD_METADATA,
    FIELD_SPACE_ID,
    FIELD_TENANT_ID,
    FIELD_TEXT,
    FIELD_SPARSE_VECTOR
)


class RetrieveParams(BaseModel):
    """召回请求参数，与 InsertItem 对称。"""

    query: str = Field(description="检索 query 文本")
    tenant_id: str = Field(description="租户 ID，格式 {collection_name}_{唯一标识符}")
    space_id: str | None = Field(default=None, description="租户空间 ID，可选")
    filter_1: str | None = Field(default=None, description="通用过滤字段1")
    filter_2: str | None = Field(default=None, description="通用过滤字段2")
    filter_3: str | None = Field(default=None, description="通用过滤字段3")
    top_k: int = Field(default=10, ge=1, le=100, description="召回数量")
    milvus_top_k: int = Field(default=20, ge=1, le=200, description="Milvus 返回的 topk 条数")
    use_rerank: bool = Field(default=True, description="是否启用 rerank")
    rerank_top_n: int = Field(default=5, ge=1, le=50, description="rerank 后保留条数")
    rerank_score_threshold: float | None = Field(
        default=None,
        description="rerank 分数阈值，仅保留 rerank_score >= 阈值的结果；None 表示不按阈值筛选",
    )


class RetrieveResult(BaseModel):
    """召回结果，包含条目列表。"""

    items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="召回条目列表，每项含 text、doc_id、metadata、milvus_score、rerank_score、score",
    )


def _build_filter_expr(
    tenant_id: str,
    space_id: str | None,
    filter_1: str | None,
    filter_2: str | None,
    filter_3: str | None,
) -> str:
    """构建 Milvus 过滤表达式。"""
    parts = [
        f'{FIELD_TENANT_ID} == "{tenant_id}"',
        f"{FIELD_IS_DELETE} == false",
    ]
    if space_id is not None:
        parts.append(f'{FIELD_SPACE_ID} == "{space_id}"')
    if filter_1 is not None:
        parts.append(f'{FIELD_FILTER_1} == "{filter_1}"')
    if filter_2 is not None:
        parts.append(f'{FIELD_FILTER_2} == "{filter_2}"')
    if filter_3 is not None:
        parts.append(f'{FIELD_FILTER_3} == "{filter_3}"')
    return " and ".join(parts)


class VDBRetrieveService:
    """
    召回服务：按 tenant_id/space_id 隔离，dense+sparse 混合检索，可选 filter 与 rerank。
    """

    def __init__(
        self,
        tenant_id: str,
        query: str,
        *,
        space_id: str | None = None,
        filter_1: str | None = None,
        filter_2: str | None = None,
        filter_3: str | None = None,
        top_k: int = 10,
        milvus_top_k: int = 20,
        use_rerank: bool = True,
        rerank_top_n: int = 5,
        rerank_score_threshold: float | None = None,
        client=None,
    ):
        _validate_tenant_id(tenant_id)
        self.tenant_id = tenant_id
        self.query = query
        self.space_id = space_id
        self.filter_1 = filter_1
        self.filter_2 = filter_2
        self.filter_3 = filter_3
        self.top_k = top_k
        self.milvus_top_k = milvus_top_k
        self.use_rerank = use_rerank
        self.rerank_top_n = rerank_top_n
        self.rerank_score_threshold = rerank_score_threshold
        self._client = client

    @property
    def collection_name(self) -> str:
        return self.tenant_id.split("_")[0]

    @classmethod
    async def create(
        cls,
        tenant_id: str,
        query: str,
        *,
        space_id: str | None = None,
        filter_1: str | None = None,
        filter_2: str | None = None,
        filter_3: str | None = None,
        top_k: int = 10,
        milvus_top_k: int = 20,
        use_rerank: bool = True,
        rerank_top_n: int = 5,
        rerank_score_threshold: float | None = None,
    ) -> "VDBRetrieveService":
        """工厂方法：校验 tenant_id，初始化 VDB，检查 collection 存在后返回实例。"""
        _validate_tenant_id(tenant_id)
        logger = get_logger("noob_rag_deps.rag.retrive")
        client = init_vdb()
        collection_name = tenant_id.split("_")[0]
        if not has_collection(collection_name, client):
            raise ValueError(
                f"collection 不存在，无法召回: collection_name={collection_name}"
            )
        logger.debug("retrieve_collection_ready", collection_name=collection_name)
        return cls(
            tenant_id=tenant_id,
            query=query,
            space_id=space_id,
            filter_1=filter_1,
            filter_2=filter_2,
            filter_3=filter_3,
            top_k=top_k,
            milvus_top_k=milvus_top_k,
            use_rerank=use_rerank,
            rerank_top_n=rerank_top_n,
            rerank_score_threshold=rerank_score_threshold,
            client=client,
        )

    async def run(self) -> RetrieveResult:
        """执行召回：embedding -> hybrid_search -> 可选 rerank -> 返回结构化结果。"""
        logger = get_logger("noob_rag_deps.rag.retrive")
        structlog.contextvars.bind_contextvars(
            tenant_id=self.tenant_id,
            space_id=self.space_id,
            collection_name=self.collection_name,
        )
        logger.info("vdb_retrieve_start")
        t0 = time.perf_counter()

        dense_client = NoobDenseEmbedding(config=conf.dense)
        dense_vec = await dense_client.aembed_query(self.query)

        expr = _build_filter_expr(
            self.tenant_id,
            self.space_id,
            self.filter_1,
            self.filter_2,
            self.filter_3,
        )

        limit = self.milvus_top_k

        req_dense = AnnSearchRequest(
            data=[dense_vec],
            anns_field=FIELD_DENSE_VECTOR,
            param={"metric_type": "COSINE", "params": {}},
            limit=limit,
            expr=expr,
        )
        req_sparse = AnnSearchRequest(
            data=[self.query],
            anns_field=FIELD_SPARSE_VECTOR,
            param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
            limit=limit,
            expr=expr,
        )

        # TODO: 混合搜索 权重 expose
        ranker = WeightedRanker(0.8, 0.2)

        res = self._client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[req_dense, req_sparse],
            ranker=ranker,
            limit=limit,
            output_fields=[FIELD_TEXT, FIELD_DOC_ID, FIELD_METADATA],
        )

        hits = res[0] if res else []
        items: list[dict[str, Any]] = []
        for hit in hits:
            entity = getattr(hit, "entity", hit)
            if callable(getattr(entity, "get", None)):
                text = entity.get(FIELD_TEXT)
                doc_id = entity.get(FIELD_DOC_ID)
                metadata = entity.get(FIELD_METADATA)
            else:
                text = getattr(entity, FIELD_TEXT, None)
                doc_id = getattr(entity, FIELD_DOC_ID, None)
                metadata = getattr(entity, FIELD_METADATA, None)
            score = getattr(hit, "distance", None) or getattr(hit, "score", None)
            items.append({
                "text": text or "",
                "doc_id": doc_id or "",
                "metadata": metadata if metadata is not None else {},
                "milvus_score": score,
                "rerank_score": None,
                "score": score,
            })

        if self.use_rerank and items:
            docs = [Document(page_content=item["text"], metadata={"doc_id": item["doc_id"], **item["metadata"]}) for item in items]
            reranker = NoobReranker(config=conf.reranking, top_n=self.rerank_top_n)
            ranked_with_scores = await reranker.arerank_with_scores(docs, self.query)
            items_by_doc_id = {i["doc_id"]: i for i in items}
            items = []
            for doc, rerank_score in ranked_with_scores:
                doc_id = doc.metadata.get("doc_id", "")
                orig = items_by_doc_id.get(doc_id, {})
                items.append({
                    "text": doc.page_content,
                    "doc_id": doc_id,
                    "metadata": {k: v for k, v in doc.metadata.items() if k != "doc_id"},
                    "milvus_score": orig.get("milvus_score"),
                    "rerank_score": rerank_score,
                    "score": rerank_score,
                })

        if self.rerank_score_threshold is not None:
            items = [
                i for i in items
                if i.get("rerank_score") is not None
                and i["rerank_score"] >= self.rerank_score_threshold
            ]

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "vdb_retrieve_done",
            latency_ms=round(latency_ms, 2),
            result_count=len(items),
        )
        structlog.contextvars.unbind_contextvars(
            "tenant_id", "space_id", "collection_name"
        )
        return RetrieveResult(items=items)
