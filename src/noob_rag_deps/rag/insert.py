from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any
import typing as t

import structlog
from pydantic import BaseModel, Field

from noob_rag_deps.config import conf
from noob_rag_deps.logging import get_logger
from noob_rag_deps.dense_embedding import NoobDenseEmbedding
from noob_rag_deps.rag.vdb_client import (
    create_new_collection,
    has_collection,
    init_vdb,
)
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
)


class InsertItem(BaseModel):
    """单条插入记录的校验模型。"""

    text: t.Annotated[str, Field(description='文本',max_length=800)]
    doc_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    filter_1: t.Annotated[str|None, Field(None, description='通用过滤字段1',max_length=20)]
    filter_2: t.Annotated[str|None, Field(None, description='通用过滤字段2',max_length=20)]
    filter_3: t.Annotated[str|None, Field(None, description='通用过滤字段3',max_length=20)]


def _default_doc_id(text: str) -> str:
    """缺省 doc_id：使用 text 的 sha256 前缀。"""
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def _validate_tenant_id(tenant_id: str) -> None:
    """验证 tenant_id 格式：必须为 {collection_name}_{唯一标识符}。"""
    if "_" not in tenant_id:
        raise ValueError(
            "tenant_id 必须符合格式 {collection_name}_{唯一标识符}，需包含下划线"
        )
    collection_name, _, identifier = tenant_id.partition("_")
    if not collection_name:
        raise ValueError(
            "tenant_id 必须符合格式 {collection_name}_{唯一标识符}，collection_name 不能为空"
        )
    if not identifier:
        raise ValueError(
            "tenant_id 必须符合格式 {collection_name}_{唯一标识符}，唯一标识符不能为空"
        )


async def _embed_with_semaphore(client: Any, texts: list[str], semaphore: asyncio.Semaphore) -> Any:
    """在 semaphore 限制下执行 embedding，受 max_concurrent_requests 控制。"""
    async with semaphore:
        return await client.aembed_documents(texts)


class VDBInsertService:
    """
    插入时参数：
    - tenant_id: 租户 ID，格式 {collection_name}_{唯一标识符}，根据前缀映射到 collection。
    - space_id: 租户空间 ID。
    - items: 待插入项列表。每个 item 为 dict，支持字段：
        - text (str, 必填): 原始文本，用于 embedding。
        - doc_id (str, 可选): 文档 ID，缺省时使用 text 的 hash。
        - metadata (dict, 可选): JSON 元数据。
        - filter_1, filter_2, filter_3 (str, 可选): 自定义过滤条件。

    tenant_id 的数量可在配置项中限制，仅有注册的 tenant_id 可使用服务（可选实现）。
    """

    def __init__(
        self,
        tenant_id: str,
        space_id: str,
        items: list[InsertItem],
        *,
        client=None,
    ):
        _validate_tenant_id(tenant_id)
        self.tenant_id = tenant_id
        self.space_id = space_id
        self.items = items
        self._client = client

    @property
    def collection_name(self) -> str:
        return self.tenant_id.split("_")[0]

    @classmethod
    async def create(
        cls,
        tenant_id: str,
        space_id: str,
        items: list[dict[str, Any]],
    ):
        """工厂方法：确保 collection 存在后返回实例。"""
        _validate_tenant_id(tenant_id)
        logger = get_logger("noob_rag_deps.rag.insert")
        client = init_vdb()
        collection_name = tenant_id.split("_")[0]

        if not has_collection(collection_name, client):
            create_new_collection(collection_name=collection_name, client=client)
            logger.info("collection_created", collection_name=collection_name)
        else:
            logger.debug("collection_exists", collection_name=collection_name)
        validated_items = [InsertItem.model_validate(i) for i in items]
        instance = cls(tenant_id=tenant_id, space_id=space_id, items=validated_items, client=client)
        return instance

    async def run(self):
        """执行插入操作。返回插入的 primary keys。"""
        logger = get_logger("noob_rag_deps.rag.insert")
        if not self.items:
            return []
        texts = [item.text for item in self.items]
        item_count = len(self.items)
        text_len = sum(len(t) for t in texts)
        structlog.contextvars.bind_contextvars(
            tenant_id=self.tenant_id,
            space_id=self.space_id,
            collection_name=self.collection_name,
            item_count=item_count,
            text_len=text_len,
        )
        logger.info("vdb_insert_start")
        t0 = time.perf_counter()

        dense_client = NoobDenseEmbedding(config=conf.dense)
        semaphore = asyncio.Semaphore(conf.dense.max_concurrent_requests)
        dense_vectors = await _embed_with_semaphore(dense_client, texts, semaphore)

        rows: list[dict[str, Any]] = []
        for i, item in enumerate(self.items):
            text = texts[i]
            row = {
                FIELD_DOC_ID: item.doc_id or _default_doc_id(text),
                FIELD_DENSE_VECTOR: dense_vectors[i],
                FIELD_TEXT: text,
                FIELD_METADATA: item.metadata,
                FIELD_TENANT_ID: self.tenant_id,
                FIELD_SPACE_ID: self.space_id,
                FIELD_FILTER_1: item.filter_1,
                FIELD_FILTER_2: item.filter_2,
                FIELD_FILTER_3: item.filter_3,
                FIELD_IS_DELETE: False,
            }
            rows.append(row)
        result = self._client.insert(
            collection_name=self.collection_name,
            data=rows,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "vdb_insert_done",
            latency_ms=round(latency_ms, 2),
            inserted_count=len(rows),
        )
        structlog.contextvars.unbind_contextvars(
            "tenant_id", "space_id", "collection_name", "item_count", "text_len"
        )
        return result
