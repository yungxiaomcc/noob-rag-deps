from __future__ import annotations

from typing import TYPE_CHECKING

from pymilvus import DataType, Function, FunctionType, MilvusClient

from noob_rag_deps.config import conf
from noob_rag_deps.logging import get_logger
from noob_rag_deps.rag.vdb_schemas import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DENSE_DIM,
    FIELD_DOC_ID,
    FIELD_DENSE_VECTOR,
    FIELD_FILTER_1,
    FIELD_FILTER_2,
    FIELD_FILTER_3,
    FIELD_ID,
    FIELD_IS_DELETE,
    FIELD_METADATA,
    FIELD_SPACE_ID,
    FIELD_SPARSE_VECTOR,
    FIELD_TENANT_ID,
    FIELD_TEXT,
)

if TYPE_CHECKING:
    from pymilvus import CollectionSchema


def create_client(
    db_name: str | None = None,
) -> MilvusClient:
    """创建并返回 MilvusClient。"""
    name = db_name if db_name is not None else conf.rag.milvus.dbname
    return MilvusClient(uri=conf.rag.milvus.uri, db_name=name)


def init_vdb() -> MilvusClient:
    """根据配置初始化 VDB。

    1. 若目标数据库不存在，则自动创建
    2. 返回连接目标数据库的 client
    """
    logger = get_logger("noob_rag_deps.rag.vdb_client")
    client = create_client(db_name=conf.rag.milvus.dbname)
    dbs = client.list_databases()
    target_db = conf.rag.milvus.dbname
    if target_db not in dbs:
        client.create_database(db_name=target_db)
        logger.info("vdb_database_created", dbname=target_db)
    return create_client(db_name=target_db)


def has_collection(collection_name: str, client: MilvusClient | None = None) -> bool:
    """检查 collection 是否存在。"""
    c = client or create_client()
    return collection_name in c.list_collections()



default_analyzer_params = {"type": "chinese","tokenizer": "jieba"}

def create_default_schema(
    dense_dim: int | None = None,
) -> CollectionSchema:
    """创建默认的 schema。

    - id: 主键，auto_id 自动生成
    - doc_id: 文档 ID (VARCHAR)
    - dense_vector: 稠密向量，维度由 dense_dim 指定
    - sparse_vector: 稀疏向量（如 BM25）
    - text: 原始文本
    - metadata: JSON 元数据
    - tenant_id: 租户id，作为 partition key
    - space_id: 租户空间id
    - filter_1: str,自定义过滤条件1
    - filter_2: str,自定义过滤条件2
    - filter_3: str,自定义过滤条件3
    - is_delete: bool, 软删除标记

    """
    dim = dense_dim if dense_dim is not None else conf.rag.dense_dim
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    schema.add_field(
        field_name=FIELD_ID,
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )
    schema.add_field(
        field_name=FIELD_DOC_ID,
        datatype=DataType.VARCHAR,
        max_length=512,
    )
    schema.add_field(
        field_name=FIELD_DENSE_VECTOR,
        datatype=DataType.FLOAT_VECTOR,
        dim=dim,
    )
    schema.add_field(
        field_name=FIELD_SPARSE_VECTOR,
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    )

    
    # https://milvus.io/docs/analyzer-overview.md
    schema.add_field(
        field_name=FIELD_TEXT,
        datatype=DataType.VARCHAR,
        max_length=65535,
        analyzer_params=default_analyzer_params,
        enable_analyzer=True,
        enable_match=True,
    )
    schema.add_field(
        field_name=FIELD_METADATA,
        datatype=DataType.JSON,
    )
    schema.add_field(
        field_name=FIELD_TENANT_ID,
        datatype=DataType.VARCHAR,
        max_length=512,
        is_partition_key=True,
        nullable=False,
    )
    schema.add_field(
        field_name=FIELD_SPACE_ID,
        datatype=DataType.VARCHAR,
        max_length=512,
        nullable=True,
    )
    schema.add_field(
        field_name=FIELD_FILTER_1,
        datatype=DataType.VARCHAR,
        max_length=512,
        nullable=True,
    )
    schema.add_field(
        field_name=FIELD_FILTER_2,
        datatype=DataType.VARCHAR,
        max_length=512,
        nullable=True,
    )
    schema.add_field(
        field_name=FIELD_FILTER_3,
        datatype=DataType.VARCHAR,
        max_length=512,
        nullable=True,
    )
    schema.add_field(
        field_name=FIELD_IS_DELETE,
        datatype=DataType.BOOL,
        default_value=False,
        nullable=False,
    )
    bm25_function = Function(
        name="text_bm25",
        input_field_names=[FIELD_TEXT],
        output_field_names=[FIELD_SPARSE_VECTOR],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)
    return schema


def create_new_collection(
    collection_name: str | None = None,
    client: MilvusClient | None = None,
    schema: CollectionSchema | None = None,
    dense_dim: int | None = None,
) -> None:
    """创建新 collection。

    若未传入 client，则使用 create_client() 创建；
    若未传入 schema，则使用 create_default_schema()。
    会为 dense_vector 和 sparse_vector 创建索引并加载。
    """
    name = collection_name or conf.rag.collection_name
    dim = dense_dim if dense_dim is not None else conf.rag.dense_dim
    if client is None:
        client = create_client()
    if schema is None:
        schema = create_default_schema(dense_dim=dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=FIELD_ID,
        index_type="STL_SORT",
    )
    index_params.add_index(
        field_name=FIELD_DOC_ID,
        index_type="STL_SORT",
    )
    index_params.add_index(
        field_name=FIELD_TENANT_ID,
        index_type="STL_SORT",
    )
    index_params.add_index(
        field_name=FIELD_SPACE_ID,
        index_type="STL_SORT",
    )
    # index_params.add_index(
    #     field_name=FIELD_IS_DELETE,
    #     index_type="STL_SORT",
    # )
    index_params.add_index(
        field_name=FIELD_DENSE_VECTOR,
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    index_params.add_index(
        field_name=FIELD_SPARSE_VECTOR,
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )

    logger = get_logger("noob_rag_deps.rag.vdb_client")
    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )

    indexes = client.list_indexes(collection_name=name)
    logger.info(
        "vdb_collection_created",
        collection_name=name,
        dense_dim=dim,
        indexes=list(indexes),
    )


if __name__ == '__main__':
    ...