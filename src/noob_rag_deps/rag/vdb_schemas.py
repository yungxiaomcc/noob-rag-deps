"""
Collection 的 schema 定义与字段名常量。

供 vdb_client、insert、retrive 等模块复用，避免魔法字符串。
"""

# 默认 collection 名称
DEFAULT_COLLECTION_NAME = "noob_rag"

# 字段名常量
FIELD_ID = "id"
FIELD_DOC_ID = "doc_id"
FIELD_DENSE_VECTOR = "dense_vector"
FIELD_SPARSE_VECTOR = "sparse_vector"
FIELD_TEXT = "text"
FIELD_METADATA = "metadata"
FIELD_TENANT_ID = "tenant_id"
FIELD_SPACE_ID = "space_id"
FIELD_FILTER_1 = "filter_1"
FIELD_FILTER_2 = "filter_2"
FIELD_FILTER_3 = "filter_3"
FIELD_IS_DELETE = "is_delete"

# 默认稠密向量维度
DEFAULT_DENSE_DIM = 1024
