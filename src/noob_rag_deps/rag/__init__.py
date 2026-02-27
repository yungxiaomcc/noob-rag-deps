from noob_rag_deps.rag.insert import VDBInsertService
from noob_rag_deps.rag.retrive import RetrieveParams, RetrieveResult, VDBRetrieveService

__all__ = [
    "VDBInsertService",
    "VDBRetrieveService",
    "RetrieveParams",
    "RetrieveResult",
]

'''
模块功能描述
1. 入库功能：
    - 提供批量文本入库（包含chunking 逻辑）
    - 支持图像
2. 语义检索功能：

'''