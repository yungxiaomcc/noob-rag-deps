from noob_rag_deps.dense_embedding import NoobDenseEmbedding
from noob_rag_deps.sparse_embedding import NoobSparseEmbedding
from noob_rag_deps.reranking import NoobReranker
from .config import conf

dense_embedding_client = NoobDenseEmbedding(
    base_url=conf.dense.tei_base_url,
    timeout=conf.dense.timeout,
    truncate=conf.dense.truncate
    )

sparse_embedding_client = NoobSparseEmbedding(
    base_url=conf.sparse.tei_base_url,
    timeout=conf.sparse.timeout,
    truncate=conf.sparse.truncate
    )

reranking_client = NoobReranker(
    base_url=conf.reranking.tei_base_url,
    timeout=conf.reranking.timeout,
    truncate=conf.reranking.truncate,
    top_n=2
    )