
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from llama_index.retrievers.bm25 import BM25Retriever

# 导入原项目已有的两个检索函数
from retriever import retrieve
from bm25_retriever import bm25_retrieve


# 1. RRF 排名融合
def rrf_fuse(
    embedding_results: list[dict],
    bm25_results: list[dict],
    top_k: int = 3,
    rrf_k: int = 60,
) -> list[dict]:

    if top_k < 1 or rrf_k < 1:
        raise ValueError("top_k 和 rrf_k 必须大于 0")

    # 用于保存融合结果
    fused = {}

    # 分别处理两种检索结果
    for method, results in (
        ("embedding", embedding_results),
        ("bm25", bm25_results),
    ):

        # rank 从 1 开始
        for rank, item in enumerate(results, start=1):

            chunk_id = item["metadata"]["chunk_id"]

            # 第一次遇到这个 Chunk
            if chunk_id not in fused:
                fused[chunk_id] = {
                    "content": item["content"],
                    "metadata": item["metadata"].copy(),
                    "score": 0.0,
                    "embedding_rank": None,
                    "bm25_rank": None,
                }

            # 累加 RRF 分数
            fused[chunk_id]["score"] += (
                1.0 / (rrf_k + rank)
            )

            # 记录该文档原来的排名
            fused[chunk_id][f"{method}_rank"] = rank

    # 按融合分数降序排列
    return sorted(
        fused.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]


# 2. Hybrid 混合检索
def hybrid_retrieve(
    question: str,
    chunks: list[Document],
    vectors: np.ndarray,
    model: SentenceTransformer,
    bm25: BM25Retriever,
    top_k: int = 3,
    candidate_k: int = 5,
    semantic_search=None,
) -> list[dict]:

    if top_k < 1 or candidate_k < top_k:
        raise ValueError("要求 1 <= top_k <= candidate_k")


    # 优先使用传进来的统一检索函数
    if semantic_search is None:
        # 旧版 NumPy 检索，保持兼容
        embedding_results = retrieve(
            question=question,
            chunks=chunks,
            vectors=vectors,
            model=model,
            top_k=candidate_k,
        )
    else:
        # 新版：由 RAGService 决定 NumPy 或 Milvus
        embedding_results = semantic_search(
            question=question,
            top_k=candidate_k,
        )


    # 第二步：BM25 召回候选文档
    bm25_results = bm25_retrieve(
        question=question,
        chunks=chunks,
        bm25=bm25,
        top_k=candidate_k,
    )

    # 第三步：RRF 融合
    return rrf_fuse(
        embedding_results,
        bm25_results,
        top_k=top_k,
    )
