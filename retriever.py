
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# 复用 Day 2 的索引构建函数和检索指令
from indexer import build_index, QUERY_INSTRUCTION

import json

from pymilvus import MilvusClient

from milvus_store import (
    COLLECTION_NAME,
    VECTOR_DIM,
)

# 1. 手写余弦相似度
def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray
) -> float:

    dot_product = np.dot(a, b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        dot_product / (norm_a * norm_b)
    )


# 2. 检索最相关文本块
def retrieve(
    question: str,
    chunks: list[Document],
    vectors: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 3,
    embedding_texts: list[str] | None = None,
) -> list[dict]:

    # 检查输入
    if not question.strip():
        raise ValueError("问题不能为空")

    if top_k < 1:
        raise ValueError("top_k 必须大于等于1")

    if len(chunks) != len(vectors):
        raise ValueError("文本块和向量数量不一致")


    # 第一步：确定需要转换成向量的文本
    if embedding_texts is None:
        texts = [question]
    else:
        texts = embedding_texts

    if not texts or any(not text.strip() for text in texts):
        raise ValueError("检索文本不能为空")

    # 第二步：分别生成 Embedding
    query_vectors = []

    for text in texts:
        query_text = QUERY_INSTRUCTION + text.strip()

        vector = model.encode(
            query_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        query_vectors.append(vector)

    # 第三步：计算多个向量的平均值
    query_vector = np.mean(
        np.stack(query_vectors),
        axis=0
    )


    # 第四步：逐个计算相似度
    results = []

    for i, chunk in enumerate(chunks):
        score = cosine_similarity(
            query_vector,
            vectors[i]
        )

        results.append({
            "content": chunk.page_content,
            "score": score,
            "metadata": chunk.metadata.copy()
        })

    # 第五步：按照相似度从高到低排序
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # 第六步：取前K个结果
    return results[:top_k]


# 3. 使用 Milvus 进行向量检索
def milvus_retrieve(
    question: str,
    model: SentenceTransformer,
    client: MilvusClient,
    top_k: int = 3,
    embedding_texts: list[str] | None = None,
    source: str | None = None,
) -> list[dict]:

    """返回与原来 retrieve() 相同格式的检索结果。"""

    # 第一步：检查输入
    if not question.strip():
        raise ValueError("问题不能为空")

    if top_k < 1:
        raise ValueError("top_k 必须大于等于 1")

    # 第二步：确定需要 Embedding 的查询文本
    if embedding_texts is None:
        texts = [question]
    else:
        texts = embedding_texts

    if not texts or any(not s.strip() for s in texts):
        raise ValueError("检索文本不能为空")

    # 第三步：生成查询向量
    vectors = []

    for text in texts:
        query_text = QUERY_INSTRUCTION + text.strip()

        vector = model.encode(
            query_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        vectors.append(vector)

    # 与旧版 retrieve() 保持一致：
    # 多个查询向量求平均
    query_vector = np.mean(
        np.stack(vectors),
        axis=0,
    )

    if (
        query_vector.ndim != 1
        or query_vector.shape[0] != VECTOR_DIM
    ):
        raise ValueError(
            f"查询向量维度不正确：{query_vector.shape}"
        )

    # 第四步：准备可选的 Metadata 过滤条件
    filter_expr = ""

    if source is not None:
        filter_expr = (
            "source == "
            + json.dumps(source, ensure_ascii=False)
        )

    # 第五步：交给 Milvus 搜索
    response = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector.tolist()],
        anns_field="vector",
        search_params={
            "metric_type": "COSINE",
            "params": {},
        },
        filter=filter_expr,
        limit=top_k,
        output_fields=[
            "text",
            "source",
            "file_type",
            "page",
            "start_index",
            "chunk_id",
        ],
        consistency_level="Strong",
    )

    # 第六步：转换成原项目使用的结果格式
    results = []

    for hit in response[0]:

        entity = hit["entity"]

        # Day 3 把无页码存为 -1
        # 这里恢复为原来的 None
        raw_page = entity.get("page", -1)

        if raw_page is None or int(raw_page) == -1:
            page = None
        else:
            page = int(raw_page)

        results.append({
            "content": entity["text"],
            "score": float(hit["distance"]),
            "metadata": {
                "source": entity["source"],
                "file_type": entity.get("file_type", ""),
                "page": page,
                "start_index": int(
                    entity.get("start_index", 0)
                ),
                "chunk_id": int(entity["chunk_id"]),
            },
        })

    return results


# 3. 测试检索
if __name__ == "__main__":

    # Day 2：建立文档索引
    chunks, vectors, model = build_index()

    # 用户的问题
    question = "员工有多少天年假？"

    # Day 3：检索
    results = retrieve(
        question=question,
        chunks=chunks,
        vectors=vectors,
        model=model,
        top_k=3
    )

    print("用户问题：", question)

    for i, result in enumerate(results, start=1):
        print(f"\n====== Top {i} ======")
        print("相似度:", round(result["score"], 4))
        print("文本:", result["content"])
        print("来源:", result["metadata"]["source"])
        print("页码:", result["metadata"].get("page"))
        print("块编号:", result["metadata"].get("chunk_id"))
