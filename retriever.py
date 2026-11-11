
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# 复用 Day 2 的索引构建函数和检索指令
from indexer import build_index, QUERY_INSTRUCTION


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
