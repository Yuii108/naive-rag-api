
import jieba

from langchain_core.documents import Document
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever


# 1. 中文分词
def chinese_tokens(text: str) -> str:

    tokens = jieba.lcut(text)

    tokens = [
        word.strip()
        for word in tokens
        if word.strip()
        and any(char.isalnum() for char in word)
    ]

    return " ".join(tokens)


# 2. 创建 BM25 检索器
def build_bm25_retriever(
    chunks: list[Document],
    candidate_k: int = 10,
) -> BM25Retriever:

    if not chunks:
        raise ValueError("chunks 不能为空")

    if candidate_k < 1:
        raise ValueError("candidate_k 必须大于 0")

    # 第一步：将原来的 Document 转成 TextNode
    nodes = []

    for i, chunk in enumerate(chunks):

        node = TextNode(
            text=chinese_tokens(chunk.page_content),
            id_=str(i),
        )

        nodes.append(node)

    # 第二步：创建 BM25 索引
    bm25 = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=min(candidate_k, len(nodes)),
        skip_stemming=True,
        token_pattern=r"(?u)\b\w+\b",
        language="en",
    )

    return bm25


# 3. 执行 BM25 检索
def bm25_retrieve(
    question: str,
    chunks: list[Document],
    bm25: BM25Retriever,
    top_k: int = 3,
) -> list[dict]:

    if not question.strip():
        raise ValueError("问题不能为空")

    if top_k < 1:
        raise ValueError("top_k 必须大于 0")

    # 第一步：对问题进行中文分词
    query = chinese_tokens(question)

    # 第二步：使用 BM25 检索
    hits = bm25.retrieve(query)

    # 第三步：整理检索结果
    results = []

    for hit in hits:

        score = float(hit.score or 0.0)

        if score <= 0:
            continue

        # 找回原来的 Chunk
        index = int(hit.node.node_id)
        chunk = chunks[index]

        results.append({
            "content": chunk.page_content,
            "score": score,
            "metadata": chunk.metadata.copy(),
        })

        if len(results) >= top_k:
            break

    return results
