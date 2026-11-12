
from llama_index.core.schema import (
    TextNode,
    NodeWithScore,
)

from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
)


# 1. 创建本地 Reranker
def build_local_reranker(
    top_n: int = 3,
    model_name: str = "BAAI/bge-reranker-base",
) -> SentenceTransformerRerank:

    if top_n < 1:
        raise ValueError("top_n 必须大于 0")

    reranker = SentenceTransformerRerank(
        model=model_name,
        top_n=top_n,
        device="cpu",
        trust_remote_code=False,
    )

    return reranker


# 2. 对 Hybrid 结果重新排序
def rerank_results(
    question: str,
    candidates: list[dict],
    reranker: SentenceTransformerRerank,
) -> list[dict]:

    if not question.strip():
        raise ValueError("问题不能为空")

    if not candidates:
        return []

    # 第一步：将候选结果转换为节点
    nodes = []

    for i, item in enumerate(candidates):

        node = TextNode(
            text=item["content"],
            id_=str(i),
        )

        node_with_score = NodeWithScore(
            node=node,
            score=float(item["score"]),
        )

        nodes.append(node_with_score)

    # 第二步：本地模型进行精排
    ranked_nodes = reranker.postprocess_nodes(
        nodes,
        query_str=question,
    )

    # 第三步：转换回原项目的数据结构
    results = []

    for hit in ranked_nodes:

        index = int(hit.node.node_id)

        original = candidates[index]

        if hit.score is None:
            raise ValueError("Reranker 未返回相关性分数")

        results.append({
            **original,
            "metadata": original["metadata"].copy(),
            "rrf_score": float(original["score"]),
            "score": float(hit.score),
        })

    return results
