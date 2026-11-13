
import re

from langchain_core.prompts import ChatPromptTemplate
from retriever import retrieve


# 定义 Multi-Query 提示词
QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是企业内部知识库的检索助手。"
        "请根据用户问题生成{num_queries}条不同角度的中文检索问题。"
        "保持原意，不编造公司的具体政策或事实。"
        "每行一个问题，不要编号、解释或回答。",
    ),
    (
        "human",
        "用户问题：{question}",
    ),
])


def generate_queries(
    llm,
    question: str,
    num_queries: int = 3,
) -> list[str]:
    """让 LLM 生成多个查询，保留原始问题。"""

    question = question.strip()

    if not question:
        raise ValueError("问题不能为空")

    if not 1 <= num_queries <= 5:
        raise ValueError("num_queries 必须在1到5之间")

    # 第一步：构造 Prompt
    messages = QUERY_PROMPT.format_messages(
        question=question,
        num_queries=num_queries,
    )

    # 第二步：调用 DeepSeek
    response = llm.invoke(messages)

    # 第三步：把模型输出转换成列表
    queries = [question]

    for line in response.content.splitlines():

        # 去掉可能出现的 1.、2.、- 等编号
        cleaned = re.sub(
            r"^\s*(?:\d+[.、．)]\s*|[-*]\s*)",
            "",
            line,
        )

        cleaned = cleaned.strip().strip('“”"')

        # 跳过空问题和重复问题
        if cleaned and cleaned not in queries:
            queries.append(cleaned)

        # 原始问题 + num_queries 个改写问题
        if len(queries) >= num_queries + 1:
            break

    return queries



def multi_query_retrieve(
    queries: list[str],
    chunks,
    vectors,
    model,
    top_k: int = 3,
    semantic_search=None,
) -> list[dict]:
    """分别检索多个问题，并合并去重。"""

    if not queries:
        raise ValueError("queries 不能为空")

    if top_k < 1:
        raise ValueError("top_k 必须大于等于1")

    # 保存合并后的检索结果
    merged = {}

    # 第一步：每个问题分别检索
    for query in queries:


        if semantic_search is None:
            hits = retrieve(
                question=query,
                chunks=chunks,
                vectors=vectors,
                model=model,
                top_k=top_k,
            )
        else:
            hits = semantic_search(
                question=query,
                top_k=top_k,
            )


        # 第二步：逐条处理检索结果
        for hit in hits:

            metadata = hit["metadata"]
            source = metadata.get("source")
            chunk_id = metadata.get("chunk_id")

            # 用来源 + chunk_id 标识文档块
            if chunk_id is not None:
                key = (source, chunk_id)
            else:
                key = (source, hit["content"])

            # 第三步：合并去重，保留最高分
            if (
                key not in merged
                or hit["score"] > merged[key]["score"]
            ):
                merged[key] = hit

    # 第四步：按相似度排序，返回最终 Top K
    results = sorted(
        merged.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]
