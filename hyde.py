
from langchain_core.language_models import BaseLanguageModel

from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.llms.langchain import LangChainLLM

from llama_index.core.prompts import PromptTemplate

HYDE_PROMPT = PromptTemplate(
    "你是知识库的检索辅助工具。\n"
    "请根据用户问题，生成一段适合语义检索的假设性中文文档。\n"
    "这不是最终答案，不需要知道真实的具体政策。\n"
    "请使用相关的专业词语，写80到120字。\n"
    "不要回答信息不足，不要要求用户补充信息。\n"
    "不要编造具体的确定性规定。\n"
    "只输出假设文档内容，不要解释。\n\n"
    "用户问题：{context_str}\n"
    "假设文档："
)


def hyde_embedding_texts(
    llm: BaseLanguageModel,
    question: str,
    include_original: bool = True,
) -> list[str]:

    # 1. 把已有的 LangChain 模型转换为 LlamaIndex 支持的模型
    llama_llm = LangChainLLM(llm=llm)

    # 2. 创建官方 HyDE 转换器
    hyde = HyDEQueryTransform(
        llm=llama_llm,
        hyde_prompt=HYDE_PROMPT,
        include_original=include_original,
    )

    # 3. 让 LLM 生成假设文档
    query_bundle = hyde.run(question)

    # 4. 返回用于向量检索的文本
    return query_bundle.embedding_strs
