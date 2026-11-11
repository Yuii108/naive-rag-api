
from rag import RAGService
from hyde import hyde_embedding_texts
from retriever import retrieve

# 复用原有 RAG 服务
rag = RAGService()

question = "公司允许在家办公吗？"

# 1. 生成 HyDE 假设文档
texts = hyde_embedding_texts(
    llm=rag.llm,
    question=question,
    include_original=True,
)

print("====== HyDE 生成的检索文本 ======")

for i, text in enumerate(texts, start=1):
    print(f"\n文本 {i}:")
    print(text)


# 2. 普通检索
naive_results = retrieve(
    question=question,
    chunks=rag.chunks,
    vectors=rag.vectors,
    model=rag.embedding_model,
    top_k=3,
)


# 3. HyDE 检索
hyde_results = retrieve(
    question=question,
    chunks=rag.chunks,
    vectors=rag.vectors,
    model=rag.embedding_model,
    top_k=3,
    embedding_texts=texts,
)


# 4. 比较两种检索结果
for name, results in [
    ("Naive", naive_results),
    ("HyDE", hyde_results),
]:
    print(f"\n====== {name} ======")

    for item in results:
        print(
            item["metadata"].get("source"),
            round(item["score"], 4),
            item["content"][:100],
        )

