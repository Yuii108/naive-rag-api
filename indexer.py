
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# 导入 Day 1 写好的文档加载器
from loader import DATA_DIR, load_documents


MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# Day 3 检索问题时会用到
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


# 1. 文档分块
def split_docs(
    docs: list[Document],
    chunk_size: int = 80,
    chunk_overlap: int = 20,
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=[
            "\n\n", "\n",
            "。", "！", "？", "；",
            "，", " ", ""
        ],
    )

    chunks = splitter.split_documents(docs)

    # 为每个文本块增加编号
    for i, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = i

    return chunks



# 2. 只准备文档块，不计算 Embedding
def prepare_chunks(
    folder: Path = DATA_DIR,
) -> list[Document]:

    docs = load_documents(folder)

    if not docs:
        raise ValueError("没有找到可以加载的文档")

    chunks = split_docs(docs)

    if not chunks:
        raise ValueError("文档分块结果为空")

    return chunks


# 3. 单独加载 Embedding 模型
def load_embedding_model() -> SentenceTransformer:

    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )


# 4. 保留原来的 build_index 接口
def build_index(
    folder: Path = DATA_DIR,
) -> tuple[list[Document], np.ndarray, SentenceTransformer]:

    # 复用新增的文档准备函数
    chunks = prepare_chunks(folder)

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    # 只有调用 build_index 时才加载模型
    model = load_embedding_model()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return chunks, vectors, model



# 测试索引构建
if __name__ == "__main__":
    chunks, vectors, model = build_index()

    print("文本块数量:", len(chunks))
    print("向量矩阵形状:", vectors.shape)

    print("\n第一个文本块:")
    print(chunks[0].page_content)

    print("\n第一个文本块的 metadata:")
    print(chunks[0].metadata)

    print("\n第一个向量的前5个数字:")
    print(vectors[0][:5])
