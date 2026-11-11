
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


# data 文件夹的位置
DATA_DIR = Path(__file__).resolve().parent / "data"


# 1. 加载 Markdown 和 TXT
def load_text_file(file_path: Path) -> Document | None:
    text = file_path.read_text(encoding="utf-8").strip()

    if not text:
        return None

    return Document(
        page_content=text,
        metadata={
            "source": file_path.name,
            "file_type": file_path.suffix.lower().lstrip("."),
            "page": None,
        },
    )


# 2. 加载 PDF（每页作为一个 Document）
def load_pdf_file(file_path: Path) -> list[Document]:
    documents = []
    reader = PdfReader(str(file_path))

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if not text:
            continue

        doc = Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "file_type": "pdf",
                "page": page_number,
            },
        )
        documents.append(doc)

    return documents


# 3. 自动加载文件夹里的所有支持文件
def load_documents(folder: Path) -> list[Document]:
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    documents = []

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix in {".md", ".txt"}:
            doc = load_text_file(file_path)

            if doc is not None:
                documents.append(doc)

        elif suffix == ".pdf":
            documents.extend(load_pdf_file(file_path))

    return documents


# 4. 测试加载结果
if __name__ == "__main__":
    docs = load_documents(DATA_DIR)

    print(f"成功加载 {len(docs)} 个 Document")

    for index, doc in enumerate(docs, start=1):
        print(f"\n--- Document {index} ---")
        print("metadata:", doc.metadata)
        print("文本长度:", len(doc.page_content))
        print("文本预览:", doc.page_content[:100])
