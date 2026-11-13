from pathlib import Path
from milvus_store import COLLECTION_NAME
from pymilvus import MilvusException
from typing import Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from openai import APIError

# 导入 Day 4 实现的 RAG 服务
from rag import RAGService, RerankerUnavailableError


# 1. 请求数据模型
class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=500,
        description="用户提出的问题"
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="返回的相关文本块数量"
    )

    strategy: Literal[
        "naive",
        "hyde",
        "multi_query",
        "multi_step",
        "bm25",
        "hybrid",
        "hybrid_rerank",
    ] = Field(
        default="naive",
        description="选择检索策略",
    )

    include_original: bool = Field(
        default=True,
        description="HyDE 是否包含原始问题",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("问题不能为空")

        return value


# 2. 来源数据模型
class SourceItem(BaseModel):
    id: int
    source: str | None = None
    page: int | None = None
    chunk_id: int | None = None
    score: float


# 3. 问答响应数据模型
class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


# 4. 健康检查响应数据模型
class HealthResponse(BaseModel):
    status: str
    chunks: int
    backend: str


# 5. 应用生命周期：启动时初始化 RAG
@asynccontextmanager
async def lifespan(app: FastAPI):

    print("正在初始化 RAG 服务...")

    # 启动时只创建一次 RAGService
    rag = RAGService()

    app.state.rag = rag

    try:
        print(
            "RAG 初始化完成，检索后端：",
            rag.backend,
        )

        print(
            "文本块数量：",
            len(rag.chunks),
        )

        # FastAPI 在这里开始处理 HTTP 请求
        yield

    finally:
        # 应用关闭时释放 Milvus 客户端连接
        rag.close()

        del app.state.rag

        print("RAG 服务已关闭")



# 6. 创建 FastAPI 应用
app = FastAPI(
    title="Hybrid RAG 文档问答 API",
    description="支持 BM25、Embedding、RRF 和本地 BGE Rerank",
    version="1.1.0",
    lifespan=lifespan,
)


# 7. 健康检查
@app.get("/health", response_model=HealthResponse)
def health(request: Request):

    rag = request.app.state.rag

    return {
        "status": "ok",
        "chunks": len(rag.chunks),
        "backend": rag.backend,
    }


# Milvus 同步状态标记
SYNC_MARKER = (
    Path(__file__).resolve().parent
    / ".milvus_sync_incomplete"
)


@app.get("/ready")
def readiness(request: Request):

    rag = request.app.state.rag

    # NumPy 模式不需要 Milvus
    if rag.backend == "numpy":
        return {
            "status": "ready",
            "backend": "numpy",
        }

    # 防止同步未完成时对外提供服务
    if SYNC_MARKER.exists():
        raise HTTPException(
            status_code=503,
            detail="Milvus 知识库同步未完成",
        )

    try:
        # 发送一个真实的数据库读取请求
        rows = rag.milvus_client.query(
            collection_name=COLLECTION_NAME,
            filter="",
            output_fields=["id"],
            limit=1,
            consistency_level="Strong",
            timeout=3,
        )

        if not rows:
            raise RuntimeError(
                "Milvus Collection 没有可查询的数据"
            )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Milvus 当前未就绪",
        ) from exc

    return {
        "status": "ready",
        "backend": "milvus",
    }

# 8. 文档问答接口
@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request):
    # 获取已经初始化的 RAG 对象
    rag = request.app.state.rag
    
    if (
        rag.backend == "milvus"
        and SYNC_MARKER.exists()
    ):
        raise HTTPException(
            status_code=503,
            detail="知识库正在同步或同步未完成",
        )
    
    try:
        # 调用 Day 4 的完整 RAG 流程
        result = rag.ask(
            question=payload.question,
            top_k=payload.top_k,
            strategy=payload.strategy,
            include_original=payload.include_original,
        )

    except RerankerUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="本地精排模型不可用，请检查模型文件和运行环境",
        ) from exc

    except MilvusException as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Milvus 检索服务不可用，"
                "请检查数据库状态"
            ),
        ) from exc

    except APIError as exc:
        # DeepSeek API 调用失败
        raise HTTPException(
            status_code=502,
            detail="模型服务调用失败，请稍后重试"
        ) from exc

    return result
