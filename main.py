from typing import Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from openai import APIError

# 导入 Day 4 实现的 RAG 服务
from rag import RAGService


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

    strategy: Literal["naive", "hyde"] = Field(
        default="naive",
        description="选择 Naive 或 HyDE 检索",
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


# 5. 应用生命周期：启动时初始化 RAG
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("正在初始化 RAG 服务...")

    # 只在本次服务启动时建立一次索引
    app.state.rag = RAGService()

    print("RAG 初始化完成！")
    print("文本块数量:", len(app.state.rag.chunks))

    yield

    # 关闭服务时释放引用
    del app.state.rag
    print("RAG 服务已关闭")


# 6. 创建 FastAPI 应用
app = FastAPI(
    title="Naive RAG 文档问答 API",
    description="基于 BGE + DeepSeek Flash 的文档问答系统",
    version="1.0.0",
    lifespan=lifespan
)


# 7. 健康检查
@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    rag = request.app.state.rag

    return {
        "status": "ok",
        "chunks": len(rag.chunks)
    }


# 8. 文档问答接口
@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request):
    # 获取已经初始化的 RAG 对象
    rag = request.app.state.rag

    try:
        # 调用 Day 4 的完整 RAG 流程
        result = rag.ask(
            question=payload.question,
            top_k=payload.top_k,
            strategy=payload.strategy,
            include_original=payload.include_original,
        )

    except APIError as exc:
        # DeepSeek API 调用失败
        raise HTTPException(
            status_code=502,
            detail="模型服务调用失败，请稍后重试"
        ) from exc

    return result
