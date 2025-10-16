from fastapi import APIRouter, UploadFile, File, Form, Depends
from typing import List

from .schemas import (
    QueryRequest,
    QueryResponse
)

from .service import KnowledgeBaseService

# 初始化路由
router = APIRouter(tags=["RAG 知识库接口"])

# 依赖注入：创建业务逻辑服务实例（每次请求复用）
def get_kb_service():
    return KnowledgeBaseService()

# ------------------- 检索接口 -------------------
@router.post("/retrieval/query", response_model=QueryResponse, summary="检索知识库中的相关文档")
async def query_kb(
    request: QueryRequest,  # 自动解析请求参数
    service: KnowledgeBaseService = Depends(get_kb_service)
):
    print("后端收到的参数：", request.query_text, request.top_k)  # 新增打印
    """从指定知识库中检索与查询文本相关的文档，返回内容、来源和相似度分数"""
    results = service.query_knowledge_base(
        query_text=request.query_text,
        top_k=request.top_k,
    )
    return QueryResponse(
        status="success",
        message=f"检索到 {len(results)} 条相关文档",
        data=results
    )