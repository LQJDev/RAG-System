from pydantic import BaseModel, Field
from typing import List

# ------------------- 请求模型 -------------------
class QueryRequest(BaseModel):
    """检索知识库的请求参数"""
    query_text: str = Field(..., description="检索的查询文本", min_length=1)
    top_k: int = Field(3, description="返回最相关的文档数量", ge=1, le=20)  # 限制1-20条

# ------------------- 响应模型 -------------------
class BaseResponse(BaseModel):
    """所有响应的基础模型"""
    status: str = Field("success", description="状态：success 或 error")
    message: str = Field("", description="状态描述或错误信息")

class RetrievedDoc(BaseModel):
    """检索返回的单条文档信息"""
    content: str = Field(..., description="文档内容文本")
    source: str = Field(..., description="文档来源路径")
    similarity_score: float = Field(..., description="与查询的相似度分数（越小越相关）")

class QueryResponse(BaseResponse):
    """检索响应结果"""
    data: List[RetrievedDoc] = Field(default_factory=list, description="检索到的相关文档列表")
