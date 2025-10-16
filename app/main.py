import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 加载环境变量
load_dotenv()

# 导入你的接口路由（与你的代码结构匹配）
from app.api import router as api_router

# 初始化FastAPI应用
app = FastAPI(
    title="RAG 知识库检索系统",
    description="基于Nomic嵌入的文档检索接口",
    version="1.0.0"
)

# 配置跨域（解决前端调用问题）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册你的路由（与你定义的APIRouter匹配）
app.include_router(api_router)

# 根路径测试接口
@app.get("/", tags=["系统状态"])
async def root():
    return {
        "status": "运行中",
        "接口文档": "/docs",
        "检索接口": "/retrieval/query"
    }

if __name__ == "__main__":
    # 启动配置
    uvicorn.run(
        app="main:app",  # 指向当前文件的app实例
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True  # 开发模式自动重载
    )
