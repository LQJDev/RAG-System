import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from fastapi import HTTPException

# 导入核心层功能（替换为你的实际模块路径）
from core.embeddings.nomic_embeddings import get_nomic_embeddings
from core.processor.file_processor import load_documents  # 你的文档加载函数
from core.text_splitter.text_splitter import split_documents  # 文本分割工具
from core.vector_store.vector_store_manager import create_or_load_vector_store
from dotenv import load_dotenv

class KnowledgeBaseService:
    """知识库业务逻辑服务类"""

    def __init__(self):
        self.vector_stores_dir = '../core/vector_store/nomic_chroma_db'  # 向量库目录（如 data/vector_stores/）

    def query_knowledge_base(self, query_text: str, top_k: int) -> List[
        Dict[str, Any]]:
        """检索知识库"""
        try:
            load_dotenv()
            nomic_api_key = os.getenv('NOMIC_API_KEY')
            # 初始化嵌入模型和向量库
            embeddings = get_nomic_embeddings(api_key=nomic_api_key)
            vector_store = create_or_load_vector_store(embeddings, self.vector_stores_dir, documents=None, batch_size=20)

            # 检索并返回结果
            results = vector_store.similarity_search_with_score(query_text, k=top_k)
            return [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "未知来源"),
                    "similarity_score": round(float(score), 4)
                }
                for doc, score in results
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"检索失败：{str(e)}")
