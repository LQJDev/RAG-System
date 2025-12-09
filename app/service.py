import os
from typing import List, Dict, Any
import dotenv
from fastapi import HTTPException
from core.vector_store.vector_store_manager import create_or_load_vector_store
from dotenv import load_dotenv
from core.embeddings.bge_large_zh_embeddings import get_bge_large_zh_embeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class KnowledgeBaseService:
    """知识库业务逻辑服务类"""

    def __init__(self):
        self.vector_stores_dir = '../core/vector_store/md_chroma_db'  # 向量库目录（如 data/vector_stores/）

    def query_knowledge_base(self, query_text: str, top_k: int) -> List[
        Dict[str, Any]]:
        """检索知识库"""
        try:
            # 初始化嵌入模型和向量库
            embeddings = get_bge_large_zh_embeddings()
            vector_store = create_or_load_vector_store(embeddings, self.vector_stores_dir, documents=None, batch_size=20)

            # 检索并返回结果
            results = vector_store.similarity_search_with_score(query_text, k=top_k)


            return [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "未知来源"),
                    "similarity_score": round(float(1.0 - score), 4)
                }
                for doc, score in results
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"检索失败：{str(e)}")


    def query_knowledge_base_by_name(self, query_text: str, kb_name: str, top_k: int) -> List[
        Dict[str, Any]]:
        """检索知识库"""
        load_dotenv()
        threshold_str = os.getenv("SIMILARITY_THRESHOLD", None)
        if threshold_str is None:
            raise HTTPException(status_code=500, detail="环境变量 SIMILARITY_THRESHOLD 未设置")
        try:
            SIMILARITY_THRESHOLD = float(threshold_str)
        except ValueError:
            raise HTTPException(status_code=500, detail="环境变量 SIMILARITY_THRESHOLD 无法转换为浮点数")
        try:
            # 初始化嵌入模型和向量库
            embeddings = get_bge_large_zh_embeddings()
            vector_store = create_or_load_vector_store(
                embeddings=embeddings,
                persist_dir=self.vector_stores_dir,
                documents=None,
                batch_size=20,
                name=kb_name
            )
            # 检索并返回结果
            results = vector_store.similarity_search_with_score(query_text, k=top_k)
            if not results:
                return []

            # 拼接检索内容
            context_text = "\n\n".join(
                [f"[来源{idx + 1}]: {doc.page_content}" for idx, (doc, _) in enumerate(results)]
            )

            best_doc, best_score = min(results, key=lambda x:x[1])
            print(f"知识库最优检索结果是: {best_doc.page_content}")
            print(f"知识库前{top_k}个匹配结果")
            for res in results:
                print(res[0].page_content)
                print("===============")

            # 如果最优距离大于阈值，将检索信息输入给DeepSeek中生成答案
            if best_score > SIMILARITY_THRESHOLD:
                chatModel = ChatOpenAI(
                    model="deepseek-chat",  # DeepSeek 官方推荐模型名
                    api_key=os.getenv("DEEPSEEK_API_KEY"),  # ✅ 换成你的实际 key
                    base_url="https://api.deepseek.com/v1",  # DeepSeek 官方 API 地址
                    temperature=0.3
                )
                messages = [
                    SystemMessage(content="你是一个知识库问答助手。如果检索内容与问题不符，请结合你自己的掌握的知识, 自主生成答案；如果检索内容充分合理，请直接引用"),
                    HumanMessage(content=(
                        f"用户问题：{query_text}\n"
                        f"检索到的相关内容共{top_k}条：\n{context_text}\n\n"
                        "请基于上述内容，用中文直接回答用户的问题"
                    ))
                ]
                response = chatModel.invoke(messages)
                return [
                    {
                        "content": response.content,
                        "source": "ai生成",
                        "similarity_score": 1
                    }
                ]
            else:
                return [
                    {
                        "content": best_doc.page_content,
                        "source": best_doc.metadata.get("source", "未知来源"),
                        "similarity_score": best_score
                    }
                ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"检索失败：{str(e)}")
