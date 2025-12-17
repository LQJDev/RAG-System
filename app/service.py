import os
from typing import List, Dict, Any
from fastapi import HTTPException
from dotenv import load_dotenv

# LangChain 相关
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 你的业务模块
from core.vector_store.vector_store_manager import create_or_load_vector_store
from core.embeddings.bge_large_zh_embeddings import get_bge_large_zh_embeddings

# ✅ 新增：导入重排序库
from sentence_transformers import CrossEncoder


class KnowledgeBaseService:
    """知识库业务逻辑服务类"""

    def __init__(self):
        self.vector_stores_dir = '../core/vector_store/md_chroma_db'

        # ✅ 优化 1：在初始化时加载重排序模型（只加载一次，常驻内存）
        # 请修改为你实际下载的模型路径，例如 "F:/huggingface/bge-reranker-v2-m3"
        rerank_model_path = os.getenv("RERANK_MODEL_PATH", "F:/huggingface/bge-reranker-v2-m3")

        print(f"正在加载重排序模型: {rerank_model_path} ...")
        try:
            self.reranker = CrossEncoder(rerank_model_path, max_length=512)
            print("重排序模型加载成功！")
        except Exception as e:
            print(f"重排序模型加载失败，请检查路径: {e}")
            raise e

    def query_knowledge_base_by_name(self, query_text: str, kb_name: str, top_k: int) -> List[Dict[str, Any]]:
        """
        RAG 完整流程：向量检索(Recall) -> 重排序(Rerank) -> 大模型生成(Generation)
        """
        load_dotenv()

        # 设定一个重排序的最低得分阈值 (根据模型不同而不同，BGE-Reranker 的 Logits 通常 > 0 表示相关)
        # 你可以根据实际测试调整这个值，比如 -2 到 2 之间
        RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.0"))

        try:
            # 1. 初始化嵌入模型和向量库 (保持原有逻辑)
            embeddings = get_bge_large_zh_embeddings()
            vector_store = create_or_load_vector_store(
                embeddings=embeddings,
                persist_dir=self.vector_stores_dir,
                documents=None,
                batch_size=20,
                name=kb_name
            )

            # ✅ 优化 2：扩大召回范围 (Recall Phase)
            # 我们先从向量库拿 10 倍的数据（比如 50 条），防止漏掉潜在答案
            initial_k = top_k * 10
            # 注意：similarity_search_with_score 返回的是 (Document, distance)
            vector_results = vector_store.similarity_search_with_score(query_text, k=initial_k)

            if not vector_results:
                return self._call_llm_general(query_text)

            # ✅ 优化 3：执行重排序 (Rerank Phase)
            print(f"向量检索召回 {len(vector_results)} 条，正在进行重排序...")

            # 提取文档内容，构造 Pair 对: [[Query, Doc1], [Query, Doc2], ...]
            doc_contents = [doc.page_content for doc, _ in vector_results]
            pairs = [[query_text, doc_text] for doc_text in doc_contents]

            # 计算得分 (Scores)
            rerank_scores = self.reranker.predict(pairs)

            # 将结果打包: (Document, RerankScore)
            # vector_results[i][0] 是 document 对象
            combined_results = []
            for i in range(len(vector_results)):
                combined_results.append({
                    "doc": vector_results[i][0],
                    "score": float(rerank_scores[i])
                })

            # 按重排序分数从高到低排序
            combined_results.sort(key=lambda x: x["score"], reverse=True)

            # ✅ 优化 4：截取 Top-K 并应用阈值过滤
            final_docs = []
            for item in combined_results[:top_k]:
                # 只有分数超过阈值才认为是有效上下文
                if item["score"] > RERANK_THRESHOLD:
                    final_docs.append(item)
                    print(f"保留文档 [分值: {item['score']:.4f}]: {item['doc'].page_content[:20]}...")
                else:
                    print(f"丢弃文档 [分值: {item['score']:.4f}]: 低于阈值")

            # ✅ 优化 5：构建 Prompt 并调用 DeepSeek
            # 如果过滤后没有文档了，就走通用问答
            if not final_docs:
                print("重排序后无相关文档，转入通用问答模式。")
                return self._call_llm_general(query_text)

            # 拼接上下文
            context_text = "\n\n".join(
                [f"[资料{i + 1}]: {item['doc'].page_content}" for i, item in enumerate(final_docs)]
            )

            # 构建消息
            system_prompt = (
                "你是一个智能知识库助手。请根据下方提供的【参考资料】回答用户问题。\n"
                "要求：\n"
                "1. 答案必须基于参考资料，不要编造。\n"
                "2. 如果参考资料无法回答问题，请说明并尝试用你的通用知识补充，但要区分开来源。\n"
                "3. 语言通顺，逻辑清晰。"
            )

            user_prompt = f"用户问题：{query_text}\n\n【参考资料】：\n{context_text}"

            chatModel = ChatOpenAI(
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
                temperature=0.3
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            print("正在请求 DeepSeek 生成答案...")
            response = chatModel.invoke(messages)

            # 返回结果，包含引用的源信息
            best_match_score = final_docs[0]['score']
            primary_source = final_docs[0]['doc'].metadata.get("source", "知识库")

            return [
                {
                    "content": response.content,
                    "source": primary_source,
                    "similarity_score": best_match_score,  # 这里返回的是重排序的置信度
                    "retrieved_count": len(final_docs)
                }
            ]

        except Exception as e:
            print(f"处理流程出错: {e}")
            raise HTTPException(status_code=500, detail=f"检索服务异常：{str(e)}")

    def _call_llm_general(self, query_text: str) -> List[Dict[str, Any]]:
        """
        兜底方法：当知识库没有相关内容时，直接问大模型
        """
        try:
            chatModel = ChatOpenAI(
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
                temperature=0.5  # 通用闲聊可以稍微高一点
            )
            messages = [
                SystemMessage(content="你是一个有用的助手。用户的问题在知识库中未找到答案，请利用你自己的知识回答。"),
                HumanMessage(content=query_text)
            ]
            response = chatModel.invoke(messages)
            return [{
                "content": response.content,
                "source": "AI通用知识",
                "similarity_score": 0.0
            }]
        except Exception:
            return [{"content": "抱歉，无法处理您的请求。", "source": "系统错误", "similarity_score": 0}]