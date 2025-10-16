import os.path
from typing import List

from exceptiongroup import catch
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma

def init_vector_store(embeddings: Embeddings, persist_dir: str) -> Chroma:
    """初始化向量存储，支持加载已有持久化数据"""
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

def add_documents_to_store(
        vector_store: Chroma,
        documents: List[Document],
        batch_size: int
) -> int:
    """向向量存储田间文档"""
    if not documents:
        return 0
    total_added = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"🧩 正在添加文档 {i} ~ {i + len(batch) - 1}")
        try:
            vector_store.add_documents(batch)
            total_added += len(batch)
        except Exception as e:
            print(f"⚠️ 批次 {i} 失败：{str(e)}")
    return total_added


def create_or_load_vector_store(
        embeddings: Embeddings,
        persist_dir: str,
        documents: List[Document],
        batch_size: int
) -> Chroma:
    """
        加载已存在的向量存储，如果不存在则创建新的并添加文档（如果提供）

        参数:
            embeddings: 嵌入模型
            persist_dir: 持久化目录
            documents: 可选，新文档列表（仅在创建新存储时使用）

        返回:
            向量存储实例
        """
    # 判断是否存在向量库
    if os.path.exists(persist_dir) and len(persist_dir) > 0:
        try:
            temp_store = init_vector_store(embeddings, persist_dir)
            if temp_store._collection.count() > 0:
                print(f"加载已存在的有效向量存储: {persist_dir}（{temp_store._collection.count()} 个文本块）")
                return temp_store
            else:
                print(f"向量存储目录存在但为空: {persist_dir}（需传入documents才能创建新存储）")

        except Exception as e:
            print(f"向量存储损坏（{str(e)}）（需传入documents才能创建新存储）")

    # 2. 仅当documents非空时，才创建新存储（避免创建空存储）
    if documents and len(documents) > 0:
        print(f"创建新的向量存储并持久化到: {persist_dir}")
        vector_store = init_vector_store(embeddings, persist_dir)
        added_cnt = add_documents_to_store(vector_store, documents, batch_size)
        print(f"已添加 {added_cnt} 个文档进入向量库")
        return vector_store
    else:
        # 若存储不存在且无文档，直接报错（避免返回空存储）
        raise ValueError("存储为空且传入的documents为空")

