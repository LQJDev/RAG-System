import os.path
from typing import List
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor, as_completed


def init_vector_store(embeddings: Embeddings, persist_dir: str, name: str = None) -> Chroma:
    """初始化向量存储，支持加载已有持久化数据"""
    if not name:
        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    else:
        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir,
            collection_name=name
        )


def add_documents_to_store(
    vector_store: Chroma,
    documents: List[Document],
    batch_size: int = 32,
    max_workers: int = 4
) -> int:
    """
    使用多线程方式批量添加文档到向量存储
    Args:
        vector_store: 已初始化的 Chroma 实例
        documents: 要添加的文档列表
        batch_size: 每批次文档数
        max_workers: 最大并发线程数
    Returns:
        成功添加的文档总数
    """
    if not documents:
        return 0

    total_added = 0
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

    def add_batch(batch, start_idx):
        """单批次添加任务"""
        try:
            vector_store.add_documents(batch)
            print(f"✅ 批次 {start_idx}~{start_idx + len(batch) - 1} 添加成功")
            return len(batch)
        except Exception as e:
            print(f"⚠️ 批次 {start_idx} 失败: {e}")
            return 0

    # 使用多线程并发添加
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(add_batch, batch, i * batch_size): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(future_to_idx):
            total_added += future.result()

    print(f"🎯 全部完成，共添加 {total_added} 个文档。")
    return total_added


def create_or_load_vector_store(
        embeddings: Embeddings,
        persist_dir: str,
        documents: List[Document],
        batch_size: int,
        name: str = None
) -> Chroma:
    """
    加载或创建向量存储，支持同目录多个知识库
    """
    # 初始化向量存储实例（指定 collection_name）
    vector_store = init_vector_store(embeddings, persist_dir, name)

    try:
        # 检查当前 collection 是否已有文档
        if vector_store._collection.count() > 0:
            print(f"✅ 已加载知识库 '{name}'（{vector_store._collection.count()} 个文本块）")
            return vector_store
    except Exception:
        print(f"⚠️ 知识库 '{name}' 不存在或损坏，将重新创建")

    # 仅当 documents 非空时才添加
    if documents and len(documents) > 0:
        added_cnt = add_documents_to_store(vector_store, documents, batch_size)
        print(f"🎯 知识库 '{name}' 构建完成，共添加 {added_cnt} 个文档")
        return vector_store
    else:
        raise ValueError(f"知识库 '{name}' 不存在且没有传入 documents，无法创建")

