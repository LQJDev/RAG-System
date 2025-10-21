from typing import Tuple
from langchain_chroma import Chroma
from core.vector_store.vector_store_manager import create_or_load_vector_store
from core.embeddings.nomic_embeddings import get_nomic_embeddings
from core.processor.file_processor import load_documents  # 导入优化后的加载函数
from core.text_splitter import text_splitter
from core.embeddings.bge_large_zh_embeddings import get_bge_large_zh_embeddings

def build_knowledge_base(
        source_path: str,
        persist_dir: str,
        name: str,  # ✅ 新增知识库名称
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        batch_size: int = 20
) -> Tuple[Chroma, int]:
    # 1. 初始化嵌入模型（全局复用）
    embeddings = get_bge_large_zh_embeddings()

    # 2. 检查是否需要重新构建
    try:
        vector_store = create_or_load_vector_store(
            embeddings,
            persist_dir,
            documents=None,
            batch_size=batch_size,
            collection_name=name  # ✅ 指定知识库名称
        )
        if vector_store._collection.count() > 0:
            print(f"检测到有效向量存储（{vector_store._collection.count()} 个文本块），直接使用...")
            return vector_store, 0
    except Exception:
        print("未检测到有效向量存储，开始构建...")

    # 3. 加载文档
    print("开始加载文档...")
    documents = load_documents(source_path)
    if not documents:
        print("未加载到任何文档内容")
        return None, 0

    # 4. 文本分割
    print(f"开始分割 {len(documents)} 个文档...")
    documents = text_splitter.split_documents(documents, chunk_size, chunk_overlap)
    print(f"文本分割完成，得到 {len(documents)} 个文本块")

    # 5. 创建向量存储
    vector_store = create_or_load_vector_store(
        embeddings=embeddings,
        persist_dir=persist_dir,
        documents=documents,
        batch_size=batch_size,
        name=name  # ✅ 指定知识库名称
    )

    return vector_store, len(documents)


def append_to_knowledge_base(
        source_path: str,
        persist_dir: str,
        name: str,  # 已存在知识库名称
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        batch_size: int = 20
) -> Tuple[Chroma, int]:
    """
    向已有知识库追加文档
    Returns:
        vector_store: Chroma 实例
        added_count: 新增的文档块数
    """
    # 1️⃣ 嵌入模型
    embeddings = get_bge_large_zh_embeddings()

    # 2️⃣ 加载已有知识库
    try:
        vector_store = create_or_load_vector_store(
            embeddings=embeddings,
            persist_dir=persist_dir,
            documents=None,
            batch_size=batch_size,
            name=name
        )
        print(f"已加载知识库 '{name}'（{vector_store._collection.count()} 个文本块）")
    except Exception as e:
        raise RuntimeError(f"加载知识库 '{name}' 失败: {e}")

    # 3️⃣ 加载新文档
    print(f"开始加载新文档: {source_path}")
    new_documents = load_documents(source_path)
    if not new_documents:
        print("未加载到任何新文档")
        return vector_store, 0

    # 4️⃣ 文本分割
    print(f"开始分割 {len(new_documents)} 个新文档...")
    new_documents = text_splitter.split_documents(new_documents, chunk_size, chunk_overlap)
    print(f"文本分割完成，得到 {len(new_documents)} 个新文本块")

    # 5️⃣ 添加到向量库
    added_count = vector_store.add_documents(new_documents)
    print(f"🎯 已向知识库 '{name}' 添加 {added_count} 个新文档块")

    return vector_store, added_count




# 测试
if __name__ == "__main__":
    source_path = "../data/知识库资料_22033/2. 日常健康咨询"
    persist_dir = "vector_store/docling_chroma_db"
    kb_name = "daily_health_consult"  # ✅ 知识库名称

    store, count = build_knowledge_base(
        source_path=source_path,
        persist_dir=persist_dir,
        name=kb_name,
        chunk_size=800,
        chunk_overlap=80,
        batch_size=20
    )
    print(f"知识库构建完成，共添加 {count} 个文本块")
