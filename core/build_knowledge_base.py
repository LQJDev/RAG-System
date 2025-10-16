from typing import Tuple
from langchain_chroma import Chroma
from core.vector_store.vector_store_manager import create_or_load_vector_store
from core.embeddings.nomic_embeddings import get_nomic_embeddings
from core.processor.file_processor import load_documents  # 导入优化后的加载函数
from core.text_splitter import text_splitter


def build_knowledge_base(
        source_path: str,
        persist_dir: str,
        nomic_api_key: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        batch_size: int = 20
) -> Tuple[Chroma, int]:
    # 1. 初始化嵌入模型（全局复用）
    embeddings = get_nomic_embeddings(nomic_api_key)

    # 2. 检查是否需要重新构建（通过create_or_load_vector_store间接判断，减少重复逻辑）
    # 先尝试加载现有存储，若有效则直接返回（避免重复创建temp_store）
    try:
        # 尝试加载已有存储（不传入documents，仅做检测）
        vector_store = create_or_load_vector_store(embeddings, persist_dir, documents=None, batch_size=batch_size)
        # 若成功加载且有数据，直接返回（0表示未新增文档）
        if vector_store._collection.count() > 0:
            print(f"检测到有效向量存储（{vector_store._collection.count()} 个文本块），直接使用...")
            return vector_store, 0
    except Exception:
        # 存储无效或不存在，进入构建流程
        print("未检测到有效向量存储，开始构建...")

    # 3. 加载并处理文档（仅当需要构建时执行）
    print("开始加载文档...")
    documents = load_documents(source_path)
    if not documents:
        print("未加载到任何文档内容")
        return None, 0

    # 文本分割
    print(f"开始分割 {len(documents)} 个文档...")
    documents = text_splitter.split_documents(documents, chunk_size, chunk_overlap)
    print(f"文本分割完成，得到 {len(documents)} 个文本块")

    # 4. 创建新向量存储并添加文档
    vector_store = create_or_load_vector_store(
        embeddings=embeddings,
        persist_dir=persist_dir,
        documents=documents,
        batch_size=batch_size
    )

    return vector_store, len(documents)


# 测试
if __name__ == "__main__":
    nomic_api_key = "nk-wi3IFnraTtjA8U_Uaiby-nSGowbWoYKwlVT15UKcoOY"
    source_path = "../data/知识库资料_22033"  # 文件夹/单个文件都支持
    persist_dir = "vector_store/deepseek_chroma_db"

    store, count = build_knowledge_base(
        source_path=source_path,
        persist_dir=persist_dir,
        nomic_api_key=nomic_api_key,
        chunk_size=800,
        chunk_overlap=80,
        batch_size=20
    )
    print(f"知识库构建完成，共添加 {count} 个文本块")