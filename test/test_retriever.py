import os

from core.embeddings.nomic_embeddings import get_nomic_embeddings
from core.vector_store.vector_store_manager import create_or_load_vector_store
from dotenv import load_dotenv


def test_retriever():
    load_dotenv()
    # 1. 载入相同的 embedding 模型
    nomic_api_key = os.getenv("NOMIC_API_KEY")
    embeddings = get_nomic_embeddings(nomic_api_key)

    # 2. 加载已有向量数据库（无需重新构建）
    persist_dir = '../core/vector_store/deepseek_chroma_db'
    vector_store = create_or_load_vector_store(
        persist_dir=persist_dir,
        embeddings=embeddings,
        documents=None,
        batch_size=20
    )

    # 3. 测试检索
    query = "呼吸的主要生理过程有哪些？"
    results = vector_store.similarity_search(query, k=3)

    print("\n=== 检索结果 ===")
    for i, doc in enumerate(results, 1):
        print(f"\n🔹 Top {i}:")
        print(doc.page_content[:300] + "...")
        print("🧩 来源:", doc.metadata["source"])


if __name__ == '__main__':
    test_retriever()
