import os

from core.embeddings.nomic_embeddings import get_nomic_embeddings
from core.vector_store.vector_store_manager import create_or_load_vector_store
from core.embeddings.bge_large_zh_embeddings import get_bge_large_zh_embeddings


def test1():
    embeddings = get_bge_large_zh_embeddings()
    # 2. 加载已有向量数据库（无需重新构建）
    persist_dir = '../core/vector_store/docling_chroma_db'
    vector_store = create_or_load_vector_store(
        persist_dir=persist_dir,
        embeddings=embeddings,
        documents=None,
        batch_size=20,
        name="warning_explain"
    )

    # 3. 测试检索
    query = "硬件故障类报警如"
    results = vector_store.similarity_search(query, k=3)

    print("\n=== 检索结果 ===")
    for i, doc in enumerate(results, 1):
        print(f"\n🔹 Top {i}:")
        print(doc.page_content[:300] + "...")
        print("🧩 来源:", doc.metadata["source"])



def test2():
    embeddings = get_bge_large_zh_embeddings()
    # 2. 加载已有向量数据库（无需重新构建）
    persist_dir = '../core/vector_store/docling_chroma_db'
    vector_store = create_or_load_vector_store(
        persist_dir=persist_dir,
        embeddings=embeddings,
        documents=None,
        batch_size=20,
        name="machine_maintenance"
    )

    # 3. 测试检索
    query = "清洁加湿器水罐"
    results = vector_store.similarity_search(query, k=3)

    print("\n=== 检索结果 ===")
    for i, doc in enumerate(results, 1):
        print(f"\n🔹 Top {i}:")
        print(doc.page_content[:300] + "...")
        print("🧩 来源:", doc.metadata["source"])


def test3():
    embeddings = get_bge_large_zh_embeddings()
    # 2. 加载已有向量数据库（无需重新构建）
    persist_dir = '../core/vector_store/docling_chroma_db'
    vector_store = create_or_load_vector_store(
        persist_dir=persist_dir,
        embeddings=embeddings,
        documents=None,
        batch_size=20,
        name="breath_report_explain"
    )

    # 3. 测试检索
    query = "治疗时长"
    results = vector_store.similarity_search(query, k=3)

    print("\n=== 检索结果 ===")
    for i, doc in enumerate(results, 1):
        print(f"\n🔹 Top {i}:")
        print(doc.page_content[:300] + "...")
        print("🧩 来源:", doc.metadata["source"])


def test4():
    embeddings = get_bge_large_zh_embeddings()
    # 2. 加载已有向量数据库（无需重新构建）
    persist_dir = '../core/vector_store/docling_chroma_db'
    vector_store = create_or_load_vector_store(
        persist_dir=persist_dir,
        embeddings=embeddings,
        documents=None,
        batch_size=20,
        name="first_use_guidance"
    )

    # 3. 测试检索
    query = "放置位置"
    results = vector_store.similarity_search(query, k=3)

    print("\n=== 检索结果 ===")
    for i, doc in enumerate(results, 1):
        print(f"\n🔹 Top {i}:")
        print(doc.page_content[:300] + "...")
        print("🧩 来源:", doc.metadata["source"])


if __name__ == '__main__':
    test4()
