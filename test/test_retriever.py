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


def test5():
    embeddings = get_bge_large_zh_embeddings()
    # 2. 加载已有向量数据库（无需重新构建）
    persist_dir = '../core/vector_store/md_chroma_db'
    vector_store = create_or_load_vector_store(
        persist_dir=persist_dir,
        embeddings=embeddings,
        documents=None,
        batch_size=20,
        name="voice_encourage_info"
    )

    # 3. 测试检索
    query = "COPD鼓励语言的周/月报模板"
    results = vector_store.similarity_search(query, k=3)

    print("\n=== 检索结果 ===")
    for i, doc in enumerate(results, 1):
        print(f"\n🔹 Top {i}:")
        print(doc.page_content)


def test6():
    embeddings = get_bge_large_zh_embeddings()

    # 2️⃣ 加载已有向量数据库
    persist_dir = '../core/vector_store/md_chroma_db'
    vector_store = create_or_load_vector_store(
        persist_dir=persist_dir,
        embeddings=embeddings,
        documents=None,
        batch_size=20,
        name="daily_use_guidance"
    )

    # 3️⃣ 测试检索
    query = "带上面罩恐惧怎么办"

    # ✅ 使用带分数的方法
    results = vector_store.similarity_search_with_score(query, k=3)

    print("\n=== 检索结果（含相似度） ===")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n🔹 Top {i}: (相似度分数: {score:.4f})")
        print(f"模块: {doc.metadata.get('module', '未知')}")
        print(f"标题: {doc.metadata.get('title', '无')}")
        print("内容片段:")
        print(doc.page_content[:300], "...\n")



if __name__ == '__main__':
    test5()
