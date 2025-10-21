# semantic_split_test.py
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

def test_semantic_chunking():
    # ===== 1️⃣ 初始化中文 bge 向量模型 =====
    print("🚀 正在加载中文向量模型 BAAI/bge-large-zh ...")
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh")

    # ===== 2️⃣ 创建语义分割器 =====
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"  # "standard_deviation" 也可以
    )

    # ===== 3️⃣ 准备一段长文本 =====
    text = """
患者因呼吸衰竭入住重症监护室，开始接受高流量鼻导管氧疗（HFNC）。
经过两天治疗后，病情有所好转，呼吸频率下降，血氧饱和度稳定在95%以上。
医生考虑逐步撤机，并评估患者自主呼吸能力。
撤机过程中需密切监测患者呼吸参数及血气变化，如出现呼吸窘迫或二氧化碳潴留需重新上机。
患者家属对治疗方案表示理解与支持。
"""

    # ===== 4️⃣ 语义分割 =====
    docs = splitter.create_documents([text])
    print(f"\n📄 共分割出 {len(docs)} 个语义片段：\n")

    # ===== 5️⃣ 打印分割结果 =====
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip().replace("\n", "")
        print(f"--- 段落 {i}（长度 {len(content)}）---")
        print(content)
        print()

if __name__ == "__main__":
    test_semantic_chunking()
