import os
from langchain_nomic.embeddings import NomicEmbeddings


def get_nomic_embeddings(api_key: str = None) -> NomicEmbeddings:
    """
    初始化 Nomic 官方嵌入模型（langchain_nomic 集成版）
    自动检测 API Key、模型可用性，并输出图标提示。
    """
    print("🔧 正在初始化 Nomic Embeddings...")

    # 检查 API Key
    api_key = os.getenv("NOMIC_API_KEY")
    if not api_key:
        print("❌ 未检测到 NOMIC_API_KEY 环境变量或传入参数")
        raise ValueError("请提供 Nomic API 密钥（参数或 NOMIC_API_KEY 环境变量）")
    print("🔑 已检测到 Nomic API Key")

    # 创建模型实例
    try:
        embeddings = NomicEmbeddings(
            nomic_api_key=api_key,
            model="nomic-embed-text-v1",
            dimensionality=768
        )
        print("🧠 成功实例化 Nomic Embedding 模型")
    except Exception as e:
        print("💥 模型实例化失败！")
        raise RuntimeError(f"初始化失败：{str(e)}") from e

    # 验证模型可用性
    try:
        print("🧪 正在验证模型可用性...")
        test_vector = embeddings.embed_query("验证现成模型是否可用")
        if len(test_vector) == 768:
            print("✅ 模型验证通过，输出维度：768")
        else:
            print(f"⚠️ 向量维度异常：{len(test_vector)}（预期 768）")
        print("🌟 Nomic Embeddings 已准备就绪！")
        return embeddings
    except Exception as e:
        print("🚨 模型验证失败！")
        raise RuntimeError(f"模型验证失败：{str(e)}") from e

# 测试代码
if __name__ == "__main__":

    # 方式1：环境变量
    nomic_embeds = get_nomic_embeddings(api_key="nk-wi3IFnraTtjA8U_Uaiby-nSGowbWoYKwlVT15UKcoOY")  # 方式2：直接传入

    # 测试批量文档嵌入
    docs = ["这是测试文档1", "这是测试文档2"]
    doc_vectors = nomic_embeds.embed_documents(docs)
    print(f"批量嵌入数量：{len(doc_vectors)}，维度：{len(doc_vectors[0])}")

    # 测试查询嵌入
    query = "测试查询"
    query_vector = nomic_embeds.embed_query(query)
    print(f"查询向量维度：{len(query_vector)}，前5值：{query_vector[:5]}...")