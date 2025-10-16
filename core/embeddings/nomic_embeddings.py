import os
from langchain_nomic.embeddings import NomicEmbeddings  # 从官方专用库导入


def get_nomic_embeddings(api_key: str = None) -> NomicEmbeddings:
    """
    使用 langchain_nomic 中的官方现成模型
    这是 LangChain 推荐的 Nomic 集成方式，适配最新版本
    """
    # 配置 API 密钥（支持参数传入或环境变量）
    api_key = api_key or os.getenv("NOMIC_API_KEY")
    if not api_key:
        raise ValueError("请提供 Nomic API 密钥（参数或 NOMIC_API_KEY 环境变量）")

    # 初始化官方现成模型（一行完成，无需自定义逻辑）
    embeddings = NomicEmbeddings(
        nomic_api_key=api_key,
        model="nomic-embed-text-v1",  # 官方指定模型名
        dimensionality=768  # 固定输出维度
    )

    # 验证模型可用性
    try:
        test_vector = embeddings.embed_query("验证现成模型是否可用")
        assert len(test_vector) == 768, f"向量维度错误，实际：{len(test_vector)}"
        print("✅ langchain_nomic 官方模型初始化成功")
        return embeddings
    except Exception as e:
        raise RuntimeError(f"模型验证失败：{str(e)}") from e


# 测试代码
if __name__ == "__main__":
    # 配置 API 密钥（二选一）
    # os.environ["NOMIC_API_KEY"] = "你的密钥"  # 方式1：环境变量
    nomic_embeds = get_nomic_embeddings(api_key="nk-wi3IFnraTtjA8U_Uaiby-nSGowbWoYKwlVT15UKcoOY")  # 方式2：直接传入

    # 测试批量文档嵌入
    docs = ["这是测试文档1", "这是测试文档2"]
    doc_vectors = nomic_embeds.embed_documents(docs)
    print(f"批量嵌入数量：{len(doc_vectors)}，维度：{len(doc_vectors[0])}")

    # 测试查询嵌入
    query = "测试查询"
    query_vector = nomic_embeds.embed_query(query)
    print(f"查询向量维度：{len(query_vector)}，前5值：{query_vector[:5]}...")