import os
from huggingface_hub import snapshot_download

# ✅ 1. 如果你在国内，加上这一行，使用国内镜像加速下载，防止超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def init_huggingface_cache(target_dir: str = "F:/huggingface", model_name: str = "BAAI/bge-reranker-v2-m3"):
    """
    下载重排序模型 BGE-Reranker 到指定目录
    """
    # 设置环境变量（只是为了指定缓存位置，如果使用了 local_dir，这个其实是辅助作用）
    os.environ["HF_HOME"] = target_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(target_dir, "hub")

    # 确保目录存在
    os.makedirs(target_dir, exist_ok=True)

    print(f"Hugging Face 缓存目录设置为：{target_dir}")
    print(f"准备下载模型：{model_name} ...")

    # 下载模型到本地
    # local_dir 指定了具体的文件夹，比如 F:/huggingface/bge-reranker-v2-m3
    save_path = os.path.join(target_dir, model_name.split('/')[-1])

    model_path = snapshot_download(
        repo_id=model_name,
        cache_dir=target_dir,
        local_dir=save_path,
        local_dir_use_symlinks=False  # ✅ 关键：设置为 False，确保下载的是真实文件而不是软链接
    )

    print(f"✅ 模型已下载到：{model_path}")

    # 4️⃣ 验证是否可加载 (注意：这里逻辑变了)
    try:
        from sentence_transformers import CrossEncoder  # ⚠️ 注意这里导入的是 CrossEncoder

        print("正在加载模型进行验证...")
        # 加载重排序模型
        model = CrossEncoder(model_path)

        # 构造一个测试对 [Query, Document]
        test_pair = ["RAG是什么？", "RAG是检索增强生成的缩写。"]

        # 预测得分
        score = model.predict(test_pair)
        print(f"✅ 模型加载成功！测试对打分：{score:.4f}")

    except Exception as e:
        print(f"❌ 模型加载验证失败: {e}")
        print("提示：请确保已安装 sentence-transformers (pip install sentence-transformers)")


if __name__ == "__main__":
    # 执行下载
    # 模型比较大（约 1-2GB），请耐心等待
    init_huggingface_cache("F:/huggingface", "BAAI/bge-reranker-v2-m3")