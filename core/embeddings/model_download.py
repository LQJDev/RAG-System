import os
from huggingface_hub import snapshot_download

def init_huggingface_cache(target_dir: str = "F:/huggingface", model_name: str = "BAAI/bge-large-zh"):
    """
    初始化 Hugging Face 环境到指定目录，并下载模型。

    Args:
        target_dir (str): 模型与缓存保存的根目录
        model_name (str): 要下载的模型名称
    """
    # 设置环境变量（在当前运行环境中生效）
    os.environ["HF_HOME"] = target_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(target_dir, "hub")

    # 确保目录存在
    os.makedirs(target_dir, exist_ok=True)

    print(f"Hugging Face 缓存目录设置为：{target_dir}")
    print(f"准备下载模型：{model_name} ...")

    # 下载模型到本地（snapshot_download 可断点续传）
    model_path = snapshot_download(
        repo_id=model_name,
        cache_dir=target_dir,
        local_dir=os.path.join(target_dir, model_name.split('/')[-1]),
        local_dir_use_symlinks=False
    )

    print(f"模型已下载到：{model_path}")

    # 4️⃣ 验证是否可加载
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path)
        emb = model.encode("你好，世界！")
        print(f"模型加载成功，向量维度：{len(emb)}")
    except Exception as e:
        print(f"模型加载验证失败: {e}")

if __name__ == "__main__":
    init_huggingface_cache("F:/huggingface", "BAAI/bge-large-zh")
