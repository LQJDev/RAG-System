from langchain_huggingface import HuggingFaceEmbeddings
def get_bge_large_zh_embeddings():
    model_path = "F:/huggingface/bge-large-zh"  # 你下载的本地模型
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    return embeddings