from core.loaders.ppt_loader import PptLoader

if __name__ == "__main__":
    ppt_path = r"D:\PythonProject\RAG-System\data\知识库资料_22033\2. 日常健康咨询\临床培训PPT\慢性阻塞性肺疾病介绍 姜然.pptx"
    loader = PptLoader(ppt_path)
    documents = loader.load()
    for document in documents:
        print(document.page_content.strip())
