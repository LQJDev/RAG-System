from core.parser.ppt_parser import PptLoader
from core.text_splitter.semantic_chunking import semantic_chunking
from core.text_splitter.text_splitter import split_documents
if __name__ == "__main__":
    ppt_path = r"D:\PythonProject\RAG-System\data\知识库资料_22033\2. 日常健康咨询\临床培训PPT\气道湿化治疗Micomme 姜然.pptx"
    loader = PptLoader(ppt_path)
    documents = loader.load()
    documents = split_documents(documents)
    for document in documents:
        print(document.page_content.strip())
        print("===========================")
