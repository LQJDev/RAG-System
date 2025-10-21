from core.parser.docx_parser import DocxLoader


if __name__ == "__main__":
    loader = DocxLoader(r"D:\PythonProject\RAG-System\data\知识库资料_22033\2. 日常健康咨询\临床培训PPT\睡眠呼吸暂停科普  姜然.docx")
    docs = loader.load()
    for doc in docs:
        print(doc.page_content[:500])