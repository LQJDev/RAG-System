from core.loaders.pdf_loader import PdfLoader



if __name__ == "__main__":
    loader = PdfLoader("D:\\PythonProject\\RAG-System\\data\\知识库资料_22033\\说明书\\0401060259 家用湿化新说明书 V2.4.pdf")
    docs = loader.load()
    for doc in docs:
        print(doc.page_content[:500])