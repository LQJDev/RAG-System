from core.parser.pdf_parser import PdfLoader



if __name__ == "__main__":
    loader = PdfLoader(r"D:\PythonProject\RAG-System\data\知识库资料_22033\2. 日常健康咨询\睡眠呼吸相关文献\2024年 睡眠呼吸障碍年度进展2023.pdf")
    docs = loader.load()
    for doc in docs:
        print(doc.page_content[:500])