from core.parser.markdown_parser import MarkDownParser
from core.text_splitter import text_splitter

def test():
    parser = MarkDownParser(r"D:\PythonProject\RAG-System\data\知识库资料_22033\2. 日常健康咨询\临床培训PPT\经鼻高流量湿化氧疗临床应用.md")
    docs = parser.load()
    docs = text_splitter.split_documents(docs, 800, 100)
    for doc in docs:
        print(doc.page_content)
        print("==============================")


if __name__ == '__main__':
    test()