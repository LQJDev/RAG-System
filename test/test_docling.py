from langchain_docling import DoclingLoader
from core.parser.docling_parser import DoclingParser
# ---------------------------
# PPTX 文件路径
# ---------------------------
pptx_file = r"D:\PythonProject\RAG-System\data\知识库资料_22033\2. 日常健康咨询\睡眠呼吸相关文献\2024年 老年阻塞性睡眠呼吸暂停患者无创正压通气应用规范专家共识.pdf"

# ---------------------------
# 初始化 DoclingLoader
# ---------------------------
loader = DoclingParser(file_path=[pptx_file])

# ---------------------------
# 加载 PPTX 文档
# ---------------------------
docs = loader.load()

# ---------------------------
# 打印每页摘要
# ---------------------------
print(f"\n✅ 共解析 {len(docs)} 页文档：\n")

for i, doc in enumerate(docs, 1):
    # 只显示前 300 个字符作为预览
    text_preview = doc.page_content[:300].replace("\n", " ")
    print("========================content===========================")
    print(text_preview)

