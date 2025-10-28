import re
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from core.embeddings.bge_large_zh_embeddings import get_bge_large_zh_embeddings

def split_markdown_to_documents(file_path: str):
    """
    将 Markdown 按 ### 分块，并生成 LangChain Document 对象列表
    """
    text = Path(file_path).read_text(encoding="utf-8")

    # 1️⃣ 提取模块（## 模块X）
    module_pattern = re.compile(r"## 模块(\d+?)：(.+)")
    module_positions = [(m.start(), f"模块{m.group(1)}：{m.group(2).strip()}")
                        for m in module_pattern.finditer(text)]

    # 2️⃣ 按三级、四级标题分块
    pattern = re.compile(r"(?=\n### |\n#### )")
    segments = pattern.split(text)

    results = []
    for seg in segments:
        seg = seg.strip()
        if not seg.startswith("###"):
            continue

        # 去掉标题序号
        content = re.sub(r"(#{3,4}) \d+(?:\.\d+)* ", r"\1 ", seg.strip())

        # 提取标题（去掉序号）
        title_match = re.match(r"(#{3,4}) (.+)", content.splitlines()[0])
        if not title_match:
            continue

        level = len(title_match.group(1))
        title = title_match.group(2).strip()

        # 查找所属模块（根据位置）
        pos = text.find(seg)
        module_name = "未知模块"
        for i, (m_start, m_name) in enumerate(module_positions):
            if i + 1 < len(module_positions):
                next_start = module_positions[i + 1][0]
                if m_start <= pos < next_start:
                    module_name = m_name
                    break
            else:
                module_name = m_name

        results.append({
            "level": level,
            "module": module_name,
            "title": title,
            "content": content
        })

    # 3️⃣ 四级标题合并到上级三级标题块
    merged = []
    current = None
    for r in results:
        if r["level"] == 3:
            if current:
                merged.append(current)
            current = {
                "module": r["module"],
                "title": r["title"],
                "content": r["content"]
            }
        elif r["level"] == 4 and current:
            current["content"] += "\n" + r["content"]

    if current:
        merged.append(current)

    # 4️⃣ 转为 LangChain Document 对象
    documents = []
    for m in merged:
        doc = Document(
            page_content=m["content"],
            metadata={
                "module": m["module"],
                "title": m["title"],
                "source": file_path
            }
        )
        documents.append(doc)

    print(f"🧩 共生成 {len(documents)} 个 Document")
    return documents


def test():

    # 1️⃣ 初始化 embedding（需和入库时一致 ）
    embeddings = get_bge_large_zh_embeddings()

    # 2️⃣ 加载已存的 Chroma 向量库
    vector_store = Chroma(
        persist_directory="./respiration_chroma",
        embedding_function=embeddings
    )

    # 3️⃣ 测试查询
    query = "COPD诊断标准与严重程度分级"
    results = vector_store.similarity_search(query, k=3)  # 返回最相似的3个文档

    # 4️⃣ 打印检索结果
    for i, doc in enumerate(results, 1):
        print(f"🔹 第{i}条匹配")
        # print(f"模块: {doc.metadata['module']}, 小节: {doc.metadata['num']}, 标题: {doc.metadata['title']}")
        print(doc.page_content)
        print("=" * 80)

def check_doc():
    md_file = r"D:\PythonProject\RAG-System\data\知识库资料_22033\2. 日常健康咨询\明康产品资料\彩页\正压通气面罩CF-10A 彩页.md"
    docs = split_markdown_to_documents(md_file)
    for doc in docs:
        print(doc.page_content)
        print("================")

def save_store():
    md_file = r"D:\PythonProject\RAG-System\data\知识库资料_22033\6. 初次使用语音指导\初次使用语音指导.md"
    #
    # # 1️⃣ Markdown → Document 列表
    docs = split_markdown_to_documents(md_file)

    # 2️⃣ 初始化 embedding
    embeddings = get_bge_large_zh_embeddings()

    # 3️⃣ 入 Chroma 向量库
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./respiration_chroma"
    )
    print("✅ 已完成向量化并入库")

if __name__ == "__main__":
    check_doc()
    # save_store()
