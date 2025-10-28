from pathlib import Path
import re


def load_and_split_markdown(file_path: str):
    """
    加载 Markdown 文件，并按标题层级（###）进行语义分块。
    最小粒度为 ### 小节。
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    print(f"📄 开始加载 Markdown 文件：{file_path.name}")

    # 1️⃣ 读取 Markdown 文本
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2️⃣ 去除多余空行
    text = re.sub(r"\n{2,}", "\n\n", text.strip())

    # 3️⃣ 正则分块，以 ### 为最小单位（保留标题）
    # 使用非贪婪匹配：匹配从 ### 开始到下一个 ### 或文件末尾
    pattern = r"(### .+?)(?=\n### |\Z)"
    matches = re.findall(pattern, text, flags=re.S)

    # 如果没有 ###，尝试按 ## 分
    if not matches:
        pattern = r"(## .+?)(?=\n## |\Z)"
        matches = re.findall(pattern, text, flags=re.S)

    print(f"🧩 共生成 {len(matches)} 个分块\n")

    # 4️⃣ 打印示例
    for i, block in enumerate(matches[:10]):  # 仅预览前 10 块
        print(f"🔹 分块 {i + 1}")
        print("-" * 40)
        print(block.strip())
        print("=" * 60 + "\n")

    return matches


if __name__ == "__main__":
    md_path = r"D:\PythonProject\RAG-System\data\知识库资料_22033\2. 日常健康咨询\临床培训PPT\OSA疾病介绍.md"
    split_docs = load_and_split_markdown(md_path)
