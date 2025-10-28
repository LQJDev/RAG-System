from core.parser.base_parser import BaseParser
from langchain.schema import Document
from typing import List
import os
from pathlib import Path
import re

class MarkDownParser(BaseParser):
    """加载MD文件内容的加载器"""

    def __init__(self, file_path: str):
        """
        初始化DOCX加载器

        Args:
            file_path: DOCX文件的路径
        """
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if not file_path.lower().endswith('.md'):
            raise ValueError(f"不支持的文件格式，需要.md: {file_path}")

    def load(self) -> List[Document]:
        """
        将 Markdown 按 ### 分块，并生成 LangChain Document 对象列表
        """
        text = Path(self.file_path).read_text(encoding="utf-8")

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
                    "source": self.file_path
                }
            )
            documents.append(doc)

        print(f"🧩 共生成 {len(documents)} 个 Document")
        return documents
