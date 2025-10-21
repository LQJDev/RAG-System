from langchain.schema import Document
from typing import List
import os
from langchain_docling import DoclingLoader
from core.parser.base_parser import BaseParser

class DoclingParser(BaseParser):
    """
    安全增强版 DoclingLoader：
    1. 自动清理复杂 metadata
    2. 提炼简洁元信息（与 DocxLoader 风格统一）
    """

    def __init__(self, file_path: str | list[str]):
        """
        初始化安全版 Docling 加载器
        """
        super().__init__(file_path=file_path)
        if isinstance(file_path, str):
            self.file_paths = [file_path]
        else:
            self.file_paths = file_path

    def load(self) -> List[Document]:
        """
        加载文件并提炼干净的 Document 对象
        """
        all_docs = []
        for path in self.file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在: {path}")

            try:
                loader = DoclingLoader(file_path=self.file_paths)
                docs = loader.load()
                for d in docs:
                    # 提取主要内容
                    if isinstance(d, str):
                        content = d
                        meta = {}
                    elif isinstance(d, Document):
                        content = d.page_content
                        meta = d.metadata
                    elif isinstance(d, dict):
                        content = d.get("page_content", "") or d.get("text", "")
                        meta = d.get("metadata", {})
                    else:
                        content = str(d)
                        meta = {}

                    # -------------------------------
                    # ✅ 精简 metadata
                    # -------------------------------
                    clean_meta = {
                        "source": path,
                        "file_name": os.path.basename(path),
                        "file_type": os.path.splitext(path)[1].replace(".", "").lower(),
                        "origin_mime": meta.get("origin", {}).get("mimetype") if isinstance(meta, dict) else None,
                        "page_count": self._extract_page_count(meta),
                        "headings": self._extract_headings(meta),
                    }

                    all_docs.append(Document(page_content=content, metadata=clean_meta))

            except Exception as e:
                raise RuntimeError(f"解析文件失败: {path}, 错误: {str(e)}") from e

        return all_docs

    # 提取页数（若 Docling 元数据中存在）
    def _extract_page_count(self, meta):
        try:
            if isinstance(meta, dict):
                doc_items = meta.get("doc_items")
                if isinstance(doc_items, list):
                    pages = {prov["page_no"] for item in doc_items for prov in item.get("prov", []) if "page_no" in prov}
                    return len(pages)
        except Exception:
            pass
        return None

    # 提取标题信息（若存在）
    def _extract_headings(self, meta):
        try:
            if isinstance(meta, dict):
                headings = meta.get("headings")
                if isinstance(headings, list):
                    return ", ".join(headings)
        except Exception:
            pass
        return None
