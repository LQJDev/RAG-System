from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from typing import List, Optional
import os
import pdfplumber



class PdfLoader(BaseLoader):
    """加载PDF文件内容的加载器"""

    def __init__(self, file_path: str, extract_images: bool = False):
        """
        初始化PDF加载器

        Args:
            file_path: PDF文件的路径
            extract_images: 是否提取图片（目前未实现）
        """
        self.file_path = file_path
        self.extract_images = extract_images
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"不支持的文件格式，需要.pdf: {file_path}")

    def load(self) -> List[Document]:
        """
        加载并解析PDF文件内容

        Returns:
            包含PDF各页内容的Document对象列表
        """
        documents = []
        try:
            # 用pdfplumber打开PDF（支持部分损坏的文件）
            with pdfplumber.open(self.file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # 提取页面文字（保留格式，支持复杂排版）
                    text = page.extract_text() or ""
                    if text.strip():  # 跳过空白页
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": self.file_path,
                                "page": page_num,
                                "file_type": "pdf"
                            }
                        )
                        documents.append(doc)
            print(f"成功解析PDF: {self.file_path}（{len(documents)}页）")
        except Exception as e:
            # 兜底：如果pdfplumber失败，尝试用unstructured的通用解析
            try:
                from unstructured.partition.pdf import partition_pdf
                elements = partition_pdf(filename=self.file_path)
                text = "\n".join([str(el) for el in elements if el.text.strip()])
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={"source": self.file_path, "file_type": "pdf"}
                    )
                    documents.append(doc)
                print(f"用unstructured兜底解析PDF: {self.file_path}")
            except Exception as e2:
                raise Exception(f"解析PDF文件失败: {str(e2)}") from e
        return documents
