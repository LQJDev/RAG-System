import os
import io
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
from langchain.schema import Document
from core.parser.base_parser import BaseParser
from typing import List
import re

# 配置 tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r"F:\Program Files\Tesseract-OCR\tesseract.exe"


class PdfParser(BaseParser):
    """智能 PDF 加载器：自动判断是否需要 OCR"""

    def __init__(self, file_path: str, use_ocr_threshold: float = 0.3):
        """
        Args:
            file_path: PDF 文件路径
            use_ocr_threshold: 文本比例低于此值时使用 OCR（0~1）
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("仅支持 PDF 文件")

        self.file_path = file_path
        self.use_ocr_threshold = use_ocr_threshold

    def _analyze_text_ratio(self) -> float:
        """快速判断 PDF 中文本比例，用 pdfplumber 统计前几页的文字含量"""
        text_chars = 0
        total_pages = 0
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for i, page in enumerate(pdf.pages[:3]):  # 仅分析前 3 页
                    text = page.extract_text() or ""
                    text_chars += len(text.strip())
                    total_pages += 1
            avg_text = text_chars / max(total_pages, 1)
            # 粗略判断：每页文字太少可能是扫描件
            return avg_text / 1000.0  # 归一化比例
        except Exception:
            return 0.0

    def _load_with_pdfplumber(self) -> List[Document]:
        """普通 PDF 文本提取"""
        docs = []
        with pdfplumber.open(self.file_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                text = self._clean_text(text)
                if text.strip():
                    docs.append(Document(
                        page_content=text.strip(),
                        metadata={"page": i, "source": self.file_path, "method": "pdfplumber"}
                    ))
        return docs

    def _load_with_ocr(self) -> List[Document]:
        """OCR 方式提取（适合扫描图片 PDF）"""
        docs = []
        with fitz.open(self.file_path) as pdf:
            for i, page in enumerate(pdf, 1):
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, lang="chi_sim+eng")
                text = self._clean_text(text)
                if text.strip():
                    docs.append(Document(
                        page_content=text.strip(),
                        metadata={"page": i, "source": self.file_path, "method": "ocr"}
                    ))
        return docs

    def _clean_text(self, text: str) -> str:
        # 去掉页眉页脚、特殊符号
        text = re.sub(r"第\s*\d+\s*页.*?(共\s*\d+\s*页)?", "", text)
        text = re.sub(r"Page\s*\d+.*", "", text)
        text = text.replace("\t", " ").replace("\r", "")

        # 把连续空行压缩为一行，而不是全部去掉
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 保留句内的换行（用空格替换），但段落之间保留空行
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        text = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", text)
        # 去除多余空格
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def load(self) -> List[Document]:
        """自动选择最佳方式解析 PDF"""
        ratio = self._analyze_text_ratio()
        print(f"📊 文本比例估计: {ratio:.2f}")

        if ratio > self.use_ocr_threshold:
            print("🧩 检测到文本型 PDF，使用 pdfplumber 解析")
            docs = self._load_with_pdfplumber()
        else:
            print("🧠 检测到扫描型 PDF，使用 OCR 模式解析")
            docs = self._load_with_ocr()

        print(f"✅ 解析完成，共 {len(docs)} 页（模式：{docs[0].metadata['method'] if docs else '未知'}）")
        return docs


