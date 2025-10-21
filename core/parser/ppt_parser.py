import os
import io
import re
from typing import List
from pptx import Presentation
from langchain.schema import Document
from core.parser.base_parser import BaseParser
from PIL import Image, ImageOps
import pytesseract

# 配置 tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r"F:\Program Files\Tesseract-OCR\tesseract.exe"

class PptParser(BaseParser):
    """智能 PPT 加载器，支持文本+图片 OCR，并过滤无文字图片"""

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if not file_path.lower().endswith('.pptx'):
            raise ValueError(f"不支持的文件格式，需要.pptx: {file_path}")
        self.file_path = file_path

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'^\s*[\d一二三四五六七]+\s*[、\.]?\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'\s{2,}', " ", text)
        return text.strip()

    import re

    def _clean_ocr_text(text: str) -> str:
        # 去掉连续空行
        text = re.sub(r'\n{2,}', '\n\n', text)
        # 合并句内换行
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        # 去掉多余空格
        text = re.sub(r'\s{2,}', ' ', text)
        # 统一项目符号
        text = re.sub(r'[*•\-]+', '-', text)
        return text.strip()

    def _ocr_image(self, image: Image.Image, min_length: int = 5) -> str:
        """对 PIL Image 执行 OCR，并过滤无效文字"""
        try:
            # 灰度化 + 二值化
            image = ImageOps.grayscale(image)
            text = pytesseract.image_to_string(image, lang="chi_sim+eng").strip()
            text_clean = self._clean_ocr_text(text)
            # # 过滤长度过短或全非文字的结果
            # text_filtered = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5]", "", text_clean)
            if len(text_clean) >= min_length:
                return text_clean
            return ""
        except Exception:
            return ""

    def load(self) -> List[Document]:
        try:
            prs = Presentation(self.file_path)
            docs = []

            for slide_idx, slide in enumerate(prs.slides, 1):
                slide_texts = []

                # 提取文本
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_texts.append(shape.text)

                # 图片 OCR
                for shape in slide.shapes:
                    if hasattr(shape, "image") and shape.image:
                        image_bytes = shape.image.blob
                        image = Image.open(io.BytesIO(image_bytes))
                        ocr_text = self._ocr_image(image)
                        if ocr_text:
                            slide_texts.append(ocr_text)

                if slide_texts:
                    content = "\n\n".join(slide_texts)
                    content = self._clean_text(content)
                    method = "text+ocr" if any(hasattr(s, "image") and s.image for s in slide.shapes) else "text"
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": self.file_path,
                            "slide_number": slide_idx,
                            "file_type": "pptx",
                            "method": method
                        }
                    ))

            return docs
        except Exception as e:
            raise RuntimeError(f"解析PPT文件失败: {str(e)}") from e
