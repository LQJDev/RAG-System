import os
from core.loaders.pdf_loader import PdfLoader
from core.loaders.ppt_loader import PptLoader
from core.loaders.docx_loader import DocxLoader
from typing import List, Tuple
LOADER_MAPPING = {
    ".pdf": PdfLoader,
    ".docx": DocxLoader,
    ".pptx": PptLoader
}

class Loader_Factory:
    @staticmethod
    def create_loader(file_path: str):
        ext = os.path.splitext(file_path)[-1].lower()
        loader_cls = LOADER_MAPPING.get(ext)
        if loader_cls is None:
            print(f"不支持的文件类型: {file_path}")
            return []
        loader = loader_cls(file_path)
        return loader

    @staticmethod
    def create_loader_for_files(file_paths: str) -> List[Tuple[str, object]]:
        """新增：批量处理文件路径，返回 (文件路径, Loader实例) 列表（无重复判断）"""
        valid_loaders = []
        for file_path in file_paths:
            # 仅判断1次后缀，直接创建Loader实例
            loader = Loader_Factory.create_loader(file_path)
            if loader:  # 过滤不支持的文件
                valid_loaders.append((file_path, loader))
        return valid_loaders


    @staticmethod
    def get_supported_extensions() -> List[str]:
        """辅助：获取所有支持的后缀（用于遍历文件夹时过滤文件）"""
        return list(LOADER_MAPPING.keys())


if __name__ == '__main__':
    file_path = '../../data/知识库资料_22033/2. 日常健康咨询/临床培训PPT/2024呼吸生理解剖.pptx'
    loader = Loader_Factory.create_loader(file_path)
    print(type(loader))