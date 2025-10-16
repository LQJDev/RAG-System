from abc import ABC, abstractmethod
from langchain.schema import Document
from typing import List

class BaseLoader(ABC):
    """加载器基类，定义统一接口"""

    @abstractmethod
    def load(self) -> List[Document]:
        """加载文件并返回LangChain Document对象列表"""
        pass
