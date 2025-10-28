from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List


def split_documents(
        documents: List[Document],
        chunk_size: int = 1500, # 文本块大小
        chunk_overlap: int = 100, # 文本块重叠部分
        separators: List[str] = None
) -> List[Document]:
    """
        分割文档为文本块，支持中文语境优化

        参数:
            documents: 原始文档列表（langchain的Document对象）
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻文本块的重叠字符数（保证上下文连贯性）
            separators: 自定义分隔符列表（优先使用）

        返回:
            分割后的文本块列表
        """
    # 默认分隔符（中文优先按段落/句子分割，英文按标点分割）
    default_separators = [
        # 新增：优先按四级标题分割（单个步骤为一个单元）
        "#### ",
        "\n\n",  # 再按空行（段落）分割
        "\n",  # 按换行分割
        "。", "！", "？",  # 中文句子结尾
        ".", "!", "?",  # 英文句子结尾
        "，", ";", ":",  # 中文逗号/分号
        " ", ""
    ]

    # 使用自定义分隔符（如果提供）
    used_separators = separators if separators else default_separators

    # 初始化递归字符分割器（最常用的分割器，支持按优先级分割）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # 文本块大小
        chunk_overlap=chunk_overlap,  # 重叠部分
        separators=used_separators,  # 分隔符列表
        length_function=len  # 计算长度的函数（中文用len即可）
    )

    # 分割文档
    split_docs = text_splitter.split_documents(documents)

    return split_docs