import os
from typing import List
from core.factory.loader_factory import Loader_Factory
from langchain.schema import Document

def traverse_folder(folder_path: str) -> List[str]:
    """遍历文件夹，仅过滤支持的文件路径（不判断Loader，交给工厂类）"""
    supported_exts = Loader_Factory.get_supported_extensions()
    file_paths = []

    for root, _, files in os.walk(folder_path):
        # 1. 排除 __MACOSX 目录（直接跳过该目录下的所有文件）
        if "__MACOSX" in root:
            print(f"跳过缓存目录: {root}")
            continue

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_exts:  # 仅过滤，不判断Loader
                file_paths.append(os.path.join(root, file))

    return file_paths

def load_documents(source_path: str) -> List[Document]:
    """统一加载（）"""
    all_documents = []

    # 第一步：获取所有需要处理的文件路径
    if os.path.isdir(source_path):
        file_paths = traverse_folder(source_path)
        print(f"从文件夹 {source_path} 筛选出 {len(file_paths)} 个支持的文件")
    elif os.path.isfile(source_path):
        file_paths = [source_path]
    else:
        print(f"无效路径：{source_path}")
        return all_documents
    # 第二步：批量获取Loader（仅1次后缀判断，在Factory内部）
    file_loader_pairs = Loader_Factory.create_loader_for_files(file_paths)
    if not file_loader_pairs:
        print("无有效文件可加载")
        return all_documents

    # 第三步：直接用匹配好的Loader加载（无需再判断后缀）
    for file_path, loader in file_loader_pairs:
        try:
            docs = loader.load()
            all_documents.extend(docs)
            print(f"成功加载：{file_path}（获取 {len(docs)} 个文档块）")
        except Exception as e:
            print(f"加载失败 {file_path}：{str(e)}")
            continue

    return all_documents