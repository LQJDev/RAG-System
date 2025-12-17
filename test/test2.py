from PyPDF2 import PdfReader, PdfWriter
import os


def split_pdf_by_pages(
        input_pdf_path: str,  # 你的大PDF路径（如：呼吸康复基础教程.pdf）
        output_dir: str,  # 拆分后的小PDF保存目录
        page_per_chunk: int = 8  # 每个小PDF的页数，建议5-10页（可调整）
):
    """
    仅按页码拆分PDF，生成独立的小PDF文件，不涉及任何文本提取
    """
    # 1. 创建输出目录（不存在则自动创建）
    os.makedirs(output_dir, exist_ok=True)

    # 2. 读取原始大PDF
    try:
        reader = PdfReader(input_pdf_path)
    except Exception as e:
        print(f"读取大PDF失败：{e}")
        return

    total_pages = len(reader.pages)
    print(f"原始大PDF共有 {total_pages} 页，将按每 {page_per_chunk} 页拆分")

    # 3. 批量生成小PDF
    for start_page in range(0, total_pages, page_per_chunk):
        # 计算当前小PDF的结束页码（避免超出总页数）
        end_page = min(start_page + page_per_chunk, total_pages)
        # 创建PDF写入器（用于生成小PDF）
        writer = PdfWriter()

        # 把对应页码的内容添加到写入器中
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        # 定义小PDF的文件名（标注页码范围，方便识别）
        input_filename = os.path.basename(input_pdf_path).replace(".pdf", "")
        output_filename = f"{input_filename}_第{start_page + 1}-{end_page}页.pdf"
        output_path = os.path.join(output_dir, output_filename)

        # 保存小PDF（核心步骤：仅生成PDF文件）
        with open(output_path, "wb") as f:
            writer.write(f)

        print(f"已生成小PDF：{output_path}")


# ------------------- 调用示例（替换为你的文件路径） -------------------
if __name__ == "__main__":
    # 替换为你的大PDF文件路径（如：呼吸康复基础教程.pdf）
    INPUT_PDF = r"D:\PythonProject\RAG-System\data\知识库资料_22033\2. 日常健康咨询\呼吸相关知识更新\内科重症监护病房工作手册.pdf"
    # 拆分后的小PDF保存目录（会自动创建）
    OUTPUT_DIR = "../data/知识库资料_22033/2. 日常健康咨询/呼吸相关知识更新/内科重症监护病房工作手册"
    # 每个小PDF的页数（建议5-10页，可根据豆包处理能力调整）
    PAGE_PER_CHUNK = 40

    # 执行拆分
    split_pdf_by_pages(INPUT_PDF, OUTPUT_DIR, PAGE_PER_CHUNK)