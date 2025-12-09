import os
import base64
from jinja2 import Template
from core.report.ReportGenerator import ReportGenerator
from core.report.DataParser import DataParser
from core.report.utils import html_to_pdf

if __name__ == "__main__":
    data_file = "../data/知识库资料_22033/报告解读案例/一周详细数据.txt"
    template_file = "../template/report_template.html"
    output_pdf = "../output/Sleep_Report_Official_Style.pdf"
    logo_file = "../template/logo.jpg"  # 确保目录下有这个图片，或者png

    if not os.path.exists(data_file):
        print("❌ Error: data.txt not found")
        exit()

    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()

    parser = DataParser(content)
    parser.parse()

    generator = ReportGenerator(parser)
    context = generator.generate_context()

    # 处理 Logo
    if os.path.exists(logo_file):
        with open(logo_file, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            context['logo_src'] = f"data:image/jpeg;base64,{encoded}"
    else:
        context['logo_src'] = ""

    with open(template_file, 'r', encoding='utf-8') as f:
        template_str = f.read()

    template = Template(template_str)
    rendered_html = template.render(**context)

    try:
        html_to_pdf(rendered_html, output_pdf)
        print(f"✅ 成功! 文件已生成: {output_pdf}")
    except Exception as e:
        print(f"❌ 失败: {e}")