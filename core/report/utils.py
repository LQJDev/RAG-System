import os
import pathlib
from playwright.sync_api import sync_playwright


def html_to_pdf(html_content, output_path):
    print(">>> [3/4] 启动浏览器渲染 PDF...")
    temp_html_path = os.path.abspath("temp_render_final_v2.html")
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    file_url = pathlib.Path(temp_html_path).as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # 核心修改 1：把 wait_until 从 'networkidle' 改为 'load' (只要页面加载完就行，不用等网络完全静止)
        # 核心修改 2：增加 timeout 到 60秒 (60000ms)
        try:
            page.goto(file_url, wait_until="load", timeout=60000)
        except Exception as e:
            print(f"⚠️ 警告: 页面加载耗时较长，尝试强制渲染... ({e})")

        # 核心修改 3：硬等待 2 秒，给 ECharts 留出画图的时间 (这一步最关键)
        print("    正在等待图表渲染...")
        page.wait_for_timeout(2000)

        page.pdf(path=output_path, format="A4", print_background=True,
                 margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        browser.close()

    if os.path.exists(temp_html_path): os.remove(temp_html_path)
    print(">>> [4/4] ✨ PDF 生成完毕！")