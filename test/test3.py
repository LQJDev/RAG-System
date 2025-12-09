import re
import ast
import os
import pathlib
import base64
import numpy as np
from datetime import datetime, timedelta
from jinja2 import Template
from playwright.sync_api import sync_playwright

# ================= 1. 配置：完全对标截图的静态信息 =================
STATIC_INFO = {
    "name": "王梅莉",
    "gender": "女性",
    "phone": "18611382920",
    "model": "RM-B80",
    "sn": "86N060456",
    "age": "",  # 截图未显示，留空
    "email": "",  # 截图未显示，留空

    # --- 治疗设置 (严格对照截图) ---
    "mode": "AUTO BPAP",
    "max_ipap": "15.0",
    "min_epap": "5.0",
    "max_ps": "5.0",
    "min_ps": "2.0",  # 截图中有
    "rise_time": "2",  # 升压档
    "trigger": "2",  # 触发灵敏度
    "cycle": "1",  # 撤换灵敏度
    "ramp": "OFF",  # 爬坡时间
    "ramp_p": "--",  # 爬坡压力 (截图显示 --)
    "humidifier": "2",  # 加湿器档位
    "tube_humidifier": "低档",  # 加湿呼吸管路 (截图显示 低档)
    "comf": "OFF"  # COMF (截图显示 OFF)
}


# ================= 2. 严格解析器 (保持不变) =================
class StrictParser:
    def __init__(self, content):
        self.content = content
        self.data_store = {}
        self.all_dates = []
        self.valid_headers = {
            "压力": "pressure", "漏气量": "leak", "潮气量": "tidal_vol",
            "平均吸呼比": "ie_ratio", "平均分钟通气量": "min_vent",
            "平均呼吸频率": "resp_rate", "呼吸事件": "events"
        }

    def parse(self):
        lines = self.content.split('\n')
        current_section = None
        current_date = None

        print(">>> [1/4] 正在解析原始数据...")

        for line in lines:
            line = line.strip()
            if not line: continue
            clean_header = line.replace("：", "").replace(":", "")
            if clean_header in self.valid_headers:
                current_section = self.valid_headers[clean_header]
                current_date = None
                continue
            if re.match(r'^\d{4}-\d{2}-\d{2}$', line):
                current_date = line
                if current_date not in self.data_store:
                    self._init_date(current_date)
                    if current_date not in self.all_dates:
                        self.all_dates.append(current_date)
                continue
            if current_section and current_date:
                arr_data = self._extract_list(line)
                if not arr_data: continue
                if line.startswith("时间"):
                    # 提取 HH:MM 用于画图
                    times = [str(t).split(' ')[1][:5] for t in arr_data if ' ' in str(t)]
                    if current_section == "pressure":
                        self.data_store[current_date]['pressure']['time'] = times
                        # 【关键】提取完整的时间字符串用于 Page 1 显示
                        # 例如: "2025-11-12 19:12:00"
                        if len(arr_data) > 0:
                            self.data_store[current_date]['pressure']['full_start'] = str(arr_data[0])
                            self.data_store[current_date]['pressure']['full_end'] = str(arr_data[-1])

                        self.data_store[current_date]['usage_seconds'] = self._calc_duration(arr_data)
                    elif current_section == "events":
                        self.data_store[current_date]['events']['time'] = arr_data
                    else:
                        self.data_store[current_date][current_section]['time'] = times
                elif current_section == "pressure":
                    if "IPAP" in line.upper():
                        self.data_store[current_date]['pressure']['ipap'] = self._clean_nums(arr_data)
                    elif "EPAP" in line.upper():
                        self.data_store[current_date]['pressure']['epap'] = self._clean_nums(arr_data)
                elif current_section == "events" and line.startswith("值"):
                    self.data_store[current_date]['events']['val'] = arr_data
                elif line.startswith("值"):
                    self.data_store[current_date][current_section]['val'] = self._clean_nums(arr_data)

    def _init_date(self, date):
        self.data_store[date] = {
            'usage_seconds': 0,
            'pressure': {'time': [], 'ipap': [], 'epap': [], 'full_start': '', 'full_end': ''},
            'leak': {'time': [], 'val': []},
            'tidal_vol': {'time': [], 'val': []},
            'ie_ratio': {'time': [], 'val': []},
            'min_vent': {'time': [], 'val': []},
            'resp_rate': {'time': [], 'val': []},
            'events': {'time': [], 'val': []}
        }

    def _extract_list(self, line):
        try:
            start = line.find('[')
            end = line.rfind(']')
            if start == -1 or end == -1: return []
            return ast.literal_eval(line[start:end + 1].replace("null", "None"))
        except:
            return []

    def _clean_nums(self, arr):
        return [float(x) if (x is not None and str(x).lower() != 'nan') else 0.0 for x in arr]

    def _calc_duration(self, arr):
        if len(arr) < 2: return 0
        try:
            t1 = datetime.strptime(arr[0], "%Y-%m-%d %H:%M:%S")
            t2 = datetime.strptime(arr[-1], "%Y-%m-%d %H:%M:%S")
            if t2 < t1: t2 += timedelta(days=1)
            return (t2 - t1).total_seconds()
        except:
            return 0


# ================= 3. 报告生成 =================
class ReportGenerator:
    def __init__(self, parser):
        self.parser = parser

    def _align_data(self, master_time, target_time, target_val):
        if not target_time or not target_val: return [0.0] * len(master_time)
        data_map = dict(zip(target_time, target_val))
        return [data_map.get(t, 0.0) for t in master_time]

    def _find_closest_index(self, target_time_str, time_axis_strs):
        try:
            def to_min(hm):
                h, m = map(int, hm.split(':')); return h * 60 + m

            target = to_min(target_time_str)
            best_idx = -1
            min_diff = 9999
            for i, t_str in enumerate(time_axis_strs):
                curr = to_min(t_str)
                diff = abs(curr - target)
                if diff > 720: diff = 1440 - diff
                if diff < min_diff: min_diff = diff; best_idx = i
                if min_diff <= 2: break
            if min_diff > 10: return -1
            return best_idx
        except:
            return -1

    def generate_context(self):
        sorted_dates = sorted(self.parser.all_dates, reverse=True)
        daily_records = []

        all_ipap, all_epap, all_leak = [], [], []
        list_ahi, list_ai, list_hi = [], [], []
        list_usage = []

        print(">>> [2/4] 正在统计与绘图映射...")

        # --- 计算精确的报告时段 (精确到秒) ---
        start_time_str = ""
        end_time_str = ""
        if sorted_dates:
            # 这里的逻辑是：最早那一天的开始时间 —— 最晚那一天的结束时间
            first_day = sorted_dates[-1]  # 最早日期
            last_day = sorted_dates[0]  # 最新日期

            p_first = self.parser.data_store[first_day]['pressure']
            p_last = self.parser.data_store[last_day]['pressure']

            start_time_str = p_first.get('full_start', '')
            end_time_str = p_last.get('full_end', '')

        specific_range = f"{start_time_str} — {end_time_str}"
        # -----------------------------------

        for d in sorted_dates:
            raw = self.parser.data_store[d]
            master_time = raw['pressure']['time']
            if not master_time: continue

            usage_h = round(raw['usage_seconds'] / 3600, 1)
            list_usage.append(usage_h)

            ipap = raw['pressure']['ipap']
            epap = raw['pressure']['epap']
            leak = self._align_data(master_time, raw['leak']['time'], raw['leak']['val'])

            L = len(master_time)
            ipap = (ipap + [0] * L)[:L]
            epap = (epap + [0] * L)[:L]

            all_ipap.extend(ipap)
            all_epap.extend(epap)
            all_leak.extend(leak)

            # 事件统计
            e_vals_raw = raw['events']['val']
            e_stats = {'OSA': 0, 'CSA': 0, 'HYP': 0, 'FL': 0, 'LL': 0}
            for evt in e_vals_raw:
                if evt and evt in e_stats: e_stats[evt] += 1

            # 绘图坐标
            processed_events_for_chart = []
            e_full_times = raw['events']['time']
            if e_full_times and e_vals_raw:
                for et, etype in zip(e_full_times, e_vals_raw):
                    if not etype: continue
                    try:
                        idx = self._find_closest_index(et.split(' ')[1][:5], master_time)
                        if idx != -1:
                            processed_events_for_chart.append({'idx': idx, 't': etype})
                    except:
                        pass

            div = usage_h if usage_h > 0 else 1.0
            ahi = round((e_stats['OSA'] + e_stats['CSA'] + e_stats['HYP']) / div, 1)
            ai = round((e_stats['OSA'] + e_stats['CSA']) / div, 1)
            hi = round(e_stats['HYP'] / div, 1)

            list_ahi.append(ahi)
            list_ai.append(ai)
            list_hi.append(hi)

            daily_records.append({
                "date": d,
                "time_axis": master_time,
                "ipap": ipap, "epap": epap, "leak": leak,
                "events": processed_events_for_chart,
                "usage_duration": usage_h
            })

        def get_p(arr, p):
            return round(np.percentile(arr, p), 1) if arr else 0

        def get_avg(arr):
            return round(np.mean(arr), 1) if arr else 0

        dates_asc = sorted_dates[::-1]

        # --- 格式化时间显示 (h min) ---
        def fmt_h_min(val):
            return f"{int(val)}h{int((val % 1) * 60)}min"

        total_h_num = sum(list_usage)
        avg_h_num = total_h_num / len(list_usage) if list_usage else 0  # 平均时长

        return {
            "info": {
                **STATIC_INFO,
                "print_date": datetime.now().strftime("%Y-%m-%d"),
                "range": f"{start_time_str} — {end_time_str}",  # 报告头部的时段
                "specific_range": specific_range  # 治疗时长部分的具体时段
            },
            "settings": STATIC_INFO,
            "usage_stats": {
                "report_days": f"{len(sorted_dates)}d",
                "therapy_days": f"{len([x for x in list_usage if x > 0])}d",
                "valid_days": f"{len([x for x in list_usage if x >= 4])}d",
                "invalid_days": f"{len([x for x in list_usage if x < 4])}d",
                "percent_valid": f"{round(len([x for x in list_usage if x >= 4]) / len(list_usage) * 100, 1) if list_usage else 0} %",
                "max_usage": fmt_h_min(max(list_usage) if list_usage else 0),
                "min_usage": fmt_h_min(min(list_usage) if list_usage else 0),
                "avg_usage": fmt_h_min(avg_h_num),  # 新增：平均治疗时长
                "total_usage": fmt_h_min(total_h_num)
            },
            "summary_stats": {
                "ipap_median": get_p(all_ipap, 50), "ipap_90": get_p(all_ipap, 90),
                "ipap_95": get_p(all_ipap, 95), "ipap_max": round(max(all_ipap), 1) if all_ipap else 0,
                "epap_median": get_p(all_epap, 50), "epap_90": get_p(all_epap, 90),
                "epap_95": get_p(all_epap, 95), "epap_max": round(max(all_epap), 1) if all_epap else 0,
                "leak_avg": get_avg(all_leak), "leak_large_time": "0 min",
                "ahi": get_avg(list_ahi), "ai": get_avg(list_ai), "hi": get_avg(list_hi),
                "oai": 0.0, "cai": 0.0
            },
            "summary_charts": {
                "dates": dates_asc,
                "ipap_median": [get_p(self.parser.data_store[d]['pressure']['ipap'], 50) for d in dates_asc],
                "ipap_95": [get_p(self.parser.data_store[d]['pressure']['ipap'], 95) for d in dates_asc],
                "ipap_max": [max(self.parser.data_store[d]['pressure']['ipap'] or [0]) for d in dates_asc],
                "epap_median": [get_p(self.parser.data_store[d]['pressure']['epap'], 50) for d in dates_asc],
                "epap_95": [get_p(self.parser.data_store[d]['pressure']['epap'], 95) for d in dates_asc],
                "epap_max": [max(self.parser.data_store[d]['pressure']['epap'] or [0]) for d in dates_asc],
                "leak_avg": [get_avg(self.parser.data_store[d]['leak']['val']) for d in dates_asc],
                "ahi": list_ahi[::-1], "ai": list_ai[::-1], "hi": list_hi[::-1],
                "usage_hours": list_usage[::-1]
            },
            "daily_records": daily_records,
            "total_pages": 5 + len(daily_records)
        }


# ================= 4. PDF 渲染 =================
def html_to_pdf(html_content, output_path):
    print(">>> [3/4] 启动浏览器渲染 PDF...")
    temp_html_path = os.path.abspath("temp_render_final_v3.html")
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    file_url = pathlib.Path(temp_html_path).as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # 🟢 修改这里：增加 timeout=0 (表示无限等待)，防止30秒自动报错
        print("    正在加载页面资源...")
        page.goto(file_url, wait_until="load", timeout=0)

        # 保持等待2秒给图表动画
        print("    正在渲染图表...")
        page.wait_for_timeout(2000)

        page.pdf(path=output_path, format="A4", print_background=True,
                 margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        browser.close()

    if os.path.exists(temp_html_path): os.remove(temp_html_path)
    print(">>> [4/4] ✨ PDF 生成完毕！")


if __name__ == "__main__":
    data_file = "../data/知识库资料_22033/报告解读案例/一周详细数据.txt"
    template_file = "temp_report6.html"
    output_pdf = "Sleep_Report_Official_Style.pdf"
    logo_file = "logo.jpg"  # 确保目录下有这个图片，或者png

    if not os.path.exists(data_file):
        print("❌ Error: data.txt not found")
        exit()

    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()

    parser = StrictParser(content)
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