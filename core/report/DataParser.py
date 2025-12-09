import re
import ast
from datetime import datetime, timedelta



class DataParser:
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
