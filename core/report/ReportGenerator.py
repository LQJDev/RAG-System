import numpy as np
from datetime import datetime, timedelta

# ================= 配置：静态信息 =================
STATIC_INFO = {
    "name": "测试用户",
    "gender": "男",
    "phone": "13800000000",
    "model": "ResMed AirSense 10",
    "sn": "86N060456",
    "mode": "AutoSet",
    "max_ipap": "15.0",
    "min_epap": "5.0",
    "max_ps": "N/A",
    "ramp": "Auto",
    "humidifier": "4"
}


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
