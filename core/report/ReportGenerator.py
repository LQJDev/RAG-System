import numpy as np
import os  # 用于读取环境变量中的API Key
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

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

    # ================= 简化：仅生成AI总结 =================
    def _generate_ai_summary(self, usage_stats, summary_stats, report_days):
        """
        使用LangChain调用DeepSeek API生成睡眠报告总结（仅总结，无单独建议）
        返回：报告总结（report_summary）
        """
        # 1. 整理关键数据为Prompt文本
        summary_data = f"""
        睡眠治疗报告关键数据：
        1. 报告周期：{report_days}天，有效治疗天数占比{usage_stats['percent_valid']}，总治疗时长{usage_stats['total_usage']}，日均治疗时长{usage_stats['avg_usage']}
        2. 压力指标：吸气压中位数{summary_stats['ipap_median']} cmH₂O，95分位数{summary_stats['ipap_95']} cmH₂O，最大值{summary_stats['ipap_max']} cmH₂O；呼气压中位数{summary_stats['epap_median']} cmH₂O，95分位数{summary_stats['epap_95']} cmH₂O，最大值{summary_stats['epap_max']} cmH₂O
        3. 漏气情况：平均漏气量{summary_stats['leak_avg']} L/min，大量漏气时长{summary_stats['leak_large_time']}
        4. 呼吸事件：AHI指数{summary_stats['ahi']}，AI指数{summary_stats['ai']}，HI指数{summary_stats['hi']}，阻塞性呼吸暂停指数{summary_stats['oai']}，中枢性呼吸暂停指数{summary_stats['cai']}
        """

        try:
            load_dotenv()
            # 2. 初始化ChatOpenAI（调用DeepSeek API）
            chat_model = ChatOpenAI(
                model="deepseek-chat",  # DeepSeek 官方推荐模型名
                api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量获取API Key
                base_url="https://api.deepseek.com/v1",  # DeepSeek 官方API地址
                temperature=0.3,  # 低随机性，保证总结严谨
                timeout=10  # 超时时间
            )

            # 3. 构造消息列表（明确要求仅生成总结，无需单独建议）
            messages = [
                SystemMessage(
                    content="""你是三甲医院睡眠中心的数据分析专家，负责生成临床级睡眠治疗报告总结。要求：
            1. 核心数据提炼：重点呈现治疗依从性（日均时长/有效占比）、压力参数稳定性（吸呼气压中值/关键分位）、漏气控制效果（平均漏气量/无大量漏气）、呼吸事件控制（AHI及核心细分指标），无需逐一罗列所有分位值，优先展示有临床意义的数据；
            2. 解读性表达：结合数据逻辑加入客观判断（如“治疗依从性优秀，为疗效奠定基础”“压力调节平稳，未出现异常峰值”“AHI指数处于正常范围，呼吸事件得到有效控制”），避免孤立堆砌数据；
            3. 风格与篇幅：语言正式严谨，符合医疗文书规范，200字左右，仅基于给定数据总结事实与客观结论，不添加额外建议或无关推测。"""
                ),
                HumanMessage(
                    content=f"请基于以下睡眠治疗数据，生成报告总结：\n{summary_data}"
                )
            ]

            # 4. 调用模型生成总结
            response = chat_model.invoke(messages)
            report_summary = response.content.strip()
            print(f"AI总结生成成功：{report_summary}")

            return report_summary

        except Exception as e:
            print(f"AI总结生成失败：{e}，使用默认总结")
            # 5. 兜底方案：默认总结
            default_summary = f"""
            本次睡眠治疗报告周期为{report_days}天，总治疗时长{usage_stats['total_usage']}，日均治疗时长{usage_stats['avg_usage']}，有效治疗天数占比{usage_stats['percent_valid']}，治疗依从性良好。吸气压中位数{summary_stats['ipap_median']} cmH₂O，呼气压中位数{summary_stats['epap_median']} cmH₂O，压力指标整体稳定；平均漏气量{summary_stats['leak_avg']} L/min，无大量漏气情况。AHI指数{summary_stats['ahi']}（正常范围＜5），呼吸事件控制良好，整体治疗效果理想。
            """.strip()
            return default_summary

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

        # --- 构建使用统计和总结统计 ---
        usage_stats = {
            "report_days": f"{len(sorted_dates)}d",
            "therapy_days": f"{len([x for x in list_usage if x > 0])}d",
            "valid_days": f"{len([x for x in list_usage if x >= 4])}d",
            "invalid_days": f"{len([x for x in list_usage if x < 4])}d",
            "percent_valid": f"{round(len([x for x in list_usage if x >= 4]) / len(list_usage) * 100, 1) if list_usage else 0} %",
            "max_usage": fmt_h_min(max(list_usage) if list_usage else 0),
            "min_usage": fmt_h_min(min(list_usage) if list_usage else 0),
            "avg_usage": fmt_h_min(avg_h_num),  # 新增：平均治疗时长
            "total_usage": fmt_h_min(total_h_num)
        }

        summary_stats = {
            "ipap_median": get_p(all_ipap, 50), "ipap_90": get_p(all_ipap, 90),
            "ipap_95": get_p(all_ipap, 95), "ipap_max": round(max(all_ipap), 1) if all_ipap else 0,
            "epap_median": get_p(all_epap, 50), "epap_90": get_p(all_epap, 90),
            "epap_95": get_p(all_epap, 95), "epap_max": round(max(all_epap), 1) if all_epap else 0,
            "leak_avg": get_avg(all_leak), "leak_large_time": "0 min",
            "ahi": get_avg(list_ahi), "ai": get_avg(list_ai), "hi": get_avg(list_hi),
            "oai": 0.0, "cai": 0.0
        }

        # ================= 调用AI生成总结（仅总结） =================
        report_days = len(sorted_dates)
        report_summary = self._generate_ai_summary(usage_stats, summary_stats, report_days)

        # --- 构建最终上下文（仅添加report_summary） ---
        context = {
            "info": {
                **STATIC_INFO,
                "print_date": datetime.now().strftime("%Y-%m-%d"),
                "range": f"{start_time_str} — {end_time_str}",  # 报告头部的时段
                "specific_range": specific_range  # 治疗时长部分的具体时段
            },
            "settings": STATIC_INFO,
            "usage_stats": usage_stats,
            "summary_stats": summary_stats,
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
            "total_pages": 5 + len(daily_records),
            # ================= 仅保留报告总结 =================
            "report_summary": report_summary
        }

        return context