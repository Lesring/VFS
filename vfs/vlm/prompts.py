from __future__ import annotations

SLOT_GUIDE = """\
可用 slot（必须从中选择，不要自创）：
- aircraft_type   飞机型号，如 "Airbus A320"、"Boeing 737-800"、"Unknown narrow-body jet"
- airline         航空公司，如 "China Eastern Airlines" / "中国东方航空"
- flight_number   航班号，如 "MU5101"（画面中若有喷涂/屏幕/播报）
- registration    机身注册号，如 "B-1234"、"N12345"
- airport         机场名称或三/四字码，如 "Shanghai Pudong (PVG/ZSPD)"
- origin          出发地机场/城市
- destination     目的地机场/城市
- landmark        地标、建筑物
- city / country  城市 / 国家
- coordinate      可见坐标文本，如 "31.23N,121.47E"
- date            日期或年月
- time_of_day     时刻/时段，如 "14:30"、"night"
- weather         天气状况，如 "晴"、"小雨"、"有雾"
- text_sign       画面中可读文字/招牌/屏幕内容（尽量逐字）
- object          显著物体，如 "boarding bridge"、"solar panels"、"railway"、"farmland"
- person_role     人物与制服，如 "民航地勤"、"飞行员"
- brand           品牌/涂装标识，如 "Star Alliance"
- activity        正在发生的事情，如 "飞机滑行"、"起飞"、"登机"、"飞机进近降落"
- viewpoint_vehicle 拍摄者所在载具，如 "train"、"bus"、"car"（仅画面能确认时）
"""

OUTPUT_SCHEMA = """\
只输出一个 JSON 对象，不要输出任何解释文字、不要用 markdown 围栏。格式：
{
  "scene": "一句话描述画面整体",
  "observations": [
    {"slot": "aircraft_type", "value": "Airbus A320", "confidence": 0.72, "note": "机翼末端有鲨鳍小翼，6 轮窄体"}
  ],
  "readable_text": ["画面中能看清的所有文字"],
  "uncertain": ["看不清或不确认真伪的地方"],
  "question_relevant": "与用户问题直接相关的观察"
}
confidence 为 0~1 的数字。看不清就写 "unclear" 并把 confidence 压到 0.3 以下，绝不臆造。
"""


def frame_prompt(question: str, hints: str, index: int, total: int, timestamp_s: float) -> str:
    parts = [
        "你是一名严谨的影像取证分析员。用户给了一段视频，你要从单帧中尽可能多地提取可核实的客观信息。",
        f"这是第 {index + 1}/{total} 帧，视频时间点约 {timestamp_s:.2f} 秒。",
    ]
    if question.strip():
        parts.append(f"用户最终要回答的问题是：{question.strip()}")
    if hints.strip():
        parts.append(f"用户补充提示（可信背景，可参考）：{hints.strip()}")
    parts.append(SLOT_GUIDE)
    parts.append(
        "要求：\n"
        "1. 只用画面里真实可见的东西作为依据，包括涂装、尾翼标志、机型特征（发动机数量与位置、翼梢小翼、机头形状、舷窗数量）、"
        "机场设施、登机桥编号、廊桥广告、地面车辆文字、天气光照、太阳方向、航班信息屏等。\n"
        "2. 每条观察必须能在画面中定位，note 里写清依据。\n"
        "3. 能读到的一切文字（含登机牌、行李牌、广告牌、机身小字、航班屏）都要放进 text_sign 或 readable_text。\n"
        "4. 不要因为问题问的是「目的地」就只答目的地，把所有能看到的线索都写出来。\n"
        "5. 不确定的一律降低 confidence。\n"
    )
    parts.append(OUTPUT_SCHEMA)
    return "\n".join(parts)


def synthesis_prompt(question: str, hints: str, per_frame_notes: str) -> str:
    parts = [
        "下面是同一段视频中多帧的独立观察记录。请做跨帧一致性与冲突检查，输出更可靠的合并结论。",
    ]
    if question.strip():
        parts.append(f"用户最终要回答的问题：{question.strip()}")
    if hints.strip():
        parts.append(f"用户提示：{hints.strip()}")
    parts.append("各帧观察：\n" + per_frame_notes)
    parts.append(
        "要求：\n"
        "1. 合并重复观察，同一事实被多帧确认则 confidence 提高；互相矛盾的（如两帧机型不同）要在 uncertain 中指出。\n"
        "2. 只保留最高置信度的结论，最多每个 slot 两条候选。\n"
        "3. 额外给出「consistency」字段，说明跨帧是否一致。\n"
    )
    parts.append(SLOT_GUIDE)
    parts.append(OUTPUT_SCHEMA)
    parts.append('额外在 JSON 顶层加 "consistency": "一致 / 部分冲突 / 严重冲突" 与 "reasoning": "合并推理过程"。')
    return "\n".join(parts)


def contact_sheet_prompt(question: str, hints: str, timestamps: list[float]) -> str:
    ts = ", ".join(f"{t:.1f}s" for t in timestamps)
    head = ["这是一段视频按时间顺序拼接的关键帧总览（从左到右、从上到下）。", f"各格对应时间点：{ts}。"]
    if question.strip():
        head.append(f"用户问题：{question.strip()}")
    if hints.strip():
        head.append(f"用户提示：{hints.strip()}")
    head.append(SLOT_GUIDE)
    head.append(
        "请先判断画面在时间上发生了什么变化（视角移动、镜头切换、主体变化），再提取客观线索。"
        "特别关注：机型细节、航司涂装、机场标识与航站楼编号、航班信息屏文字、地面车辆、光照与天气。"
    )
    head.append(OUTPUT_SCHEMA)
    return "\n".join(head)
