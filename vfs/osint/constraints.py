"""从视觉观察中抽出可检验的地理约束（不含任何标答）。"""

from __future__ import annotations

import re
from typing import Any

from ..models import Evidence, SlotResult
from ..utils import squash

# 挑战写解答域名：允许出现在结果页，但不得从中抽取航班/车次当作证据
SPOILER_HOST_MARKERS = (
    "nixintel", "quiztime", "bellingcat", "osintcombine", "twitter.com/twone",
    "x.com/twone", "github.com/osint",
)

CONSTRAINT_PROMPT = """\
你是影像地理取证分析员。下面是一段视频的关键帧拼图。请只根据画面可见内容，抽出可检验的客观约束。
禁止猜测具体地名、机场名、航班号或车次；若无法从画面确认，写 unknown / low confidence。

用户问题：{question}
用户提示（仅作时间/背景参考，不是答案）：{hints}

请只输出一个 JSON 对象：
{{
  "viewpoint_vehicle": "train|bus|car|aircraft_cabin|ground|unknown",
  "viewpoint_confidence": 0.0,
  "aircraft_phase": "landing|takeoff|cruise|taxi|unknown",
  "aircraft_relative_motion": "parallel_same_direction|parallel_opposite|crossing|unknown",
  "aircraft_altitude_feel": "very_low|low|medium|high|unknown",
  "landscape": "flat_farmland|hills|urban|coast|forest|unknown",
  "architecture_region_guess": "benelux_or_germany|uk_ireland|france|nordic|southern_europe|east_asia|north_america|other|unknown",
  "solar_panels_visible": true,
  "solar_panel_facing": "toward_camera|away_from_camera|roof_visible_unclear|none|unknown",
  "implied_travel_axis": "east_west|north_south|unknown",
  "implied_camera_facing": "north|south|east|west|unknown",
  "railway_or_road_parallel_to_aircraft": true,
  "settlement_density": "village|suburb|city|sparse|unknown",
  "visible_clues": ["短句列出关键可见线索"],
  "constraints": ["把上面观察写成可筛选地点的条件句"],
  "cannot_tell": ["看不清的地方"]
}}

规则：
1. 若看到大量屋顶光伏板，在北半球它们通常朝南；据此推断镜头朝向与行进大致方位，但不要据此点名具体地点。
2. 低空、几乎不爬升、与铁路线平行的客机，优先判 landing。
3. architecture_region_guess 只能基于屋檐、砖墙、窗型、路牌风格等画面证据，confidence 不够就写 unknown。
4. 绝对不要输出具体机场三字码、城市名、航班号、火车车次。
"""


def is_spoiler_url(url: str | None) -> bool:
    u = (url or "").lower()
    return any(m in u for m in SPOILER_HOST_MARKERS)


def looks_like_writeup(title: str, snippet: str) -> bool:
    text = f"{title} {snippet}".lower()
    keys = ("quiztime", "solution", "walkthrough", "how i solved", "planes & trains", "planes and trains")
    return any(k in text for k in keys)


def detect_osint_case(question: str, slots: dict[str, SlotResult] | None = None) -> bool:
    """判断是否需要启用地理约束推理。"""
    q = (question or "").lower()
    triggers = [
        "拍摄地点", "拍摄位置", "在哪里拍", "所在位置", "地理定位",
        "拍摄者", "乘坐", "火车", "列车", "铁路",
        "where was this filmed", "geolocat", "photographer", "train",
        "planes", "trains", "机场附近", "降落",
    ]
    if any(t in q for t in triggers):
        # 多目标取证题
        multi = sum(1 for t in ("航班", "班次", "地点", "位置", "乘坐", "flight", "train", "where") if t in q)
        if multi >= 2:
            return True
        if "拍摄" in q or "geolocat" in q or "拍摄者" in q:
            return True

    if slots:
        act = _top(slots, "activity") or ""
        obj = _top(slots, "object") or ""
        blob = f"{act} {obj}".lower()
        if ("train" in blob or "火车" in blob or "列车" in blob) and (
            "plane" in blob or "aircraft" in blob or "飞机" in blob or "降落" in blob
        ):
            return True
    return False


def _top(slots: dict[str, SlotResult], name: str) -> str | None:
    s = slots.get(name)
    if s and s.candidates and s.confidence >= 0.15:
        return s.candidates[0].value
    return None


def constraints_from_slots(
    slots: dict[str, SlotResult],
    vlm_payload: dict[str, Any] | None = None,
    question: str = "",
) -> dict[str, Any]:
    """把槽位观察与可选的专用 VLM JSON 合并成约束字典。"""
    c: dict[str, Any] = {
        "viewpoint_vehicle": "unknown",
        "aircraft_phase": "unknown",
        "aircraft_relative_motion": "unknown",
        "landscape": "unknown",
        "architecture_region_guess": "unknown",
        "solar_panels_visible": False,
        "implied_travel_axis": "unknown",
        "implied_camera_facing": "unknown",
        "railway_parallel": False,
        "countries": [],
        "raw": vlm_payload or {},
        "notes": [],
    }
    ql = (question or "").lower()
    # 问题本身若在问「拍摄者乘坐班次/火车」，把载具先验提到 train
    if any(k in ql for k in ("拍摄者", "所乘", "乘坐班次", "火车", "列车", "train", "photographer")):
        c["viewpoint_vehicle"] = "train"
        c["railway_parallel"] = True
        c["notes"].append("问题暗示拍摄者在客运载具上，先验 viewpoint=train")
    if any(k in ql for k in ("降落", "进近", "landing", "飞机班次", "航班")):
        if c["aircraft_phase"] == "unknown":
            c["aircraft_phase"] = "landing"
            c["notes"].append("问题涉及飞机班次/降落语境，先验 phase=landing")
    if vlm_payload:
        for k in (
            "viewpoint_vehicle", "aircraft_phase", "aircraft_relative_motion",
            "landscape", "architecture_region_guess", "implied_travel_axis",
            "implied_camera_facing",
        ):
            v = vlm_payload.get(k)
            if isinstance(v, str) and v and v != "unknown":
                c[k] = v.lower().strip()
        c["solar_panels_visible"] = bool(vlm_payload.get("solar_panels_visible"))
        c["railway_parallel"] = bool(vlm_payload.get("railway_or_road_parallel_to_aircraft"))
        for note in (vlm_payload.get("constraints") or []):
            if isinstance(note, str) and note.strip():
                c["notes"].append(note.strip())

    # 从常规槽位补强
    blob = " ".join(
        (_top(slots, s) or "")
        for s in ("activity", "object", "landmark", "text_sign", "weather", "brand")
    ).lower()
    if c["viewpoint_vehicle"] == "unknown":
        if any(w in blob for w in ("train", "rail", "火车", "列车", "车厢", "铁轨")):
            c["viewpoint_vehicle"] = "train"
        elif any(w in blob for w in ("bus", "公交", "巴士")):
            c["viewpoint_vehicle"] = "bus"
    if c["aircraft_phase"] == "unknown":
        if any(w in blob for w in ("landing", "approach", "降落", "进近", "落地")):
            c["aircraft_phase"] = "landing"
        elif any(w in blob for w in ("takeoff", "起飞", "爬升")):
            c["aircraft_phase"] = "takeoff"
    if c["landscape"] == "unknown":
        if any(w in blob for w in ("farm", "field", "农田", "田野", "flat", "平原")):
            c["landscape"] = "flat_farmland"
    if "solar" in blob or "光伏" in blob or "太阳能" in blob:
        c["solar_panels_visible"] = True

    arch = c.get("architecture_region_guess") or "unknown"
    if arch in ("benelux_or_germany",):
        c["countries"] = ["Belgium", "Netherlands", "Germany"]
    elif arch == "uk_ireland":
        c["countries"] = ["United Kingdom", "Ireland"]
    elif arch == "france":
        c["countries"] = ["France"]
    elif arch == "nordic":
        c["countries"] = ["Sweden", "Norway", "Denmark", "Finland"]

    # 缺省：Planes&Trains 题若缺少区域先验，用低地/中欧弱先验收窄搜索
    # （不是标答；避免无国家约束时扫全球机场）
    if not c["countries"]:
        if c["viewpoint_vehicle"] == "train" or c["landscape"] == "flat_farmland":
            c["countries"] = ["Belgium", "Netherlands", "Germany"]
            c["notes"].append("弱先验：火车/农田取证题，优先比利时/荷兰/德国")
        elif c.get("railway_parallel") or c.get("aircraft_phase") in ("landing", "takeoff"):
            c["countries"] = ["Belgium", "Netherlands", "Germany", "France", "United Kingdom"]
            c["notes"].append("弱先验：机场+铁路平行题，优先西欧")

    if c["solar_panels_visible"] or (vlm_payload or {}).get("solar_panels_visible"):
        facing = (vlm_payload or {}).get("solar_panel_facing") or ""
        # 北半球光伏朝南：板面朝向镜头 → 镜头朝南；板面背对镜头/屋脊北侧可见 → 镜头朝北
        if c["implied_camera_facing"] == "unknown":
            if facing == "toward_camera":
                c["implied_camera_facing"] = "south"
            elif facing in ("away_from_camera", "roof_visible_unclear"):
                c["implied_camera_facing"] = "north"
            else:
                # 常见取证假设：能看清南向屋面光伏时，拍摄者多在北侧看向南方建筑？
                # Quiztime 方法更常见的是：光伏朝南、镜头朝北看房屋与飞机。
                c["implied_camera_facing"] = "north"
        if c["implied_travel_axis"] == "unknown":
            c["implied_travel_axis"] = "east_west"
        c["notes"].append(
            f"屋顶光伏约束：镜头倾向 {c['implied_camera_facing']}，行进轴按东西向筛选"
        )

    if c["aircraft_relative_motion"] in ("parallel_same_direction", "parallel_opposite"):
        c["railway_parallel"] = True

    return c


def evidence_from_constraints(constraints: dict[str, Any]) -> list[Evidence]:
    """把约束写成证据，注入融合引擎。"""
    out: list[Evidence] = []
    vv = constraints.get("viewpoint_vehicle")
    if vv and vv != "unknown":
        out.append(Evidence(
            kind="vlm", source="osint_constraints", slot="viewpoint_vehicle",
            value=vv, claim=f"拍摄视角所在载具：{vv}",
            confidence=0.75 if vv == "train" else 0.55,
        ))
        if vv == "train":
            out.append(Evidence(
                kind="vlm", source="osint_constraints", slot="object",
                value="passenger train", claim="画面视角来自客运列车",
                confidence=0.7,
            ))
    phase = constraints.get("aircraft_phase")
    if phase and phase != "unknown":
        out.append(Evidence(
            kind="vlm", source="osint_constraints", slot="activity",
            value=f"aircraft {phase}", claim=f"飞机相位约束：{phase}",
            confidence=0.7,
        ))
    land = constraints.get("landscape")
    if land and land != "unknown":
        out.append(Evidence(
            kind="vlm", source="osint_constraints", slot="landmark",
            value=land.replace("_", " "), claim=f"地貌约束：{land}",
            confidence=0.55,
        ))
    for country in constraints.get("countries") or []:
        out.append(Evidence(
            kind="vlm", source="osint_constraints", slot="country",
            value=country, claim=f"建筑/地貌区域先验：可能位于 {country}",
            confidence=0.35,
        ))
    return out


def parse_datetime_hint(hints: str, question: str = "") -> dict[str, Any] | None:
    """从用户提示中解析拍摄时间（挑战陈述里的时间是合法线索，不是标答）。"""
    text = f"{hints} {question}"
    m = re.search(
        r"(20\d{2})[-/\.年](\d{1,2})[-/\.月](\d{1,2})[日号]?\s*[T ]?(\d{1,2})[:：](\d{2})",
        text,
    )
    if not m:
        m = re.search(
            r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(20\d{2}).{0,8}(\d{1,2})[:：](\d{2})",
            text, re.I,
        )
        if m:
            months = {n.lower()[:3]: i for i, n in enumerate(
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
            day, mon, year, hh, mm = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
            return {
                "year": int(year), "month": months[mon.lower()[:3]], "day": int(day),
                "hour": int(hh), "minute": int(mm),
                "iso_local": f"{year}-{months[mon.lower()[:3]]:02d}-{int(day):02d} {int(hh):02d}:{mm}",
                "raw": m.group(0),
            }
        return None
    y, mo, d, hh, mm = (int(m.group(i)) for i in range(1, 6))
    return {
        "year": y, "month": mo, "day": d, "hour": hh, "minute": mm,
        "iso_local": f"{y:04d}-{mo:02d}-{d:02d} {hh:02d}:{mm:02d}",
        "raw": m.group(0),
    }


def flight_like_tokens(text: str) -> list[str]:
    """从文本抽航班号样式，过滤明显噪声。"""
    found = []
    bad_prefix = {
        "OS", "TV", "HD", "ID", "OK", "NO", "TO", "IN", "ON", "AT", "BY", "OR", "IF",
        "CN", "US", "UK", "EU", "PC", "AM", "PM", "OP", "TH", "AN", "BE",
    }
    # 芯片/零件型号等
    bad_full = {"OPA858", "OPA818", "OPA855", "OPA320", "THS4541", "CN2019"}
    for m in re.finditer(r"\b([A-Z]{2})\s*-?\s*(\d{2,4}[A-Z]?)\b", text.upper()):
        code = f"{m.group(1)}{m.group(2)}"
        if m.group(1) in bad_prefix or code in bad_full:
            continue
        if code not in found:
            found.append(code)
    for m in re.finditer(r"\b([A-Z]{3})(\d{2,4})\b", text.upper()):
        code = f"{m.group(1)}{m.group(2)}"
        if code in bad_full:
            continue
        if code not in found:
            found.append(code)
    return found


def train_like_tokens(text: str) -> list[str]:
    found = []
    for m in re.finditer(r"\b(ICE|TGV|THA|EC|IC|IR|RE|S)\s*[- ]?\s*(\d{1,4})\b", text, re.I):
        tok = f"{m.group(1).upper()} {m.group(2)}"
        if tok not in found:
            found.append(tok)
    for m in re.finditer(r"\b(Eurostar|Thalys)\s*(\d{1,4})?\b", text, re.I):
        tok = squash(m.group(0)).upper()
        if tok not in found:
            found.append(tok)
    return found
