"""答案生成。分两层：
1. 主答案：直接回应用户的问题（主目标槽位）。
2. 推理链：按"看到什么 → 联网核实了什么 → 怎么推出来的"组织，可审计。
"""

from __future__ import annotations

import re
from typing import Any

from .intent import goal_candidates
from .models import Evidence, FrameInfo, KnowledgeHit, SlotResult
from .reason import FusionEngine
from .utils import clip, confidence_label, fmt_pct, squash

SLOT_CONF_CAP = 0.97


def build_answer(
    engine: FusionEngine,
    intent: dict,
    question: str,
    frames: list[FrameInfo],
    evidence: list[Evidence],
    knowledge: list[KnowledgeHit],
    warnings: list[str],
    allow_abstain: bool = True,
) -> tuple[str, float, list[str]]:
    slots = engine.slots
    primary = intent.get("primary_goal") or "activity"
    goals = goal_candidates(intent)
    lines: list[str] = []

    # ---- 主答案 ----
    ans_slot = None
    for slot in goals:
        s = slots.get(slot)
        if s and s.candidates and s.confidence >= 0.2:
            ans_slot = s
            break

    if ans_slot is None:
        if allow_abstain:
            lines.append("**结论：信息不足，无法给出可靠判定。**")
            lines.append("")
            lines.append(_what_we_saw(engine, frames, evidence))
        else:
            lines.append("**结论：未能从画面中得到可判定的线索。**")
        return "\n".join(lines), 0.0, engine.trace

    conf = min(SLOT_CONF_CAP, ans_slot.confidence)
    # 冲突在下面才计算，这里先占位，稍后回填标签
    label = confidence_label(conf, engine.low_conf)

    if ans_slot.slot == primary:
        lines.append(f"**结论：{ans_slot.label} — {ans_slot.chosen}**（置信度 {fmt_pct(conf)}，{label}）")
    else:
        # 主目标答不出，说明改答什么
        plabel = intent.get("primary_label") or primary
        lines.append(f"**关于「{plabel}」：画面中缺少直接证据，无法确定。**")
        lines.append("")
        lines.append(f"**但可以确定的是：{ans_slot.label} — {ans_slot.chosen}**（置信度 {fmt_pct(conf)}，{label}）")

    # 候选与依据
    top = ans_slot.candidates[0]
    if top.reasons:
        lines.append("")
        lines.append("主要依据：")
        for r in top.reasons[:5]:
            lines.append(f"- {r}")

    # 次优候选
    others = [c for c in ans_slot.candidates[1:] if c.score >= 0.15]
    if others:
        alt = " / ".join(f"{c.value}（{fmt_pct(c.score)}）" for c in others[:3])
        lines.append(f"- 其他可能：{alt}")

    # 多目标题：航班 / 地点 / 乘坐班次分项列出
    prefer = ["flight_number", "landmark", "airport", "city", "photographer_ride", "viewpoint_vehicle"]
    multi_goals = []
    for g in prefer + list(goals):
        if g in multi_goals:
            continue
        if g in slots and slots[g].candidates and slots[g].confidence >= 0.25:
            multi_goals.append(g)
    if len([g for g in multi_goals if g in (goals or prefer)]) >= 2:
        lines.append("")
        lines.append("**分项结论：**")
        for g in multi_goals[:8]:
            s = slots[g]
            c = s.candidates[0]
            lines.append(f"- {s.label}：{c.value}（{fmt_pct(min(SLOT_CONF_CAP, c.score))}）")
        if "flight_number" in goals and (
            "flight_number" not in multi_goals
            or (slots.get("flight_number") and slots["flight_number"].confidence < 0.35)
        ):
            lines.append("- 航班号：历史 ADS-B/到达数据源不可用，尚无法可靠锁定具体班次")

    # ---- 其他已识别的关键信息 ----
    lines.append("")
    lines.append("**其他识别到的信息：**")
    shown = 0
    shown_multi = set(multi_goals if len(multi_goals) >= 2 else [])

    for slot in ("flight_number", "photographer_ride", "viewpoint_vehicle", "airline",
                 "aircraft_type", "registration", "airport",
                 "origin", "destination", "city", "country", "date", "time_of_day",
                 "weather", "landmark", "activity", "text_sign"):
        if slot == ans_slot.slot or slot in shown_multi:
            continue
        s = slots.get(slot)
        if not s or not s.candidates:
            continue
        c = s.candidates[0]
        if c.score < 0.15:
            continue
        mark = "" if c.score >= engine.low_conf else "（推测）"
        lines.append(f"- {s.label}：{c.value}{mark} — {fmt_pct(c.score)}")
        shown += 1
    if shown == 0 and len(shown_multi) == 0:
        lines.append("- 画面中未提取到其他可靠实体")

    # ---- 冲突与不确定性 ----
    _CONFLICT_WORDS = ("冲突", "不一致", "不匹配", "张力", "不符", "矛盾")
    conflicts = [n for s in slots.values() for n in s.notes if any(w in n for w in _CONFLICT_WORDS)]
    if conflicts:
        lines.append("")
        lines.append("**需要注意的矛盾点：**")
        for c in dict.fromkeys(conflicts):
            lines.append(f"- {c}")
        # 有未解决的矛盾时，结论标签不能显示为"高"
        if label == "高":
            label = "中（存在矛盾，需复核）"
            lines[0] = lines[0].replace(f"，{confidence_label(conf, engine.low_conf)}）",
                                        f"，{label}）", 1)

    # ---- 外部佐证 ----
    if knowledge:
        lines.append("")
        lines.append("**外部佐证：**")
        for h in knowledge[:4]:
            src = h.source or "web"
            title = clip(h.title, 70)
            line = f"- [{src}] {title}"
            if h.snippet:
                line += f"：{clip(h.snippet, 130)}"
            if h.url:
                line += f"（{h.url}）"
            lines.append(line)

    # ---- 假设链 ----
    chain = _hypothesis_chain(engine, ans_slot, conflicts)
    if chain:
        lines.append("")
        lines.append("**推理链：**")
        for i, step in enumerate(chain, 1):
            lines.append(f"{i}. {step}")

    if warnings:
        lines.append("")
        lines.append("**分析限制：**")
        for w in dict.fromkeys(warnings):
            lines.append(f"- {w}")

    return "\n".join(lines), conf, engine.trace


def _what_we_saw(engine: FusionEngine, frames: list[FrameInfo], evidence: list[Evidence]) -> str:
    out = ["**画面中能确认的信息：**"]
    n = 0
    for slot, s in engine.slots.items():
        if s.candidates and s.candidates[0].score >= 0.3:
            out.append(f"- {s.label}：{s.candidates[0].value}（{fmt_pct(s.candidates[0].score)}）")
            n += 1
    if n == 0:
        out.append("- 未能提取到可核实的实体，建议提供更清晰的画面或补充拍摄背景")
    out.append("")
    out.append(f"共分析 {len(frames)} 个关键帧、{len(evidence)} 条证据。"
               "要提高结论可靠性，建议：① 提供含航班信息屏/登机桥编号的清晰画面；"
               "② 补充拍摄时间与地点；③ 配置视觉模型以获得更细的机型与涂装判读。")
    return "\n".join(out)


def _hypothesis_chain(engine: FusionEngine, ans_slot: SlotResult, conflicts: list[str] | None = None) -> list[str]:
    """把关键证据串成人类可读的推理链。

    关键点：结论常常不是直接从画面读出来的，而是"画面读到 A → 用 A 查库 → 得到 B"。
    所以要沿上游线索回溯，否则推理链会变成没有前提的断言。
    """
    chain: list[str] = []
    top = ans_slot.candidates[0]
    by_id = {e.id: e for e in engine.evidence_list()}

    visual = [by_id[i] for i in top.evidence_ids if i in by_id and by_id[i].kind in ("vlm", "ocr")]
    external = [by_id[i] for i in top.evidence_ids if i in by_id and by_id[i].kind in ("knowledge", "web")]

    # 没有视觉前提时，回溯到 citable 的上游线索（航班号/注册号/坐标等）
    if not visual:
        for up_slot in ("flight_number", "registration", "airport", "landmark", "coordinate"):
            if up_slot == ans_slot.slot:
                continue
            up = engine.slots.get(up_slot)
            if not up or not up.candidates:
                continue
            up_top = up.candidates[0]
            up_visual = [by_id[i] for i in up_top.evidence_ids
                         if i in by_id and by_id[i].kind in ("vlm", "ocr")]
            if not up_visual:
                continue
            ev = up_visual[0]
            where = f"（{ev.timestamp_s:.1f}s）" if ev.timestamp_s is not None else ""
            chain.append(f"画面中读到{up.label} =「{clip(ev.value, 50)}」{where}：{clip(ev.claim, 110)}")
            chain.append(f"以该{up.label}为线索查外部数据源")
            break

    for ev in visual[:2]:
        where = f"（{ev.timestamp_s:.1f}s）" if ev.timestamp_s is not None else ""
        chain.append(f"画面中读到 {ev.slot} =「{clip(ev.value, 60)}」{where}：{clip(ev.claim, 120)}")

    if external:
        for ev in external[:2]:
            chain.append(f"外部数据源返回 → {clip(ev.claim, 150)}（{ev.source}）")
    elif visual:
        chain.append("该结论仅来自画面判读，未获得外部数据源交叉验证")

    if not chain:
        chain.append(f"综合多帧观察，{ans_slot.label} 最可能为「{ans_slot.chosen}」")

    conf = ans_slot.confidence
    has_conflict = bool(conflicts)
    if has_conflict:
        chain.append(f"但存在上述矛盾点，综合置信度 {fmt_pct(conf)}，结论应视为待核实")
    elif conf >= 0.7:
        chain.append(f"多源交叉验证一致，综合置信度 {fmt_pct(conf)}")
    elif conf >= 0.35:
        chain.append(f"线索方向一致但缺乏直接佐证，置信度 {fmt_pct(conf)}，建议人工复核")
    else:
        chain.append(f"证据薄弱，仅供参考，置信度仅 {fmt_pct(conf)}")

    return chain


def summarize_question(question: str) -> str:
    q = squash(question)
    return q if q else "（用户未给出明确问题，按通用取证流程全面分析）"
