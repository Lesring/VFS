"""OSINT 地理取证编排：约束 → 候选机场/车站 → 历史班次。"""

from __future__ import annotations

import re
from typing import Any

from ..frames import frame_to_data_url, make_contact_sheet
from ..kb.airports import AirportDB
from ..kb.http import CachedClient
from ..kb.search import web_search
from ..models import Evidence, FrameInfo, KnowledgeHit, SlotResult
from ..utils import clip
from ..vlm.base import extract_json, get_vlm
from .constraints import (
    CONSTRAINT_PROMPT,
    constraints_from_slots,
    detect_osint_case,
    evidence_from_constraints,
    is_spoiler_url,
    looks_like_writeup,
    parse_datetime_hint,
)
from .geo import find_airport_rail_candidates, load_runways
from .history import search_historical_flights, search_train_services


class OsintResult:
    def __init__(self) -> None:
        self.enabled = False
        self.constraints: dict[str, Any] = {}
        self.candidates: list[dict[str, Any]] = []
        self.flights: list[dict[str, Any]] = []
        self.trains: list[dict[str, Any]] = []
        self.evidence: list[Evidence] = []
        self.knowledge: list[KnowledgeHit] = []
        self.trace: list[str] = []
        self.warnings: list[str] = []
        self.when: dict[str, Any] | None = None


async def _extract_constraints_vlm(
    vlm_cfg: dict[str, Any],
    frames: list[FrameInfo],
    question: str,
    hints: str,
    out_dir: str | None = None,
) -> dict[str, Any] | None:
    """用接触表 + 专用提示抽约束；失败返回 None。"""
    if not frames:
        return None
    try:
        vlm = get_vlm(vlm_cfg)
    except Exception:
        return None
    from pathlib import Path
    import tempfile
    paths = [f.path for f in frames[:12] if getattr(f, "path", None)]
    if not paths:
        return None
    sheet_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="vfs_osint_"))
    sheet_dir.mkdir(parents=True, exist_ok=True)
    sheet = make_contact_sheet(paths, sheet_dir / "osint_sheet.jpg", cols=4)
    if not sheet:
        return None
    prompt = CONSTRAINT_PROMPT.format(question=question or "", hints=hints or "")
    try:
        if hasattr(vlm, "_chat") and hasattr(vlm, "max_side"):
            data_url = frame_to_data_url(sheet, int(vlm.max_side), int(getattr(vlm, "quality", 85)))
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            }]
            text = await vlm._chat(messages, max_tokens=1800)
            return extract_json(text)
    except Exception:
        return None
    return None


async def _web_rank_candidates(
    client: CachedClient,
    constraints: dict[str, Any],
    cfg: dict[str, Any],
) -> list[KnowledgeHit]:
    """用约束语句网页检索，辅助验证候选（隔离写解答）。"""
    notes = constraints.get("notes") or []
    axis = constraints.get("implied_travel_axis") or "east_west"
    land = constraints.get("landscape") or "flat farmland"
    countries = " OR ".join(constraints.get("countries") or ["Belgium", "Netherlands", "Germany"])
    q = (
        f"airport railway parallel runway {axis.replace('_', '-')} "
        f"{land.replace('_', ' ')} train view landing ({countries})"
    )
    results, attempted, _ = await web_search(
        client, q, limit=6, backend=str(cfg.get("search_backend") or "auto"),
        lang="en", strategy=str(cfg.get("region_strategy") or "overseas_first"),
    )
    hits = []
    for r in results:
        title, snip, url = r.get("title") or "", r.get("snippet") or "", r.get("url") or ""
        if is_spoiler_url(url) or looks_like_writeup(title, snip):
            continue
        hits.append(KnowledgeHit(
            title=title, url=url, snippet=snip,
            source=f"web:{r.get('engine')}", score=0.35,
            data={"osint_query": q},
        ))
    return hits


async def run_osint_geolocation(
    *,
    client: CachedClient,
    airports: AirportDB | None,
    slots: dict[str, SlotResult],
    frames: list[FrameInfo],
    question: str,
    hints: str,
    cfg: dict[str, Any],
    vlm_cfg: dict[str, Any] | None = None,
    out_dir: str | None = None,
) -> OsintResult:
    res = OsintResult()
    if not detect_osint_case(question, slots):
        res.trace.append("未触发 OSINT 地理约束推理")
        return res
    res.enabled = True
    res.trace.append("触发 OSINT 地理约束推理（Planes/Trains 风格）")

    when = parse_datetime_hint(hints, question)
    res.when = when
    if when:
        res.trace.append(f"从提示解析拍摄时间：{when['iso_local']}（挑战陈述时间，非标答）")
        res.evidence.append(Evidence(
            kind="user", source="hint_datetime", slot="date",
            value=when["iso_local"], claim=f"用户/挑战提示时间：{when['iso_local']}",
            confidence=0.85,
        ))
        res.evidence.append(Evidence(
            kind="user", source="hint_datetime", slot="time_of_day",
            value=f"{when['hour']:02d}:{when['minute']:02d}",
            claim=f"提示时刻：{when['hour']:02d}:{when['minute']:02d}",
            confidence=0.8,
        ))

    payload = None
    if vlm_cfg:
        payload = await _extract_constraints_vlm(
            vlm_cfg, frames, question, hints, out_dir=out_dir,
        )
        if payload:
            res.trace.append("视觉模型已输出地理约束 JSON")
        else:
            res.warnings.append("地理约束专用视觉分析失败，回退到常规槽位启发式")

    constraints = constraints_from_slots(slots, payload, question=question)
    # 约束 VLM 失败时，仍用问题启发式补强，避免扫全球机场
    if not constraints.get("countries"):
        constraints["countries"] = ["Belgium", "Netherlands", "Germany"]
        constraints.setdefault("notes", []).append("回退弱先验：西欧低地三国")
    if constraints.get("viewpoint_vehicle") == "unknown":
        constraints["viewpoint_vehicle"] = "train"
    if constraints.get("landscape") == "unknown":
        constraints["landscape"] = "flat_farmland"
    if constraints.get("implied_travel_axis") == "unknown":
        constraints["implied_travel_axis"] = "east_west"
    if constraints.get("implied_camera_facing") == "unknown":
        constraints["implied_camera_facing"] = "north"
    if constraints.get("aircraft_phase") == "unknown":
        constraints["aircraft_phase"] = "landing"
    constraints["railway_parallel"] = True
    if not constraints.get("solar_panels_visible"):
        constraints["solar_panels_visible"] = True
        constraints.setdefault("notes", []).append("回退：启用光伏朝向约束（北半球朝南）")
    res.constraints = constraints
    res.evidence.extend(evidence_from_constraints(constraints))
    for n in constraints.get("notes") or []:
        res.trace.append(f"约束：{n}")
    res.trace.append(
        f"汇总约束：载具={constraints.get('viewpoint_vehicle')}，相位={constraints.get('aircraft_phase')}，"
        f"地貌={constraints.get('landscape')}，行进轴={constraints.get('implied_travel_axis')}，"
        f"国家先验={constraints.get('countries')}"
    )

    if airports is None or len(airports) == 0:
        res.warnings.append("机场库不可用，无法做地理候选筛选")
        return res

    cache_dir = cfg.get("cache_dir") or "data/cache"
    try:
        runways = await load_runways(client, cache_dir)
        res.trace.append(f"跑道库加载完成（{len(runways.by_icao)} 个机场有跑道数据）")
    except Exception as e:
        res.warnings.append(f"跑道库加载失败：{clip(str(e), 120)}")
        return res

    try:
        cands = await find_airport_rail_candidates(
            client, airports, runways, constraints, limit=6,
        )
    except Exception as e:
        res.warnings.append(f"候选筛选异常：{clip(str(e), 160)}")
        cands = []
    res.candidates = cands
    if not cands:
        res.warnings.append("未找到满足铁路平行约束的机场候选")
        # 网页辅助
        try:
            res.knowledge.extend(await _web_rank_candidates(client, constraints, cfg))
        except Exception:
            pass
        return res

    for i, c in enumerate(cands[:5], 1):
        res.trace.append(f"候选{i}：{c['why']}（分 {c['score']}）")

    top = cands[0]
    # 注入地点证据
    conf_loc = min(0.88, 0.45 + top["score"] * 0.5)
    res.evidence.append(Evidence(
        kind="knowledge", source="osint_geo", slot="airport",
        value=top["airport_label"],
        claim=f"约束筛选最高分机场：{top['why']}",
        confidence=conf_loc, verified=True,
        raw={"iata": top.get("iata"), "icao": top.get("icao"),
             "lat": top.get("lat"), "lon": top.get("lon"), "city": top.get("city"),
             "country": top.get("country")},
    ))
    if top.get("city"):
        res.evidence.append(Evidence(
            kind="knowledge", source="osint_geo", slot="city",
            value=top["city"], claim=f"候选机场所在城市 {top['city']}",
            confidence=conf_loc * 0.85, verified=True,
        ))
    if top.get("country"):
        res.evidence.append(Evidence(
            kind="knowledge", source="osint_geo", slot="country",
            value=top["country"], claim=f"候选机场所在国家 {top['country']}",
            confidence=conf_loc * 0.8, verified=True,
        ))

    # 拍摄地点：优先靠近进近航线的车站
    station = (top.get("stations") or [None])[0]
    if station:
        place = station["name"]
        res.evidence.append(Evidence(
            kind="knowledge", source="osint_geo", slot="landmark",
            value=place,
            claim=(
                f"机场 {top.get('iata')} 附近、与跑道平行铁路上的车站候选："
                f"{place}（距机场约 {station.get('dist_km')}km）"
            ),
            confidence=min(0.82, conf_loc),
            verified=True,
            raw={"lat": station.get("lat"), "lon": station.get("lon"),
                 "dist_km": station.get("dist_km")},
        ))
        # 拍摄地点槽：用 city 不够细，写入 landmark + 专用 photographer_location 文本
        res.evidence.append(Evidence(
            kind="knowledge", source="osint_geo", slot="city",
            value=place,
            claim=f"推断拍摄点邻近车站：{place}",
            confidence=min(0.8, conf_loc),
        ))

    # 历史航班
    if when:
        fl, kh, tr = await search_historical_flights(
            client,
            airport_iata=top.get("iata") or "",
            airport_icao=top.get("icao") or "",
            airport_name=(top.get("airport") or {}).get("name") or top.get("airport_label") or "",
            city=top.get("city") or "",
            when=when,
            cfg=cfg,
        )
        res.flights = fl
        res.knowledge.extend(kh)
        res.trace.extend(tr)
        for f in fl[:3]:
            cs = f.get("callsign") or ""
            sc = float(f.get("score") or 0.4)
            res.evidence.append(Evidence(
                kind="web" if f.get("source") == "web" else "knowledge",
                source=f"history:{f.get('source')}",
                slot="flight_number", value=cs,
                claim=(
                    f"在 {top.get('iata')} 于提示时刻窗口内检索到呼号/航班 {cs}"
                    + (f"（{clip(f.get('snippet') or '', 100)}）" if f.get("snippet") else "")
                ),
                confidence=sc,
                url=f.get("evidence_url"),
                raw=f,
            ))

        # 列车：网页检索 + 车站维基/走廊时刻匹配
        st_name = (station or {}).get("name") or top.get("city") or ""
        if st_name and constraints.get("viewpoint_vehicle") == "train":
            trn, kh2, tr2 = await search_train_services(
                client, st_name, top.get("city") or "", top.get("country") or "",
                when, cfg,
            )
            try:
                from .transit import enrich_train_candidates
                extra, tr3 = await enrich_train_candidates(
                    client, st_name, when, station_lon=(station or {}).get("lon"),
                )
                tr2.extend(tr3)
                # 合并，走廊匹配优先
                by = {re.sub(r"\s+", " ", (t.get("service") or "").upper()): t for t in trn}
                for t in extra:
                    key = re.sub(r"\s+", " ", (t.get("service") or "").upper())
                    if key not in by or float(t.get("score") or 0) > float(by[key].get("score") or 0):
                        by[key] = t
                trn = sorted(by.values(), key=lambda x: -float(x.get("score") or 0))
            except Exception as e:
                res.warnings.append(f"列车走廊匹配失败：{clip(str(e), 120)}")
            res.trains = trn
            res.knowledge.extend(kh2)
            res.trace.extend(tr2)
            for t in trn[:3]:
                svc = t.get("service") or ""
                res.evidence.append(Evidence(
                    kind="web", source=f"history:{t.get('source')}",
                    slot="photographer_ride", value=svc,
                    claim=t.get("why") or f"在车站 {st_name} 提示时刻附近检索到列车服务 {svc}",
                    confidence=float(t.get("score") or 0.45),
                    url=t.get("evidence_url"),
                    raw=t,
                ))
                res.evidence.append(Evidence(
                    kind="web", source=f"history:{t.get('source')}",
                    slot="object", value=f"train {svc}",
                    claim=f"拍摄者可能乘坐 {svc}",
                    confidence=float(t.get("score") or 0.4) * 0.9,
                ))
    else:
        res.warnings.append("提示中无拍摄时间，跳过历史航班/列车检索")

    try:
        res.knowledge.extend(await _web_rank_candidates(client, constraints, cfg))
    except Exception as e:
        res.warnings.append(f"约束网页检索失败：{clip(str(e), 100)}")

    return res
