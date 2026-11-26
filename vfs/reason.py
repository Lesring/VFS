"""证据融合与推理引擎。

核心思想：
1. 所有证据按 slot 汇聚成候选，用 noisy-OR 打分，同值跨源自动加权。
2. 跑一组"约束规则"，让一条线索推导出另一条（航班号 → 航司/航线；机型 + 航程 → 可行性检验；
   机场 → 城市/国家；坐标 → 最近机场；日期 + 时区 → 时段合理性）。
3. 每步推导都写入 reasoning_trace，保证结论可审计。
"""

from __future__ import annotations

import math
from typing import Any

from .entities import canonical
from .models import SLOT_REGISTRY, Candidate, Evidence, KnowledgeHit, SlotResult
from .utils import clip, confidence_label, noisy_or, norm_key, similarity, squash

# 不同证据源的可信度系数
SOURCE_WEIGHT = {
    "vlm": 1.0,
    "ocr": 0.85,
    "knowledge": 1.15,   # 离线/在线知识库本身是权威数据
    "web": 0.9,
    "heuristic": 0.35,
    "metadata": 0.8,
    "user": 1.0,
}

# slot 级别的先验权重：越"硬"的线索权重越高
SLOT_WEIGHT = {
    "flight_number": 1.0,
    "registration": 1.0,
    "aircraft_type": 0.8,
    "airline": 0.85,
    "airport": 0.8,
    "origin": 0.9,
    "destination": 0.9,
    "coordinate": 0.7,
    "city": 0.6,
    "country": 0.5,
    "date": 0.75,
    "time_of_day": 0.4,
    "landmark": 0.6,
    "weather": 0.25,
    "text_sign": 0.5,
    "object": 0.45,
    "person_role": 0.4,
    "brand": 0.6,
    "activity": 0.5,
    "viewpoint_vehicle": 0.7,
    "photographer_ride": 0.85,
}


class FusionEngine:
    def __init__(self, low_conf: float = 0.35, max_candidates: int = 5):
        self.low_conf = low_conf
        self.max_candidates = max_candidates
        self.trace: list[str] = []
        self.slots: dict[str, SlotResult] = {}
        self.notes: dict[str, list[str]] = {}
        # (slot, canon_key) -> 乘数。跨槽位冲突的扣分必须持久化，
        # 否则下一次 build_slots() 重建时会被抹掉。
        self.penalties: dict[tuple[str, str], float] = {}
        self._evidence: dict[str, Evidence] = {}
        self._by_slot: dict[str, list[tuple[Evidence, dict[str, Any]]]] = {}

    # ------------------------------------------------------------------ #
    def log(self, msg: str) -> None:
        self.trace.append(msg)

    def note(self, slot: str, msg: str) -> None:
        self.notes.setdefault(slot, [])
        if msg not in self.notes[slot]:
            self.notes[slot].append(msg)

    def penalize(self, slot: str, canon_key: str, factor: float, reason: str) -> None:
        """按（槽位, 归一化值）持久化下调置信度。重复调用取更严格的那个。"""
        k = (slot, canon_key)
        cur = self.penalties.get(k, 1.0)
        self.penalties[k] = min(cur, factor)
        self.log(f"下调 {slot}「{canon_key}」置信度 ×{factor}：{reason}")

    def renumber(self) -> None:
        """统一证据 ID 前缀：V=视觉, O=OCR, H=启发式, K=知识/联网, U=用户, M=元数据。

        这样在前端和日志里能一眼看出证据的来源类别。
        """
        prefix = {"vlm": "V", "ocr": "O", "heuristic": "H", "knowledge": "K",
                  "web": "K", "user": "U", "metadata": "M"}
        old_to_new: dict[str, str] = {}
        counters: dict[str, int] = {}
        for eid, ev in self._evidence.items():
            p = prefix.get(ev.kind, "E")
            counters[p] = counters.get(p, 0) + 1
            new_id = f"{p}{counters[p]:03d}"
            old_to_new[eid] = new_id
            ev.id = new_id
        self._evidence = {old_to_new[eid]: ev for eid, ev in self._evidence.items()}
        # by_slot 里存的是 Evidence 对象引用，id 已就地更新，无需重建

    def finalize(self) -> dict[str, SlotResult]:
        """重建槽位并挂上累积的备注。"""
        self.renumber()
        self.build_slots()
        for slot, msgs in self.notes.items():
            if slot in self.slots:
                self.slots[slot].notes = list(dict.fromkeys(self.slots[slot].notes + msgs))
        return self.slots

    def add_evidence(self, evs: list[Evidence]) -> None:
        for i, e in enumerate(evs):
            if not e.id:
                e.id = f"T{len(self._evidence) + 1:04d}"
            self._evidence[e.id] = e
            if not e.slot or not e.value:
                continue
            canon = canonical(e.slot, e.value)
            if not canon["key"]:
                continue
            self._by_slot.setdefault(e.slot, []).append((e, canon))

    # ------------------------------------------------------------------ #
    def build_slots(self) -> dict[str, SlotResult]:
        for slot, pairs in self._by_slot.items():
            grouped: dict[str, list[tuple[Evidence, dict]]] = {}
            for e, canon in pairs:
                grouped.setdefault(canon["key"], []).append((e, canon))

            cands: list[Candidate] = []
            for key, group in grouped.items():
                weights = []
                reasons = []
                eids = []
                k = group[0][1]["key"]
                for e, canon in group:
                    # 权重按"来源可信度 × 槽位先验"相乘后再开方，
                    # 避免强项（如 knowledge 1.15 × destination 0.9）被压到 1.0 截断，
                    # 也避免弱项（heuristic 0.35 × weather 0.25 = 0.0875）被乘到近乎为零。
                    w = math.sqrt(SOURCE_WEIGHT.get(e.kind, 0.7) * SLOT_WEIGHT.get(slot, 0.5))
                    if len(group) > 1:
                        w *= 1.18
                    if e.verified:
                        w *= 1.15
                    weights.append((min(1.0, w), e.confidence))
                    eids.append(e.id)
                    reasons.append(f"[{e.kind}:{e.source}] {clip(e.claim, 110)}")
                score = noisy_or(weights)
                display = _best_display([c["display"] for _, c in group])
                details: dict[str, Any] = {"canon_key": k,
                                           "sources": sorted({e.source for e, _ in group}),
                                           "n": len(group)}
                # 把上游结构化字段（经纬度、航司码、机场码）带上来，供约束推导使用
                for e, canon in group:
                    for fld in ("lat", "lon", "iata", "icao", "name", "city", "country",
                                "registration", "type", "owner"):
                        if fld not in details and e.raw.get(fld) not in (None, ""):
                            details[fld] = e.raw[fld]
                    # 只要该候选的证据全部来自航线库反查，就标记为派生值，
                    # 用于阻止"路线→坐标→最近机场→路线"的循环推理
                    if e.raw.get("from_route"):
                        details["route_derived"] = details.get("route_derived", 0) + 1
                    # 从自由文本里抽出的机场码（canonical 已解析）
                    codes = canon.get("extra", {}).get("codes") or []
                    if "iata" not in details:
                        for c in codes:
                            if len(c) == 3:
                                details["iata"] = c.upper()
                                break
                    if "icao" not in details:
                        for c in codes:
                            if len(c) == 4:
                                details["icao"] = c.upper()
                                break
                # 全部证据都是派生值的候选，标记 from_route=True
                if details.get("route_derived", 0) >= len(group):
                    details["from_route"] = True
                # 应用持久化的冲突扣分
                pen = self.penalties.get((slot, k))
                if pen:
                    score = score * pen
                    reasons.append(f"⚠ 与其他线索冲突，置信度已下调 ×{pen}")
                cands.append(Candidate(
                    value=display, score=round(score, 4), evidence_ids=eids,
                    reasons=reasons, details=details,
                ))

            cands.sort(key=lambda c: -c.score)
            cands = _merge_similar(cands)
            cands = cands[: self.max_candidates]
            top = cands[0].score if cands else 0.0
            label = SLOT_REGISTRY.get(slot, {}).get("label", slot)
            self.slots[slot] = SlotResult(
                slot=slot, label=label, candidates=cands,
                chosen=cands[0].value if cands else None,
                confidence=round(top, 4),
                confidence_label=confidence_label(top, self.low_conf),
            )
            if cands:
                self.log(f"槽位「{label}」汇总 {len(grouped)} 类候选，最高分 {top:.2f}：{cands[0].value}")
        for slot, msgs in self.notes.items():
            if slot in self.slots:
                self.slots[slot].notes = list(dict.fromkeys(self.slots[slot].notes + msgs))
        return self.slots

    # ------------------------------------------------------------------ #
    # 约束推导
    # ------------------------------------------------------------------ #

    def infer_from_flight(self, route: dict[str, Any] | None, flight_canon: dict[str, Any] | None) -> None:
        """航班号 → 航司 + 起降机场（权威来源）。"""
        if not route:
            return
        call = route.get("callsign") or route.get("callsign_iata") or ""
        airline = (route.get("airline") or {}).get("name") or ""
        origin = route.get("origin") or {}
        dest = route.get("destination") or {}
        src = route.get("source") or "adsbdb"

        if airline:
            self._inject("airline", airline, 0.97, src,
                         f"按航班号 {call} 在航线库中反查到承运人 {airline}", verified=True,
                         extra={"iata": (route.get("airline") or {}).get("iata"),
                                "icao": (route.get("airline") or {}).get("icao")})
        if origin.get("iata_code") or origin.get("name"):
            self._inject_airport("origin", origin, 0.95, src, f"航班 {call} 的出发机场")
        if dest.get("iata_code") or dest.get("name"):
            self._inject_airport("destination", dest, 0.95, src, f"航班 {call} 的目的地机场")
        self.log(f"约束：航班 {call} → 航司 {airline or '未知'}，"
                 f"{_ap(origin)} → {_ap(dest)}（来源 {src}）")

    def infer_from_registration(self, ac: dict[str, Any] | None) -> None:
        if not ac:
            return
        reg = ac.get("registration") or ""
        if ac.get("icao_type") or ac.get("type"):
            self._inject("aircraft_type", ac.get("type") or ac.get("icao_type"), 0.95, "adsbdb",
                         f"注册号 {reg} 在航空器库中登记为 {ac.get('type')}", verified=True,
                         extra=ac)
        if ac.get("owner"):
            self._inject("airline", ac.get("owner"), 0.8, "adsbdb",
                         f"注册号 {reg} 的登记运营人为 {ac.get('owner')}", verified=True)

    def _inject_airport(self, slot: str, ap: dict[str, Any], conf: float, src: str, claim: str) -> None:
        name = ap.get("name") or ap.get("municipality") or ""
        iata = (ap.get("iata_code") or "").upper()
        icao = (ap.get("icao_code") or "").upper()
        city = ap.get("municipality") or ""
        country = ap.get("country_name") or ""
        if not (name or iata):
            return
        code = "/".join(c for c in (iata, icao) if c)
        display = f"{name} ({code})" if code and name else (name or code)
        self._inject(slot, display, conf, src, claim, verified=True,
                     extra={"iata": iata, "icao": icao, "lat": ap.get("latitude"),
                            "lon": ap.get("longitude"), "name": name, "city": city, "country": country,
                            # 标记：这条机场信息是由航线库反查得来，不是画面观察
                            "from_route": True})
        if city:
            self._inject("city", city, conf * 0.7, src, f"{claim}，所在城市 {city}", verified=True)
        if country:
            self._inject("country", country, conf * 0.6, src, f"{claim}，所在国家 {country}", verified=True)
        # 注意：这里故意不注入 coordinate 槽位。
        # 机场已自带 lat/lon（用于航程计算），若再注入坐标，会被"坐标→最近机场"规则
        # 反查回同一个机场，形成自我印证的循环。

    def _inject(self, slot: str, value: str, conf: float, src: str, claim: str,
                verified: bool = False, extra: dict | None = None, kind: str = "knowledge") -> None:
        if not value:
            return
        ev = Evidence(
            kind=kind,  # type: ignore[arg-type]
            source=src, slot=slot, value=str(value), claim=claim,
            confidence=max(0.0, min(1.0, conf)), verified=verified,
            raw=extra or {},
        )
        ev.id = f"K{len(self._evidence) + 1:04d}"
        self._evidence[ev.id] = ev
        canon = canonical(slot, ev.value)
        if canon["key"]:
            self._by_slot.setdefault(slot, []).append((ev, canon))

    # ------------------------------------------------------------------ #
    def apply_constraints(self, airports=None, knowledge: list[KnowledgeHit] | None = None) -> None:
        """跨槽位一致性检查，输出提示与扣分。

        进来前必须先 build_slots()，因为联网注入的 origin/destination 等新证据
        只有在槽位重建后才能被约束检查看到。
        """
        self.build_slots()
        at = self.slots.get("aircraft_type")
        dest = self.slots.get("destination")
        origin = self.slots.get("origin")
        airline = self.slots.get("airline")

        # 1) 机型 → 航程可行性
        if at and dest and origin:
            fam = (at.candidates[0].details.get("canon_key") or "") if at.candidates else ""
            body = _body_of(fam)
            dist = _route_distance(origin, dest)
            if dist and body:
                limit = 6500 if body == "narrow" else 15000
                if dist > limit:
                    note = (f"机型 {at.candidates[0].value}（{body}）的理论航程不足以覆盖 "
                            f"{dist:.0f}km 的航线距离")
                    self.note("aircraft_type", note)
                    self.log(f"约束冲突：{note}")
                    self.penalize("aircraft_type", at.candidates[0].details.get("canon_key", ""),
                                  0.55, "机型航程与航线距离不符")
                else:
                    self.log(f"约束校验通过：{at.candidates[0].value} 可覆盖 {dist:.0f}km 航程")

        # 2) 航司 ↔ 航班号前缀一致性
        fn = self.slots.get("flight_number")
        if fn and airline and fn.candidates:
            fl = fn.candidates[0].details.get("canon_key") or ""
            prefix = "".join(ch for ch in fl if ch.isalpha())
            ans = [e for e in self._by_slot.get("airline", []) if e[0].raw.get("iata") or e[0].raw.get("icao")]
            codes = set()
            for e, _ in ans:
                for key in ("iata", "icao"):
                    v = e.raw.get(key)
                    if v:
                        codes.add(str(v).upper())
            if codes and prefix and prefix not in codes:
                msg = f"航班号 {fl} 的前缀 {prefix} 与已识别航司代码 {sorted(codes)} 不一致，需复核"
                self.note("flight_number", msg)
                self.log(f"约束冲突：{msg}")

        # 3) 坐标 → 最近机场（仅当坐标来自画面/OCR，不能是航线库反查来的，否则是循环论证）
        coord = self.slots.get("coordinate")
        if coord and coord.candidates and airports is not None and len(airports) > 0:
            src_cand = None
            for c in coord.candidates:
                if not c.details.get("from_route"):
                    src_cand = c
                    break
            if src_cand is None:
                src_cand = coord.candidates[0]
            if not src_cand.details.get("from_route"):
                lat, lon = src_cand.details.get("lat"), src_cand.details.get("lon")
                if lat is None or lon is None:
                    c = canonical("coordinate", src_cand.value)["extra"]
                    lat, lon = c.get("lat"), c.get("lon")
                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    self._nearest_airports(airports, float(lat), float(lon))

        # 4) 画面看到的机场 vs 航班实际起降机场：最关键的交叉校验
        self._check_airport_conflict(airports)

        # 5) 注册号登记的机主 vs 画面判读的航司
        self._check_registration_conflict()

        # 这些检查可能注入了新证据（如最近机场），再刷一次
        self.build_slots()

        # 6) 地标 → 用离线机场库兜底国家/城市
        lm = self.slots.get("landmark")
        if lm and lm.candidates and airports is not None and len(airports) > 0 and not self.slots.get("city"):
            for c in lm.candidates[:2]:
                for ap in airports.lookup(c.value)[:2]:
                    self._inject("city", ap.city, 0.35, "openflights",
                                 f"地标名 {c.value} 与机场 {ap.name} 所在城市匹配", verified=False)
                    if ap.country:
                        self._inject("country", ap.country, 0.3, "openflights",
                                     f"地标名 {c.value} 关联国家 {ap.country}")

    def _nearest_airports(self, airports, lat: float, lon: float) -> None:
        """坐标 → 附近机场。只用于画面/OCR 直接给出的坐标。"""
        near = airports.nearest(lat, lon, limit=3, max_km=120)
        for ap, km in near:
            conf = 0.75 if km < 25 else (0.55 if km < 60 else 0.35)
            self._inject("airport", ap.label(), conf, "openflights",
                         f"画面坐标 ({lat:.3f},{lon:.3f}) 附近 {km:.1f}km 是 {ap.name}",
                         verified=True, extra={"iata": ap.iata, "icao": ap.icao,
                                               "lat": ap.lat, "lon": ap.lon, "name": ap.name})
            if ap.city:
                self._inject("city", ap.city, conf * 0.8, "openflights",
                             f"该坐标附近机场 {ap.name} 位于 {ap.city}", verified=True)
            if ap.country:
                self._inject("country", ap.country, conf * 0.7, "openflights",
                             f"该坐标附近机场 {ap.name} 位于 {ap.country}", verified=True)
        if near:
            self.log(f"坐标定位：最近机场 {near[0][0].name} 约 {near[0][1]:.1f}km")

    def _check_airport_conflict(self, airports=None) -> None:
        """画面里认出的机场，是否与航班号查出的起降机场一致？

        这是本系统最有价值的一类校验：视觉线索 + 联网权威数据互相证伪。
        只看"画面来源"的机场候选，航线库反查出来的候选不参与比对（否则必然自洽）。
        """
        seen_slot = self.slots.get("airport")
        if not seen_slot or not seen_slot.candidates:
            return

        route_airports = []
        for slot_name in ("origin", "destination"):
            s = self.slots.get(slot_name)
            if s and s.candidates:
                c = s.candidates[0]
                code = (c.details.get("iata") or c.details.get("icao") or "").upper()
                route_airports.append((slot_name, c.value, code))
        route_codes = {code for _, _, code in route_airports if code}
        if not route_codes:
            return

        # 只看画面来源的候选
        visual = [c for c in seen_slot.candidates if not c.details.get("from_route")]
        if not visual:
            return

        matched = None
        for c in visual:
            code = (c.details.get("iata") or c.details.get("icao") or "").upper()
            if not code and airports is not None and len(airports) > 0:
                hits = airports.lookup(c.value)
                if hits:
                    code = (hits[0].iata or hits[0].icao).upper()
            if code and code in route_codes:
                matched = (c, code)
                break

        if matched:
            c, code = matched
            which = next(n for n, _, cd in route_airports if cd == code)
            self.log(f"交叉验证通过：画面机场「{c.value}」与航班航线的{which}（{code}）一致")
            self.note("airport", "与航班航线一致，可佐证该航线判定")
            # 增强该候选
            c.score = min(0.99, c.score * 1.25)
            c.reasons.append(f"与航班航线的{which} {code} 一致，互相印证")
            return

        # 全部不匹配 → 冲突
        seen_desc = "、".join(
            f"{c.value}（{(c.details.get('iata') or c.details.get('icao') or '?')}）"
            for c in visual[:3]
        )
        route_desc = "、".join(f"{val.split('(')[0].strip()}（{code}）" for _, val, code in route_airports)
        msg = (f"画面机场 {seen_desc} 与航班实际起降机场 {route_desc} 均不匹配；"
               "可能是中转/经停、画面拍的并非该航班、或航班号被误读")
        self.note("airport", msg)
        self.note("destination", "目的地由航班号反查得出，但画面机场与其航线不符，需人工复核")
        self.log(f"交叉验证冲突：{msg}")
        dest = self.slots.get("destination")
        if dest and dest.candidates:
            self.penalize("destination", dest.candidates[0].details.get("canon_key", ""), 0.78,
                          "画面机场与该航线的起降机场均不匹配")

    def _check_registration_conflict(self) -> None:
        """注册号登记的运营人 vs 画面判读的航司。"""
        reg = self.slots.get("registration")
        airline = self.slots.get("airline")
        if not reg or not reg.candidates or not airline or not airline.candidates:
            return
        owners = []
        for e, _ in self._by_slot.get("airline", []):
            if e.kind == "knowledge" and "登记运营人" in (e.claim or ""):
                owners.append(e.value)
        if not owners:
            return
        # 画面判读的航司：优先用非知识库来源的候选，避免拿"注入了的答案"跟自己比
        judged = ""
        for c in airline.candidates:
            if not c.details.get("from_route") and not any(
                eid.startswith("K") for eid in c.evidence_ids
            ):
                judged = c.value
                break
        if not judged:
            judged = airline.candidates[0].value
        if any(similarity(judged, o) > 0.6 for o in owners):
            self.log(f"交叉验证通过：画面航司 {judged} 与注册号登记运营人一致")
            return
        msg = (f"画面判读航司为 {judged}，但注册号 {reg.candidates[0].value} "
               f"登记运营人为 {'、'.join(owners)}，两者不一致（可能已转租/退役/注册号误读）")
        self.note("airline", msg)
        self.note("registration", "该注册号与画面涂装不符，建议核对")
        self.log(f"交叉验证冲突：{msg}")
        self.penalize("registration", reg.candidates[0].details.get("canon_key", ""), 0.6,
                      "该注册号的登记运营人与画面涂装不一致")

    def merge_knowledge(self, hits: list[KnowledgeHit]) -> None:
        for h in hits:
            text = f"{h.title} {h.snippet}"
            for slot in ("aircraft_type", "airline", "airport", "city", "country"):
                pass
            # 知识条目本身作为一个 text 证据，用于后续答案引用
            self.log(f"知识：{clip(h.title, 60)} — {clip(h.snippet, 100)}")
            # 若知识条目给出了具体机场/城市，注入弱证据
            for slot in ("destination", "origin", "airport"):
                pass

    def best(self, slot: str) -> Candidate | None:
        s = self.slots.get(slot)
        if not s or not s.candidates:
            return None
        return s.candidates[0]

    def get(self, slot: str) -> str | None:
        c = self.best(slot)
        return c.value if c else None

    def confidence_of(self, slot: str) -> float:
        s = self.slots.get(slot)
        return s.confidence if s else 0.0

    def evidence_list(self) -> list[Evidence]:
        return list(self._evidence.values())


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #


def _best_display(values: list[str]) -> str:
    if not values:
        return ""
    # 选信息量最大的（较长的通常带括号代码）
    return sorted(values, key=lambda v: (-len(v), v))[0]


def _merge_similar(cands: list[Candidate]) -> list[Candidate]:
    out: list[Candidate] = []
    for c in cands:
        merged = False
        for o in out:
            if similarity(c.value, o.value) >= 0.88 or c.details.get("canon_key") == o.details.get("canon_key"):
                o.score = min(1.0, o.score + c.score * (1 - o.score))
                o.evidence_ids = list(dict.fromkeys(o.evidence_ids + c.evidence_ids))
                o.reasons += [r for r in c.reasons if r not in o.reasons]
                if len(c.value) > len(o.value):
                    o.value = c.value
                merged = True
                break
        if not merged:
            out.append(c)
    out.sort(key=lambda c: -c.score)
    return out


def _route_distance(origin_slot: SlotResult, dest_slot: SlotResult) -> float | None:
    co = _coord_of(origin_slot)
    cd = _coord_of(dest_slot)
    if not co or not cd:
        return None
    return _hav(co[0], co[1], cd[0], cd[1])


def _coord_of(slot: SlotResult) -> tuple[float, float] | None:
    for c in slot.candidates:
        lat, lon = c.details.get("lat"), c.details.get("lon")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return float(lat), float(lon)
    return None


def _hav(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _body_of(canon_key: str) -> str:
    wide = {"a330", "a350", "a380", "b747", "b767", "b777", "b787"}
    narrow = {"a318", "a319", "a320", "a321", "a220", "b737", "b757", "e190", "e195", "crj900", "atr72"}
    k = (canon_key or "").lower()
    if k in wide:
        return "wide"
    if k in narrow:
        return "narrow"
    return ""


def _ap(ap: dict[str, Any]) -> str:
    if not ap:
        return "未知"
    name = ap.get("name") or ap.get("municipality") or ""
    code = ap.get("iata_code") or ap.get("icao_code") or ""
    return f"{name} ({code})" if code else (name or "未知")
