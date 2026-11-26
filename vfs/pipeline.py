"""编排层：视频 → 帧 → 视觉/OCR/启发式 → 融合 → 联网核实 → 再融合 → 答案。"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from . import answer as answer_mod
from . import frames as frames_mod
from . import heuristics, ocr
from .config import CONFIG
from .entities import canonical
from .intent import parse_intent
from .kb.airports import load_airports
from .kb.flights import load_airlines
from .kb.http import CachedClient
from .models import (
    AnalysisResult, Evidence, KnowledgeHit, SlotResult, StageTiming, VideoMeta,
)
from .reason import FusionEngine
from .retrieval import build_map, run_retrieval
from .utils import clip, confidence_label, ensure_dir
from .vlm.base import get_vlm

ProgressFn = Callable[[str, str, dict[str, Any] | None], None]


def _noop(stage: str, msg: str, data: dict[str, Any] | None = None) -> None:
    return None


class AnalysisJob:
    """一次分析任务。可同步跑，也可通过 progress 回调推送阶段进度。"""

    def __init__(self, video_path: str | Path, question: str = "", hints: str = "",
                 job_id: str | None = None, config: dict[str, Any] | None = None,
                 progress: ProgressFn | None = None, online: bool = True,
                 max_frames: int | None = None):
        self.cfg = config or CONFIG
        self.job_id = job_id or uuid.uuid4().hex[:12]
        self.video_path = Path(video_path)
        self.question = question or ""
        self.hints = hints or ""
        self.progress = progress or _noop
        self.online = online
        self.max_frames = max_frames
        self.timings: list[StageTiming] = []
        self.warnings: list[str] = []
        out_root = Path(self.cfg["paths"]["outputs_dir"])
        self.out_dir = ensure_dir(out_root / self.job_id)
        self.frame_dir = ensure_dir(self.out_dir / "frames")
        self._carry_sidecar()
        self.client = CachedClient(
            cache_dir=self.cfg["retrieval"]["cache_dir"],
            ttl_s=int(self.cfg["retrieval"]["cache_ttl_s"]),
            timeout_s=float(self.cfg["retrieval"]["timeout_s"]),
            user_agent=self.cfg["retrieval"]["user_agent"],
        )

    def _carry_sidecar(self) -> None:
        """把视频同目录的 observations.json 旁挂真值复制到任务目录。

        仅在 mock VLM 下生效：这是给测试/演示用的"预录观察"，
        让没有 API key 的人也能验证推理链本身。真实模型路径完全绕开它。
        """
        if str(self.cfg["vlm"].get("provider", "")).lower() != "mock":
            return
        sidecar_allowed = self.cfg["vlm"].get("allow_sidecar", True)
        if not sidecar_allowed:
            return
        src_dir = self.video_path.parent
        candidates = [
            src_dir / f"{self.video_path.stem}.observations.json",
            src_dir / "observations.json",
        ]
        for cand in candidates:
            if cand.exists():
                try:
                    (self.out_dir / "observations.json").write_text(
                        cand.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                    self.warnings.append("检测到旁挂观察文件，视觉证据来自该文件（非真实模型输出）")
                except Exception:
                    pass
                break

    # ------------------------------------------------------------------ #
    def _stage(self, name: str, t0: float, note: str = "") -> None:
        self.timings.append(StageTiming(stage=name, ms=int((time.time() - t0) * 1000), note=note))

    def run(self) -> AnalysisResult:
        result = AnalysisResult(job_id=self.job_id, question=self.question, hints=self.hints)
        result.warnings = self.warnings

        # ---------- 1. 抽帧 ----------
        self.progress("frames", "正在解码视频并抽取关键帧…", None)
        t0 = time.time()
        fcfg = self.cfg["frames"]
        max_frames = int(self.max_frames or fcfg["max_frames"])
        try:
            frames, meta = frames_mod.extract_frames(
                self.video_path, self.frame_dir,
                max_frames=max_frames,
                min_gap_s=float(fcfg["min_gap_s"]),
                scene_threshold=float(fcfg["scene_threshold"]),
                uniform_ratio=float(fcfg["uniform_ratio"]),
                blur_reject=float(fcfg["blur_reject"]),
            )
        except Exception as e:
            self.warnings.append(f"抽帧失败：{e}")
            frames, meta = [], VideoMeta(filename=self.video_path.name, path=str(self.video_path))
        result.video = meta
        result.frames = frames
        self._stage("抽帧", t0, f"{len(frames)} 帧")
        self.progress("frames", f"抽出 {len(frames)} 个关键帧", {"frames": [f.model_dump() for f in frames]})
        if not frames:
            result.answer = "**无法解码该视频。** 请确认格式受支持（推荐 H.264 编码的 MP4），或先转码后重试。"
            result.answer_confidence_label = "无法判定"
            result.timeline = self.timings
            return result

        # ---------- 2. 意图 ----------
        t0 = time.time()
        intent = parse_intent(self.question)
        result.intent = intent
        self._stage("理解问题", t0, intent["primary_label"])

        engine = FusionEngine(
            low_conf=float(self.cfg["reasoning"]["low_confidence"]),
            max_candidates=int(self.cfg["reasoning"]["max_candidates"]),
        )

        # ---------- 3. 视觉模型 ----------
        self.progress("vision", "视觉模型正在读帧…", None)
        t0 = time.time()
        vlm = get_vlm(self.cfg["vlm"])
        batch = None
        try:
            batch = vlm.analyze_frames(frames, self.question, self.hints)
        except Exception as e:
            self.warnings.append(f"视觉模型调用失败：{clip(str(e), 240)}")
        vlm_evidence: list[Evidence] = []
        if batch:
            for ob in batch.observations:
                vlm_evidence.append(Evidence(
                    kind="vlm", source=f"{batch.provider}:{batch.model}",
                    claim=ob.note or f"画面显示 {ob.slot} 为 {ob.value}",
                    slot=ob.slot, value=ob.value,
                    confidence=max(0.0, min(1.0, ob.confidence)),
                    timestamp_s=ob.timestamp_s, frame_id=ob.frame_id,
                    raw=ob.raw,
                ))
            self.warnings.extend(batch.warnings)
            self.progress("vision", f"视觉模型返回 {len(vlm_evidence)} 条观察",
                          {"observations": [e.model_dump() for e in vlm_evidence]})
        self._stage("视觉分析", t0, f"{len(vlm_evidence)} 条观察")

        # ---------- 4. 启发式 ----------
        t0 = time.time()
        heur_ev, heur_summary = heuristics.analyze_frames(frames)
        self._stage("图像统计", t0, f"{len(heur_ev)} 条")
        result.reasoning_trace.append(f"本地图像统计：{heur_summary.get('mean_brightness')} 亮度，"
                                     f"天空占比 {heur_summary.get('sky_ratio')}")

        # ---------- 5. OCR ----------
        t0 = time.time()
        ocr_ev, ocr_stats = ocr.run_ocr(frames, self.cfg["ocr"])
        if ocr_stats.get("skipped"):
            self.progress("ocr", f"OCR 跳过（{ocr_stats['skipped']}）", None)
        else:
            self.progress("ocr", f"OCR 命中 {ocr_stats.get('hits', 0)} 帧", None)
        self._stage("OCR", t0, f"{len(ocr_ev)} 条")

        # ---------- 6. 首轮融合 ----------
        t0 = time.time()
        engine.add_evidence(vlm_evidence)
        engine.add_evidence(ocr_ev)
        engine.add_evidence(heur_ev)
        if self.hints:
            engine.add_evidence([Evidence(
                kind="user", source="user_hint", slot="date", value=self.hints,
                claim=f"用户提示：{self.hints}", confidence=0.55,
            )])
        engine.build_slots()
        self._stage("首轮融合", t0, f"{len(engine.slots)} 个槽位")
        self.progress("fuse", "初步实体已汇总", {"slots": _slots_brief(engine.slots)})

        # ---------- 7. 联网核实 ----------
        knowledge: list[KnowledgeHit] = []
        retrieval_trace: list[str] = []
        if self.online:
            self.progress("retrieval", "联网核实中…", None)
            t0 = time.time()
            try:
                airports_db = await_or_run(load_airports(self.client, self.cfg["retrieval"]["cache_dir"]))
                airlines_db = await_or_run(load_airlines(self.client, self.cfg["retrieval"]["cache_dir"]))
            except Exception as e:
                airports_db, airlines_db = None, None
                self.warnings.append(f"知识库加载失败：{clip(str(e), 160)}")

            rr = await_or_run(run_retrieval(
                engine.slots, self.client, airlines_db, airports_db,
                self.cfg["retrieval"], self.question,
            ))
            engine.infer_from_flight(rr.route, None)
            engine.infer_from_registration(rr.aircraft)
            knowledge = rr.knowledge
            retrieval_trace = rr.trace
            result.searches = rr.searches
            self.warnings.extend(rr.warnings)
            self._stage("联网核实", t0, f"{len(knowledge)} 条知识 / {len(rr.searches)} 次搜索")
            self.progress("retrieval", f"联网返回 {len(knowledge)} 条佐证", None)

            # 二次融合（加入了权威数据）
            t0 = time.time()
            try:
                engine.apply_constraints(airports=airports_db, knowledge=knowledge)
            except Exception as e:
                self.warnings.append(f"约束检查异常：{clip(str(e), 160)}")
            engine.finalize()
            self._stage("二次融合", t0, "")

            # ---------- 7.4 OSINT 地理约束推理（Planes&Trains 风格）----------
            try:
                from .osint import run_osint_geolocation
                self.progress("osint", "地理约束推理中…", None)
                t0 = time.time()
                osint = await_or_run(run_osint_geolocation(
                    client=self.client,
                    airports=airports_db,
                    slots=engine.slots,
                    frames=frames,
                    question=self.question,
                    hints=self.hints,
                    cfg=self.cfg["retrieval"],
                    vlm_cfg=self.cfg.get("vlm"),
                    out_dir=str(self.out_dir),
                ))
                if osint.enabled:
                    engine.add_evidence(osint.evidence)
                    knowledge.extend(osint.knowledge)
                    retrieval_trace.extend(osint.trace)
                    self.warnings.extend(osint.warnings)
                    # 若 OSINT 给出航班号，再查一次航线库
                    for ev in osint.evidence:
                        if ev.slot == "flight_number" and ev.value and ev.confidence >= 0.4:
                            try:
                                route = await_or_run(
                                    __import__("vfs.kb.flights", fromlist=["route_by_callsign"])
                                    .route_by_callsign(self.client, ev.value)
                                )
                                if route:
                                    engine.infer_from_flight(route, None)
                            except Exception:
                                pass
                    engine.finalize()
                    try:
                        engine.apply_constraints(airports=airports_db, knowledge=knowledge)
                    except Exception:
                        pass
                    engine.finalize()
                    self._stage("OSINT地理推理", t0, f"{len(osint.candidates)} 候选")
                    self.progress(
                        "osint",
                        f"OSINT 完成：{len(osint.candidates)} 个地点候选 / "
                        f"{len(osint.flights)} 航班 / {len(osint.trains)} 列车",
                        {"candidates": osint.candidates[:5]},
                    )
                else:
                    self._stage("OSINT地理推理", t0, "未触发")
            except Exception as e:
                self.warnings.append(f"OSINT 地理推理异常：{clip(str(e), 200)}")
        else:
            airports_db = None
            self.warnings.append("本次为离线模式，未做联网核实")

        # 关键词类知识注入：让"搜索到的机场/机型"进入槽位（弱证据）
        _inject_knowledge_hits(engine, knowledge, airports_db)
        engine.finalize()
        result.slots = engine.slots
        result.knowledge = knowledge
        result.evidence = engine.evidence_list()
        for t in retrieval_trace:
            if t not in result.reasoning_trace:
                result.reasoning_trace.append(t)

        # ---------- 7.5 地图可视化 ----------
        if self.online:
            t0 = time.time()
            try:
                m = await_or_run(build_map(
                    self.client, engine.slots, self.cfg.get("google", {}),
                    job_id=self.job_id, output_dir=str(self.out_dir),
                ))
                if m:
                    result.map = m
                    result.reasoning_trace.append(
                        f"地图定位：{m.get('label')}（{m.get('provider')}）"
                    )
                self._stage("地图", t0, (m or {}).get("provider", "无"))
                self.progress("map", f"地图：{m.get('provider')}" if m else "未生成地图",
                              {"map": m})
            except Exception as e:
                self.warnings.append(f"地图生成失败：{clip(str(e), 160)}")

        # ---------- 8. 生成答案 ----------
        t0 = time.time()
        text, conf, _ = answer_mod.build_answer(
            engine, intent, self.question, frames, result.evidence, knowledge,
            self.warnings, allow_abstain=bool(self.cfg["reasoning"]["allow_abstain"]),
        )
        result.answer = text
        result.answer_confidence = round(conf, 4)
        result.answer_confidence_label = confidence_label(conf, engine.low_conf)
        self._stage("生成答案", t0)

        # 合并 trace（引擎的推导日志放在前面更易读）
        result.reasoning_trace = list(dict.fromkeys(engine.trace + result.reasoning_trace))
        result.timeline = self.timings
        result.warnings = list(dict.fromkeys(self.warnings))
        return result


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #


def await_or_run(coro):
    """允许在已有事件循环或没有事件循环的环境下运行协程。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # 在同一个线程里已有 loop（如 FastAPI async 上下文）时用新线程跑
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _slots_brief(slots: dict[str, SlotResult]) -> list[dict[str, Any]]:
    out = []
    for s in slots.values():
        if s.candidates:
            out.append({"slot": s.slot, "label": s.label, "value": s.candidates[0].value,
                        "score": s.candidates[0].score})
    return out


def _inject_knowledge_hits(engine: FusionEngine, hits: list[KnowledgeHit], airports_db) -> None:
    """把机场库/百科里的强匹配补充为弱证据，避免搜索结果的自由文本污染。"""
    if not hits:
        return
    for h in hits:
        if h.source != "openflights":
            continue
        data = h.data or {}
        iata = data.get("iata") or ""
        icao = data.get("icao") or ""
        if not (iata or icao):
            continue
        city = data.get("city") or ""
        country = data.get("country") or ""
        if city:
            engine._inject("city", city, 0.55, "openflights",
                           f"机场库：{data.get('name')} 位于 {city}")
        if country:
            engine._inject("country", country, 0.5, "openflights",
                           f"机场库：{data.get('name')} 位于 {country}")
