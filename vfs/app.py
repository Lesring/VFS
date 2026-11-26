from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import CONFIG
from .models import AnalysisResult
from .pipeline import AnalysisJob
from .utils import ensure_dir, human_bytes, read_json, write_json

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

app = FastAPI(title="视频取证推理系统", version="0.1.0")

DATA = ensure_dir(CONFIG["paths"]["data_dir"])
UPLOADS = ensure_dir(CONFIG["paths"]["uploads_dir"])
OUTPUTS = ensure_dir(CONFIG["paths"]["outputs_dir"])

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    if (OUTPUTS).exists():
        app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")


# 内存中的任务状态：job_id -> {status, stage, message, queue, result_path, error}
JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _job_dir(job_id: str) -> Path:
    return ensure_dir(OUTPUTS / job_id)


def _safe_name(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w\u4e00-\u9fff.\-]+", "_", name)
    return name[:120] or "video.mp4"


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    page = STATIC / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>前端文件缺失</h1>", status_code=500)
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/api/config")
async def api_config() -> JSONResponse:
    from .models import SLOT_REGISTRY

    vlm = CONFIG["vlm"]
    return JSONResponse({
        "vlm_provider": vlm.get("provider"),
        "vlm_model": vlm.get("model"),
        "search_backend": CONFIG["retrieval"].get("search_backend"),
        "region_strategy": CONFIG["retrieval"].get("region_strategy"),
        "google_search_ready": bool(CONFIG["google"].get("api_key") and CONFIG["google"].get("cx")),
        "google_maps_ready": bool(CONFIG["google"].get("maps_api_key")),
        "ocr_enabled": CONFIG["ocr"].get("enabled"),
        "max_upload": CONFIG["server"]["max_upload_bytes"],
        "slots": {k: v["label"] for k, v in SLOT_REGISTRY.items()},
    })


@app.get("/api/settings")
async def get_settings() -> JSONResponse:
    """配置项描述 + 当前值（密钥打码）。供页面配置面板渲染。"""
    from .kb import health
    from .settings import schema

    vlm = CONFIG["vlm"]
    return JSONResponse({
        "fields": schema(),
        "status": {
            "vlm_active": f"{vlm.get('provider')}:{vlm.get('model')}",
            "vlm_ready": vlm.get("provider") != "mock",
            "region_strategy": CONFIG["retrieval"].get("region_strategy"),
        },
    })


@app.post("/api/settings")
async def save_settings(payload: dict[str, Any]) -> JSONResponse:
    """保存用户配置并热生效（无需重启）。"""
    from .settings import save

    result = save(payload or {})
    vlm = CONFIG["vlm"]
    return JSONResponse({
        **result,
        "vlm_active": f"{vlm.get('provider')}:{vlm.get('model')}",
        "vlm_ready": vlm.get("provider") != "mock",
    })


@app.get("/api/health/sources")
async def source_health(force: bool = False) -> JSONResponse:
    """探测各数据源可达性，用于展示"境外可用/已降级国内"。"""
    from .kb import health

    if force:
        health.reset()
    names = list(health.SOURCE_HOSTS.keys())
    # 并发探测，避免逐个等
    results = await asyncio.gather(*(asyncio.to_thread(health.check, n, None, force) for n in names))
    status = {n: {"ok": ok, **health.detail(n)} for n, ok in zip(names, results)}
    order = health.ordered_sources(CONFIG["retrieval"].get("region_strategy", "overseas_first"))
    return JSONResponse({
        "sources": status,
        "ordered": order,
        "strategy": CONFIG["retrieval"].get("region_strategy"),
        "degraded": not status.get("google", {}).get("ok", False),
    })


@app.post("/api/health/sources/refresh")
async def refresh_source_health() -> JSONResponse:
    from .kb import health

    health.reset()
    names = list(health.SOURCE_HOSTS.keys())
    await asyncio.gather(*(asyncio.to_thread(health.check, n, None, True) for n in names))
    return JSONResponse({"sources": health.snapshot()})


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    question: str = Form(""),
    hints: str = Form(""),
    online: bool = Form(True),
    max_frames: int | None = Form(None),
    provider: str | None = Form(None),
) -> JSONResponse:
    max_bytes = int(CONFIG["server"]["max_upload_bytes"])
    job_id = uuid.uuid4().hex[:12]
    dest = UPLOADS / f"{job_id}_{_safe_name(file.filename or 'video.mp4')}"

    size = 0
    with dest.open("wb") as fh:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"文件超过上限 {human_bytes(max_bytes)}")
            fh.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "上传的文件为空")

    cfg = CONFIG
    if provider:
        cfg = {**CONFIG, "vlm": {**CONFIG["vlm"], "provider": provider}}

    with _LOCK:
        JOBS[job_id] = {
            "status": "queued", "stage": "queued", "message": "排队中",
            "events": [], "event_lock": threading.Lock(),
            "result_path": None, "error": None,
            "created_at": time.time(), "question": question,
            "video": file.filename, "size": size,
        }

    def push(stage: str, message: str, data: dict[str, Any] | None = None) -> None:
        job = JOBS.get(job_id)
        if not job:
            return
        with job["event_lock"]:
            job["events"].append({"stage": stage, "message": message, "data": data, "t": time.time()})

    def work() -> None:
        job = JOBS[job_id]
        job["status"] = "running"
        try:
            runner = AnalysisJob(
                dest, question=question, hints=hints, job_id=job_id, config=cfg,
                progress=push, online=online, max_frames=max_frames,
            )
            result = runner.run()
            path = _job_dir(job_id) / "result.json"
            write_json(path, result.model_dump(mode="json"))
            job["result_path"] = str(path)
            job["status"] = "done"
            push("done", "分析完成", {"job_id": job_id})
        except Exception as e:  # noqa: BLE001
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
            push("error", job["error"], None)

    threading.Thread(target=work, name=f"job-{job_id}", daemon=True).start()
    return JSONResponse({"job_id": job_id, "status": "running", "size": size})


@app.post("/api/demo")
async def analyze_demo(
    question: str = Form(""),
    online: bool = Form(True),
) -> JSONResponse:
    """用内置演示视频跑一次完整分析，方便在没有任何模型 key 时体验系统。"""
    demo = Path(CONFIG["paths"]["data_dir"]) / "demo" / "airport_plane.mp4"
    if not demo.exists():
        raise HTTPException(
            404,
            "内置演示视频不存在。请先运行：python tools/make_demo_video.py",
        )
    job_id = uuid.uuid4().hex[:12]
    cfg = CONFIG
    with _LOCK:
        JOBS[job_id] = {
            "status": "queued", "stage": "queued", "message": "排队中",
            "events": [], "event_lock": threading.Lock(),
            "result_path": None, "error": None,
            "created_at": time.time(), "question": question,
            "video": demo.name, "size": demo.stat().st_size,
        }

    def push(stage: str, message: str, data: dict[str, Any] | None = None) -> None:
        job = JOBS.get(job_id)
        if not job:
            return
        with job["event_lock"]:
            job["events"].append({"stage": stage, "message": message, "data": data, "t": time.time()})

    def work() -> None:
        job = JOBS[job_id]
        job["status"] = "running"
        try:
            runner = AnalysisJob(demo, question=question, job_id=job_id, config=cfg,
                                 progress=push, online=online)
            result = runner.run()
            path = _job_dir(job_id) / "result.json"
            write_json(path, result.model_dump(mode="json"))
            job["result_path"] = str(path)
            job["status"] = "done"
            push("done", "分析完成", {"job_id": job_id})
        except Exception as e:  # noqa: BLE001
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
            push("error", job["error"], None)

    threading.Thread(target=work, name=f"job-{job_id}", daemon=True).start()
    return JSONResponse({"job_id": job_id, "status": "running", "size": job_size(job_id)})


def job_size(job_id: str) -> int:
    job = JOBS.get(job_id)
    return int(job["size"]) if job else 0


@app.get("/api/jobs/{job_id}/stream")
async def stream(job_id: str) -> StreamingResponse:
    if job_id not in JOBS:
        raise HTTPException(404, "任务不存在")

    async def gen():
        cursor = 0
        deadline = time.time() + 1800
        while time.time() < deadline:
            job = JOBS.get(job_id)
            if not job:
                break
            with job["event_lock"]:
                new = job["events"][cursor:]
                cursor += len(new)
            for ev in new:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if job["status"] in ("done", "error") and cursor >= len(job["events"]):
                break
            await asyncio.sleep(0.25)
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return JSONResponse({
        "job_id": job_id, "status": job["status"], "stage": job["stage"],
        "message": job["message"], "error": job["error"],
        "result_path": job["result_path"],
        "n_events": len(job["events"]),
    })


@app.get("/api/jobs/{job_id}/result")
async def job_result(job_id: str) -> JSONResponse:
    path = _job_dir(job_id) / "result.json"
    if not path.exists():
        if job_id in JOBS and JOBS[job_id]["status"] == "error":
            raise HTTPException(500, JOBS[job_id]["error"] or "分析失败")
        raise HTTPException(404, "结果尚未生成")
    return JSONResponse(read_json(path))


@app.get("/api/jobs")
async def list_jobs() -> JSONResponse:
    items = []
    for jid, job in sorted(JOBS.items(), key=lambda kv: -kv[1]["created_at"])[:40]:
        items.append({
            "job_id": jid, "status": job["status"], "video": job["video"],
            "question": job["question"], "created_at": job["created_at"],
            "size": job["size"],
        })
    return JSONResponse({"jobs": items})


@app.get("/api/map/{job_id}/map.png")
async def map_image(job_id: str) -> FileResponse:
    p = OUTPUTS / job_id / "map.png"
    if not p.exists():
        raise HTTPException(404, "地图尚未生成")
    return FileResponse(p, media_type="image/png")


@app.get("/api/frames/{job_id}/{frame_file}")
async def frame_file(job_id: str, frame_file: str) -> FileResponse:
    p = OUTPUTS / job_id / "frames" / _safe_name(frame_file)
    if not p.exists():
        raise HTTPException(404, "帧不存在")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/favicon.ico")
async def favicon() -> JSONResponse:
    # 返回 204 避免浏览器在控制台刷 404
    from fastapi import Response
    return Response(status_code=204)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True, "jobs": len(JOBS)})


def run(host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    uvicorn.run(
        "vfs.app:app",
        host=host or CONFIG["server"]["host"],
        port=int(port or CONFIG["server"]["port"]),
        reload=False,
    )
