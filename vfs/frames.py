from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .models import FrameInfo, VideoMeta
from .utils import ensure_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class FrameSample:
    index: int
    timestamp_s: float
    image: np.ndarray
    sharpness: float
    brightness: float
    scene_change: float
    reason: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 探测
# --------------------------------------------------------------------------- #


def probe_video(path: str | Path) -> VideoMeta:
    p = Path(path)
    meta = VideoMeta(filename=p.name, path=str(p), size_bytes=p.stat().st_size if p.exists() else 0)
    if p.suffix.lower() in IMAGE_EXTS:
        with Image.open(p) as im:
            meta.width, meta.height = im.size
        meta.duration_s, meta.fps, meta.codec = 0.0, 0.0, "image"
        return meta

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return meta
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        meta.fps = round(fps, 3) if fps and not math.isnan(fps) else 0.0
        if fps and fps > 0 and count > 0:
            meta.duration_s = round(count / fps, 3)
        meta.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        meta.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        if fourcc:
            meta.codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
        meta.container_meta["opencv_backend"] = cap.getBackendName() if hasattr(cap, "getBackendName") else ""
    finally:
        cap.release()
    return meta


# --------------------------------------------------------------------------- #
# 抽帧
# --------------------------------------------------------------------------- #


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _scene_delta(prev_small: np.ndarray | None, small: np.ndarray) -> float:
    if prev_small is None:
        return 1.0
    diff = cv2.absdiff(prev_small, small)
    return float(np.mean(diff) / 255.0)


def _hist_dist(prev_hist: np.ndarray | None, hist: np.ndarray) -> float:
    if prev_hist is None:
        return 1.0
    return float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))


def extract_frames(
    path: str | Path,
    out_dir: str | Path,
    max_frames: int = 24,
    min_gap_s: float = 0.35,
    scene_threshold: float = 0.32,
    uniform_ratio: float = 0.35,
    blur_reject: float = 28.0,
) -> tuple[list[FrameInfo], VideoMeta]:
    """返回 (关键帧列表, 视频元数据)。策略：场景切换 + 清晰度 + 均匀兜底。"""
    p = Path(path)
    out_dir = ensure_dir(out_dir)
    meta = probe_video(p)

    if p.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(p))
        if img is None:
            return [], meta
        fi = _save_frame(img, out_dir, 0, 0.0, "输入为静态图片")
        print_meta = meta
        return [fi], print_meta

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"无法解码视频：{p.name}（可能是编码不受支持，建议转成 H.264 MP4）")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps / 4)))  # 每秒采样约 4 帧做候选
        min_gap_frames = max(1, int(round(min_gap_s * fps)))

        candidates: list[FrameSample] = []
        prev_small = None
        prev_hist = None
        idx = 0
        last_kept = -10**9

        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step != 0:
                idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                idx += 1
                continue
            t = idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            sharp = _sharpness(gray)
            bright = float(np.mean(gray))
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            delta = max(_scene_delta(prev_small, small), _hist_dist(prev_hist, hist))
            prev_small, prev_hist = small, hist

            candidates.append(
                FrameSample(idx, round(t, 2), frame.copy(), sharp, bright, delta, "候选")
            )
            idx += 1
    finally:
        cap.release()

    if not candidates:
        return [], meta

    # 1) 场景切换点
    keep: list[FrameSample] = []
    for c in candidates:
        is_cut = c.scene_change >= scene_threshold
        clear_enough = c.sharpness >= blur_reject
        far_enough = c.index - (keep[-1].index if keep else -10**9) >= min_gap_frames
        if is_cut and clear_enough and far_enough:
            c.reason = "场景切换"
            keep.append(c)

    # 2) 均匀兜底，避免长镜头只有一个视角
    target = max(1, max_frames)
    n_uniform = int(target * uniform_ratio)
    span = len(candidates)
    for i in range(n_uniform):
        pos = int((i + 0.5) / max(1, n_uniform) * span)
        pos = min(span - 1, max(0, pos))
        c = candidates[pos]
        if all(abs(c.index - k.index) >= min_gap_frames for k in keep):
            c2 = FrameSample(c.index, c.timestamp_s, c.image, c.sharpness, c.brightness, c.scene_change, "均匀采样")
            keep.append(c2)

    # 3) 场景帧不够时，按"清晰 × 变化"补位
    if len(keep) < min(target, span):
        rest = [c for c in candidates if all(abs(c.index - k.index) >= min_gap_frames for k in keep)]
        rest.sort(key=lambda c: c.sharpness * (0.4 + c.scene_change), reverse=True)
        for c in rest:
            if len(keep) >= target:
                break
            if c.sharpness < blur_reject * 0.5:
                continue
            c.reason = "清晰度补位"
            keep.append(c)

    keep.sort(key=lambda c: c.index)
    if len(keep) > max_frames:
        # 均匀抽稀保留
        sel = np.linspace(0, len(keep) - 1, max_frames).round().astype(int)
        keep = [keep[i] for i in sorted(set(sel))]

    frames = [_save_frame(c.image, out_dir, i, c.timestamp_s, c.reason, c.sharpness, c.brightness, c.scene_change)
              for i, c in enumerate(keep)]
    return frames, meta


def _save_frame(
    img: np.ndarray,
    out_dir: Path,
    index: int,
    ts: float,
    reason: str,
    sharp: float = 0.0,
    bright: float = 0.0,
    delta: float = 0.0,
) -> FrameInfo:
    h, w = img.shape[:2]
    fid = f"f{index:03d}_{ts:07.2f}s".replace(".", "p")
    path = out_dir / f"{fid}.jpg"
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return FrameInfo(
        id=fid, index=index, timestamp_s=ts, path=str(path),
        width=w, height=h, sharpness=round(sharp, 1),
        brightness=round(bright, 1), scene_change=round(delta, 3), reason=reason,
    )


# --------------------------------------------------------------------------- #
# 编码给模型
# --------------------------------------------------------------------------- #


def frame_to_data_url(path: str | Path, max_side: int = 1280, quality: int = 85) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_side:
            scale = max_side / max(im.size)
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def frame_bytes(path: str | Path, max_side: int = 1280, quality: int = 85) -> bytes:
    data_url = frame_to_data_url(path, max_side, quality)
    return base64.b64decode(data_url.split(",", 1)[1])


def make_contact_sheet(paths: list[str | Path], out_path: str | Path, cols: int = 3, cell: int = 480) -> str | None:
    """把多帧拼成一张总览图，供模型做跨帧综合。"""
    if not paths:
        return None
    imgs = []
    for p in paths:
        try:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
        except Exception:
            continue
    if not imgs:
        return None
    cols = min(cols, len(imgs))
    rows = math.ceil(len(imgs) / cols)
    sheet = Image.new("RGB", (cols * cell, rows * cell), (18, 18, 18))
    for i, im in enumerate(imgs):
        im.thumbnail((cell, cell))
        x = (i % cols) * cell + (cell - im.width) // 2
        y = (i // cols) * cell + (cell - im.height) // 2
        sheet.paste(im, (x, y))
    out = Path(out_path)
    ensure_dir(out.parent)
    sheet.save(out, format="JPEG", quality=88)
    return str(out)
