"""Turn public driver-monitoring videos into the browser's per-frame signals.

Each video is streamed with ffmpeg (only the needed segments are fetched), run
through the same MediaPipe Face Landmarker model file and options the app uses,
and reduced to numeric signals with ml/signals.py. Frames are discarded as soon
as they are measured; only numbers and, for DMD, 32x32 eye crops are kept.

    python ml/extract_signals.py dmd   --workers 8
    python ml/extract_signals.py rldd  --workers 12 --segments 5 --seconds 40

Datasets (research use, cite the authors):
  * DMD   - Ortega et al., "DMD: A Large-Scale Multi-Modal Driver Monitoring
            Dataset for Attention and Alertness Analysis", ECCV Workshops 2020.
  * RLDD  - Ghoddoosian et al., "A Realistic Dataset and Baseline Temporal Model
            for Early Drowsiness Detection", CVPR Workshops 2019.
Both are read from the public Hugging Face mirror nyvantran/drowsy_dataset.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signals import BLENDSHAPE_NAMES, EYE_CROP_SIZE, eye_crop_transform, face_signals, to_gray  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "public" / "models" / "face_landmarker.task"
MIRROR = "https://huggingface.co/datasets/nyvantran/drowsy_dataset/resolve/main/"
DATA = ROOT / "ml" / "data"


def url_for(path: str) -> str:
    return MIRROR + urllib.parse.quote(path)


def _ffmpeg() -> str:
    # A pinned modern static build: system ffmpeg builds vary widely and older
    # ones neither seek over HTTPS reliably nor apply phone rotation metadata.
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG = _ffmpeg()


def probe(url: str):
    """Display width, height, frame rate, and duration, read from ffmpeg's header dump."""
    text = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", url], capture_output=True, text=True, timeout=180
    ).stderr
    video = next(line for line in text.splitlines() if "Video:" in line)
    width, height = map(int, re.search(r"\b(\d{2,5})x(\d{2,5})\b", video).groups())
    fps_match = re.search(r"([\d.]+) fps", video)
    fps = float(fps_match.group(1)) if fps_match else 30.0
    rotation = re.search(r"rotation of (-?[\d.]+) degrees|rotate\s*:\s*(-?\d+)", text)
    if rotation and abs(int(float(rotation.group(1) or rotation.group(2)))) % 180 == 90:
        width, height = height, width
    hours, minutes, seconds = re.search(r"Duration: (\d+):(\d+):([\d.]+)", text).groups()
    return width, height, fps, int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def frames(url: str, size: tuple[int, int], start: float | None, seconds: float | None, fps: float | None):
    width, height = size
    command = [FFMPEG, "-v", "error"]
    if start is not None:
        command += ["-ss", f"{start:.2f}"]
    command += ["-i", url]
    if seconds is not None:
        command += ["-t", f"{seconds:.2f}"]
    filters = [f"scale={width}:{height}"]
    if fps:
        filters.insert(0, f"fps={fps}")
    command += ["-vf", ",".join(filters), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=width * height * 3 * 4)
    frame_bytes = width * height * 3
    try:
        while True:
            buffer = process.stdout.read(frame_bytes)
            if len(buffer) < frame_bytes:
                break
            yield np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
    finally:
        process.stdout.close()
        process.wait()


def output_size(width: int, height: int, longest: int = 640) -> tuple[int, int]:
    scale = longest / max(width, height)
    return int(round(width * scale / 2) * 2), int(round(height * scale / 2) * 2)


def landmarker():
    from mediapipe.tasks import python as tasks
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.55,
        min_face_presence_confidence=0.55,
        min_tracking_confidence=0.55,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def frame_quality(rgb: np.ndarray):
    """Brightness and contrast on the same 32x18 thumbnail the app measures."""
    import cv2

    small = cv2.resize(rgb, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    luma = small[..., 0] * 0.2126 + small[..., 1] * 0.7152 + small[..., 2] * 0.0722
    return float(luma.mean()), float(luma.std())


def measure(detector, rgb: np.ndarray, timestamp_ms: int, crops: bool):
    import cv2
    import mediapipe as mp

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = detector.detect_for_video(image, timestamp_ms)
    brightness, contrast = frame_quality(rgb)
    row = {"brightness": brightness, "contrast": contrast, "face_found": False}
    eyes = None
    if result.face_landmarks:
        points = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float64)
        matrix = None
        if result.facial_transformation_matrixes:
            # MediaPipe JS exposes the matrix column-major; match it.
            matrix = np.asarray(result.facial_transformation_matrixes[0]).flatten(order="F").tolist()
        scores = None
        if result.face_blendshapes:
            scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        height, width = rgb.shape[:2]
        signals = face_signals(points, matrix, scores, aspect=width / height)
        if signals:
            row.update(signals)
            row["face_found"] = True
            for name in BLENDSHAPE_NAMES[1:]:
                row[f"bs_{name}"] = (scores or {}).get(name, np.nan)
            if crops:
                gray = to_gray(rgb.astype(np.float32))
                eyes = []
                for side in ("left", "right"):
                    transform = eye_crop_transform(points, side, width, height)
                    if transform is None:
                        eyes = None
                        break
                    eyes.append(cv2.warpAffine(
                        gray, transform, (EYE_CROP_SIZE, EYE_CROP_SIZE),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                    ))
                if eyes is not None:
                    eyes = np.clip(np.stack(eyes), 0, 255).astype(np.uint8)
    return row, eyes


# --- DMD ---------------------------------------------------------------------

def dmd_labels(annotation: dict, total: int):
    openlabel = annotation["openlabel"]
    eyes = np.full(total, "undefined", dtype=object)
    blink = np.zeros(total, dtype=bool)
    yawn = np.zeros(total, dtype=bool)
    for action in openlabel.get("actions", {}).values():
        kind = action.get("type", "")
        for interval in action.get("frame_intervals", []):
            span = slice(interval["frame_start"], min(total, interval["frame_end"] + 1))
            if kind.startswith("eyes_state/"):
                eyes[span] = kind.split("/", 1)[1]
            elif kind.startswith("blinks/"):
                blink[span] = True
            elif kind.startswith("yawning/"):
                yawn[span] = True
    return eyes, blink, yawn


def process_dmd(item):
    name, face_path, annotation_path = item
    target = DATA / "signals" / "dmd" / f"{name}.parquet"
    if target.exists():
        return name, "cached"
    started = time.time()
    annotation = json.loads(urllib.request.urlopen(url_for(annotation_path), timeout=120).read())
    total = annotation["openlabel"]["streams"]["face_camera"]["stream_properties"]["total_frames"]
    eyes_state, blink, yawn = dmd_labels(annotation, total)

    url = url_for(face_path)
    width, height, fps, _ = probe(url)
    size = output_size(width, height)
    detector = landmarker()
    rows, crops, crop_index = [], [], []
    for index, rgb in enumerate(frames(url, size, None, None, None)):
        if index >= total:
            break
        row, eyes = measure(detector, rgb, int(index * 1000 / fps), crops=True)
        row.update(frame=index, time_ms=index * 1000 / fps, eyes_state=eyes_state[index],
                   blink_label=bool(blink[index]), yawn_label=bool(yawn[index]))
        rows.append(row)
        if eyes is not None:
            crops.append(eyes)
            crop_index.append(index)
    detector.close()

    subject = name.split("_")[1]
    frame = pd.DataFrame(rows)
    frame.insert(0, "subject_id", f"dmd_{subject}")
    frame.insert(0, "session", name)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    crop_dir = DATA / "crops" / "dmd"
    crop_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(crop_dir / f"{name}.npz", crops=np.stack(crops), frame=np.array(crop_index))
    return name, f"{len(rows)} frames in {time.time() - started:.0f}s"


def dmd_items():
    listing = json.loads(urllib.request.urlopen(
        "https://huggingface.co/api/datasets/nyvantran/drowsy_dataset/tree/main/DMD?recursive=true",
        timeout=60,
    ).read())
    paths = [entry["path"] for entry in listing if entry["type"] == "file"]
    items = []
    for path in paths:
        if path.endswith("_rgb_face.mp4"):
            stem = path.rsplit("/", 1)[1].replace("_rgb_face.mp4", "")
            annotation = path.replace("_rgb_face.mp4", "_rgb_ann_drowsiness.json")
            if annotation in paths:
                name = stem.split("_")[0] + "_" + stem.split("_")[1] + "_" + stem.split("_")[3][:10] + "_" + stem.split("T")[1][:8].replace(";", "")
                items.append((name, path, annotation))
    return items


# --- UTA-RLDD -----------------------------------------------------------------

RLDD_LABELS = {"0": "alert", "5": "low_vigilant", "10": "drowsy"}


def process_rldd(item):
    entry, segments, seconds, fps = item
    level = entry["level"].split("_")[0]
    name = f"rldd_{entry['subject']}_{entry['level']}"
    target = DATA / "signals" / "rldd" / f"{name}.parquet"
    if target.exists():
        return name, "cached"
    started = time.time()
    url = url_for(entry["path"])
    try:
        width, height, _, duration = probe(url)
    except Exception as error:  # a missing or unreadable video is reported, not fatal
        return name, f"probe failed: {error}"
    size = output_size(width, height)
    rows = []
    usable = max(0.0, duration - seconds)
    for segment in range(segments):
        start = usable * (segment + 0.5) / segments
        detector = landmarker()
        for index, rgb in enumerate(frames(url, size, start, seconds, fps)):
            time_ms = start * 1000 + index * 1000 / fps
            row, _ = measure(detector, rgb, int(index * 1000 / fps), crops=False)
            row.update(segment=segment, frame=index, time_ms=time_ms)
            rows.append(row)
        detector.close()
    frame = pd.DataFrame(rows)
    frame.insert(0, "label", RLDD_LABELS[level])
    frame.insert(0, "fold", entry["fold"])
    frame.insert(0, "subject_id", f"rldd_{entry['subject']}")
    frame.insert(0, "session", name)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return name, f"{len(rows)} frames in {time.time() - started:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["dmd", "rldd"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--segments", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=40)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N videos")
    args = parser.parse_args()

    if args.dataset == "dmd":
        items, worker = dmd_items(), process_dmd
    else:
        manifest = json.loads((DATA / "rldd_manifest.json").read_text())
        items = [(entry, args.segments, args.seconds, args.fps) for entry in manifest]
        worker = process_rldd
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} videos, {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        for done, (name, status) in enumerate(pool.imap_unordered(worker, items), 1):
            print(f"[{done}/{len(items)}] {name}: {status}", flush=True)


if __name__ == "__main__":
    main()
