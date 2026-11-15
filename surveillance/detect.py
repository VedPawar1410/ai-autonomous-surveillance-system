"""Real-time person detection + tracking -> surveillance events.

    python -m surveillance.detect data/samples/vtest.avi --save-video

YOLOv8 finds people in each frame; ByteTrack links boxes across frames into persistent track IDs;
EventDetector turns tracks into events an operator cares about (loitering, crowding).
Frames are processed as a stream, so memory stays flat however long the video is.
"""
import argparse
import json
import time
from pathlib import Path

import cv2

from surveillance import config


class EventDetector:
    """Pure rule logic over per-frame track IDs (no model inside, so it is unit-testable)."""

    def __init__(self, fps: float, loiter_s: float = config.LOITER_SECONDS,
                 crowd_n: int = config.CROWD_MIN_PEOPLE, crowd_s: float = config.CROWD_MIN_SECONDS):
        self.fps, self.loiter_s, self.crowd_n, self.crowd_s = fps, loiter_s, crowd_n, crowd_s
        self.first_seen, self.last_seen, self.loiter_flagged = {}, {}, set()
        self.crowd_start, self.crowd_open = None, None
        self.events, self.last_frame = [], 0

    def update(self, frame_idx: int, person_ids) -> list:
        """Feed the person track IDs visible in one frame; returns events that fired on this frame."""
        t, fired = frame_idx / self.fps, []
        self.last_frame = frame_idx
        for pid in person_ids:
            self.first_seen.setdefault(pid, t)
            self.last_seen[pid] = t
            if pid not in self.loiter_flagged and t - self.first_seen[pid] >= self.loiter_s:
                self.loiter_flagged.add(pid)
                ev = {"type": "loitering", "track_id": int(pid), "start_s": round(self.first_seen[pid], 1),
                      "flagged_at_s": round(t, 1)}
                self.events.append(ev)
                fired.append(ev)

        if len(person_ids) >= self.crowd_n:
            if self.crowd_start is None:
                self.crowd_start = t
            if self.crowd_open is None and t - self.crowd_start >= self.crowd_s:
                self.crowd_open = {"type": "crowd", "start_s": round(self.crowd_start, 1), "peak_count": 0}
                self.events.append(self.crowd_open)
                fired.append(self.crowd_open)
            if self.crowd_open is not None:
                self.crowd_open["peak_count"] = max(self.crowd_open["peak_count"], len(person_ids))
        else:
            self._close_crowd(t)
        return fired

    def _close_crowd(self, t):
        if self.crowd_open is not None:
            self.crowd_open["end_s"] = round(t, 1)
        self.crowd_start, self.crowd_open = None, None

    def finish(self) -> list:
        self._close_crowd(self.last_frame / self.fps)
        for ev in self.events:
            if ev["type"] == "loitering":
                ev["end_s"] = round(self.last_seen[ev["track_id"]], 1)
        return self.events


def run(video: str, model_name: str = config.DETECTOR_MODEL, device: str = config.DETECTOR_DEVICE,
        imgsz: int = config.DETECTOR_IMGSZ, save_video: bool = False, max_seconds: float | None = None) -> dict:
    """Track people in a video and return a summary with events. max_seconds caps how much video is processed
    (the hosted demo uses it so one long upload can't monopolise a shared CPU)."""
    from ultralytics import YOLO  # imported here so the rule logic above has no heavy dependency

    config.MODELS_DIR.mkdir(exist_ok=True)
    model = YOLO(str(config.MODELS_DIR / model_name))  # downloads the official weights on first use
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    stem = Path(video).stem
    writer = None
    if save_video:
        writer = cv2.VideoWriter(str(config.OUTPUTS_DIR / f"{stem}_tracked.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    events, ids_seen, n, n_boxes = EventDetector(fps), set(), 0, 0
    start = time.perf_counter()
    stream = model.track(source=str(video), stream=True, persist=True, tracker="bytetrack.yaml",
                         classes=[config.PERSON_CLASS], device=device, imgsz=imgsz, verbose=False)
    max_frames = int(max_seconds * fps) if max_seconds else None
    for n, r in enumerate(stream):
        if max_frames is not None and n >= max_frames:
            n -= 1          # frame n was decoded but not processed
            break
        ids = [] if r.boxes.id is None else r.boxes.id.int().tolist()
        ids_seen.update(ids)
        n_boxes += len(ids)
        events.update(n, ids)
        if writer:
            writer.write(r.plot())
    elapsed = time.perf_counter() - start
    if writer:
        writer.release()

    summary = {"video": str(video), "model": model_name, "device": device, "imgsz": imgsz, "frames": n + 1,
               "video_fps": fps, "processing_fps": round((n + 1) / elapsed, 2),
               "tracked_people_per_frame": round(n_boxes / (n + 1), 2),
               "unique_track_ids": len(ids_seen), "events": events.finish()}
    (config.OUTPUTS_DIR / f"{stem}_events.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video", nargs="?", default=str(config.SAMPLE_VIDEO))
    p.add_argument("--model", default=config.DETECTOR_MODEL)
    p.add_argument("--device", default=config.DETECTOR_DEVICE)
    p.add_argument("--imgsz", type=int, default=config.DETECTOR_IMGSZ)
    p.add_argument("--save-video", action="store_true")
    a = p.parse_args()
    s = run(a.video, a.model, a.device, a.imgsz, a.save_video)
    print(json.dumps({k: v for k, v in s.items() if k != "events"}, indent=1))
    for ev in s["events"]:
        print(ev)
