"""Detector throughput on this machine: model size x input size x device, on the sample video.

    python -m surveillance.benchmark

Each config runs in its own subprocess with a timeout: on Apple-silicon MPS one config hung the whole
process during development, so isolation keeps one bad config from blocking the rest.
Measures detection only (no tracking/annotation), after 5 warm-up frames, over the first 200 frames.
"""
import json
import subprocess
import sys
import time

from surveillance import config

CONFIGS = [("yolov8n.pt", 640, "cpu"), ("yolov8n.pt", 320, "cpu"), ("yolov8n.pt", 640, "mps"),
           ("yolov8s.pt", 640, "cpu"), ("yolov8x.pt", 640, "cpu")]
N_FRAMES, TIMEOUT_S = 200, 600


def measure(model_name: str, imgsz: int, device: str):
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(config.SAMPLE_VIDEO))
    frames = [cap.read()[1] for _ in range(N_FRAMES)]
    model = YOLO(str(config.MODELS_DIR / model_name))
    for f in frames[:5]:
        model.predict(f, imgsz=imgsz, device=device, classes=[config.PERSON_CLASS], verbose=False)
    t = time.perf_counter()
    for f in frames:
        model.predict(f, imgsz=imgsz, device=device, classes=[config.PERSON_CLASS], verbose=False)
    return len(frames) / (time.perf_counter() - t)


def main():
    results = []
    for model_name, imgsz, device in CONFIGS:
        row = {"model": model_name, "imgsz": imgsz, "device": device, "frames": N_FRAMES}
        try:
            out = subprocess.run([sys.executable, "-m", "surveillance.benchmark", model_name, str(imgsz), device],
                                 capture_output=True, text=True, timeout=TIMEOUT_S)
            row["fps"] = round(float(out.stdout.strip().splitlines()[-1]), 2)
        except subprocess.TimeoutExpired:
            row["error"] = f"timeout after {TIMEOUT_S}s (hang)"
        except (IndexError, ValueError):
            row["error"] = "crashed: " + out.stderr.strip().splitlines()[-1][:200]
        results.append(row)
        print(row, flush=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "detector_fps.json").write_text(json.dumps(
        {"machine": "Apple M1, 16 GB (laptop)", "video": "vtest.avi 768x576 @ 10 fps", "results": results}, indent=1))


if __name__ == "__main__":
    if len(sys.argv) == 4:      # child process: measure one config and print its FPS
        print(measure(sys.argv[1], int(sys.argv[2]), sys.argv[3]))
    else:
        main()
