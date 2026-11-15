"""Hugging Face Space: live person tracking + UCF-Crime anomaly explorer.

Locally (from a repo checkout, with `pip install -e .`):   python space/app.py
On Hugging Face: the Space installs the `surveillance` package from GitHub and fetches small result files.
"""
import json
import subprocess
import urllib.request
from pathlib import Path

# ZeroGPU requires at least one @spaces.GPU function. `spaces` must be imported before CUDA is touched.
# On CPU hardware (and locally, where `spaces` isn't installed) the decorator is a no-op.
try:
    import spaces
    # Each call reserves this many GPU-seconds from the visitor's daily quota up front; the real work takes
    # a few seconds (300 frames at ~54 FPS + model load), so keep the reservation small.
    gpu_task = spaces.GPU(duration=30)
except ImportError:
    def gpu_task(fn):
        return fn

import gradio as gr
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surveillance import config, detect, summarize

LLM_DAILY_CAP = 200  # public demo: cap paid/free-tier LLM calls per day; beyond it, the template summary is used
_llm_usage = {"day": None, "calls": 0}

GITHUB_RAW = "https://raw.githubusercontent.com/VedPawar1410/ai-autonomous-surveillance-system/main"
SAMPLE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
MAX_SECONDS = 30  # cap per request: the free Space has 2 shared vCPUs and no GPU

# In a repo checkout, use the repo's own files. Installed from GitHub (the Space), keep everything in ./runtime.
IN_REPO = (config.ROOT / ".git").exists()
if not IN_REPO:
    runtime = Path(__file__).resolve().parent / "runtime"
    config.MODELS_DIR, config.OUTPUTS_DIR, config.REPORTS_DIR = runtime / "models", runtime / "outputs", runtime / "reports"
    config.SAMPLE_VIDEO = runtime / "samples" / "vtest.avi"


def ensure(path: Path, url: str) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


ensure(config.SAMPLE_VIDEO, SAMPLE_URL)
SCORES = json.loads(ensure(config.REPORTS_DIR / "test_scores.json", f"{GITHUB_RAW}/reports/test_scores.json").read_text())
METRICS = json.loads(ensure(config.REPORTS_DIR / "metrics.json", f"{GITHUB_RAW}/reports/metrics.json").read_text())
VIDEOS = {v["name"]: v for v in SCORES["videos"]}

# Reference palette (same as the report charts)
SURFACE, INK, INK_2, GRID, BLUE, GT_FILL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#d9d8d3"


# ---------- Tab 1: live tracking ----------
def to_browser_mp4(src: Path) -> Path:
    """OpenCV writes MPEG-4 Part 2, which browsers won't play; re-encode to H.264."""
    dst = src.with_name(src.stem + "_h264.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)
    return dst


@gpu_task
def _track_on_best_device(video_path):
    """Only the detection + tracking work runs here (on ZeroGPU this is the GPU-attached part)."""
    device = "cuda" if torch.cuda.is_available() else config.DETECTOR_DEVICE
    return detect.run(video_path, device=device, save_video=True, max_seconds=MAX_SECONDS)


def track(video_path):
    if not video_path:
        raise gr.Error("Upload a video or click the sample below first.")
    gpu_note = ""
    try:
        s = _track_on_best_device(video_path)
    except Exception as e:  # ZeroGPU quota exhausted / GPU busy -> the demo must still work: run on CPU
        print(f"GPU path failed, falling back to CPU: {type(e).__name__}: {e}", flush=True)
        gpu_note = f" (GPU unavailable: {str(e)[:80] or type(e).__name__}; ran on CPU instead)"
        s = detect.run(video_path, device="cpu", save_video=True, max_seconds=MAX_SECONDS)
    out = to_browser_mp4(config.OUTPUTS_DIR / f"{Path(video_path).stem}_tracked.mp4")  # CPU work, outside the GPU slot
    rows = [[e["type"], e.get("track_id", "—"), e["start_s"], e.get("end_s", "—"), e.get("peak_count", "—")]
            for e in s["events"]]
    stats = (f"**{s['frames']} frames** processed ({s['frames'] / s['video_fps']:.0f} s of video, capped at "
             f"{MAX_SECONDS} s) at **{s['processing_fps']} FPS** on this host's **{s['device'].upper()}** · "
             f"{s['tracked_people_per_frame']} people tracked per frame · {s['unique_track_ids']} track IDs · "
             f"YOLOv8n @ {s['imgsz']} px + ByteTrack{gpu_note}")
    return str(out), rows, stats, s, ""


def summarize_alerts(s):
    """LLM triage note from the last tracking result (Groq -> Gemini -> template), with a daily cap."""
    if not s:
        raise gr.Error("Run tracking first.")
    import datetime
    today = datetime.date.today()
    if _llm_usage["day"] != today:
        _llm_usage.update(day=today, calls=0)
    providers = summarize.PROVIDERS if _llm_usage["calls"] < LLM_DAILY_CAP else []
    text, source, errors = summarize.summarize(s, providers=providers)
    if source != "template":
        _llm_usage["calls"] += 1
    note = {"template": "template (no LLM available or daily cap reached)"}.get(source, f"LLM: {source}")
    return f"{text}\n\n<sub>Written by {note}. The LLM sees only the event table (not the video); "\
           f"review windows and % saved are computed in code.</sub>"


# ---------- Tab 2: anomaly explorer ----------
def explore(name):
    v = VIDEOS[name]
    n = SCORES["frames_per_clip"]
    x = [i * n for i in range(len(v["clip_scores"]))]
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(x, 0, v["clip_gt"], step="post", color=GT_FILL, linewidth=0, label="ground-truth anomaly")
    ax.step(x, v["clip_scores"], where="post", color=BLUE, linewidth=1.6, label="model anomaly score")
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlabel("frame", color=INK_2)
    ax.set_ylabel("score", color=INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, labelcolor=INK, loc="upper right", fontsize=8)
    fig.tight_layout()
    plt.close(fig)

    if v["label"]:
        stats = (f"**Anomalous video** · per-video frame AUC **{v['frame_auc']:.3f}** (how well the score ranks this "
                 f"video's anomalous frames above its normal ones) · peak score {v['max_score']:.2f}")
    else:
        alarms = sum(s > 0.5 for s in v["clip_scores"]) / len(v["clip_scores"])
        stats = f"**Normal video** · peak score {v['max_score']:.2f} · clips above 0.5 (false alarms): {alarms:.1%}"
    return fig, stats


k1 = METRICS["configs"]["k1"]
final = METRICS["final"]["test"]
HEADER = f"""# AI Surveillance demo
Real-time person tracking with event alerts, plus a weakly-supervised anomaly detector benchmarked on UCF-Crime.
Code, method and full results: [GitHub](https://github.com/VedPawar1410/ai-autonomous-surveillance-system)
"""
CONSTRAINTS = f"""### Built under constraints
- **Hosting:** free Hugging Face tier. On ZeroGPU a GPU is attached only for the tracking call and is limited by a
  daily quota; on CPU hardware it runs on shared vCPUs. Uploads are capped at {MAX_SECONDS} s. The stats line after
  each run shows the device actually used. On an M1 laptop CPU the same pipeline measured 23.8 FPS.
- **Model size chosen for CPU:** YOLOv8n at 320 px, the only configuration that kept up with a 10 FPS camera on a laptop
  CPU. The larger YOLOv8x would run at about 0.9 FPS on that hardware.
- **Data:** public, RGB-only I3D features (no optical-flow stream). Raw UCF-Crime video is about 100 GB, beyond a
  laptop's disk and compute budget, so the anomaly model is trained on precomputed features. That's also why this tab
  explores precomputed scores rather than scoring new uploads.
- **LLM alerts on free tiers:** Groq (gpt-oss-20b) first, Gemini (3.5 Flash-Lite) as the fallback, and a plain template
  if both fail or the daily cap of {LLM_DAILY_CAP} calls is reached. The LLM never sees video, only the event table.
- **Result under those limits:** test frame-level ROC AUC **{k1['frame_auc']['mean']:.3f} ± {k1['frame_auc']['std']:.3f}**
  over 5 seeds (final model {final['frame_auc']:.3f}), with 2.1% false alarms on normal videos.
"""

with gr.Blocks(title="AI Surveillance demo") as demo:
    gr.Markdown(HEADER)
    with gr.Tab("Live tracking"):
        gr.Markdown("Upload a short fixed-camera video. People are detected (YOLOv8n), linked across frames "
                    "(ByteTrack), and turned into events: **loitering** (one person on screen > 20 s) and "
                    "**crowd** (≥ 8 people for ≥ 2 s).")
        with gr.Row():
            video_in = gr.Video(label="Input video", sources=["upload"])
            video_out = gr.Video(label="Tracked output")
        run_btn = gr.Button("Run tracking", variant="primary")
        stats_md = gr.Markdown()
        events_df = gr.Dataframe(headers=["event", "track id", "start (s)", "end (s)", "peak people"],
                                 label="Events", interactive=False)
        last_summary = gr.State(None)
        summary_btn = gr.Button("Summarize alerts for the operator (LLM)")
        summary_md = gr.Markdown()
        gr.Examples(examples=[[str(config.SAMPLE_VIDEO)]], inputs=video_in,
                    label="Sample: OpenCV vtest.avi (pedestrians, 80 s, Apache-2.0)")
        run_btn.click(track, inputs=video_in, outputs=[video_out, events_df, stats_md, last_summary, summary_md])
        summary_btn.click(summarize_alerts, inputs=last_summary, outputs=summary_md)

    with gr.Tab("Anomaly explorer (UCF-Crime)"):
        gr.Markdown(f"Scores from the final MIL model (`models/final.pt`) on all 290 UCF-Crime test videos. "
                    f"Overall test frame-level AUC: **{final['frame_auc']:.3f}**. Gray = frames labelled anomalous.")
        names = sorted(VIDEOS, key=lambda n: (VIDEOS[n]["label"] == 0, n))
        pick = gr.Dropdown(names, value="Vandalism007" if "Vandalism007" in VIDEOS else names[0],
                           label="Test video (anomalous first, then normal)")
        plot = gr.Plot()
        explore_md = gr.Markdown()
        pick.change(explore, inputs=pick, outputs=[plot, explore_md])
        demo.load(explore, inputs=pick, outputs=[plot, explore_md])

    with gr.Tab("About & constraints"):
        gr.Markdown(CONSTRAINTS)

if __name__ == "__main__":
    demo.launch(show_error=True)  # show the real error text instead of a bare "Error"
