# AI Surveillance: Weakly-Supervised Video Anomaly Detection + Real-Time Tracking

Two parts. The first detects **anomalous events** (robbery, assault, road accidents…) in CCTV video when only *video-level* labels exist for training (UCF-Crime benchmark). The second **tracks people in real time** and raises rule-based alerts (loitering, crowding).

**Live demo:** https://huggingface.co/spaces/vedpawar/ai-surveillance-demo (free Hugging Face ZeroGPU tier; the first load after idle takes about a minute)

| Headline (all measured in this repo, see `reports/`) | Value |
|---|---|
| UCF-Crime test **frame-level ROC AUC** (290 videos, 1.11M frames) | **0.802 ± 0.005** (5 seeds; final model 0.803) |
| False alarms on normal videos (score > 0.5) | 2.1% of frames |
| Person detection + tracking speed, laptop CPU (Apple M1) | **23.8 FPS** (real time for a 10 FPS camera) |
| Same pipeline on the live demo (free ZeroGPU, CUDA) | **50–96 FPS** (300 frames; cold vs. warm GPU, measured on the live Space) |
| Tests | 21 passing |

---

## How it works

```
                         ┌────────────── Part 1: anomaly detection (offline, benchmarked) ──────────────┐
UCF-Crime video  ──►  I3D features (precomputed, 2048-d)  ──►  MLP scorer  ──►  anomaly score per 16 frames
                      train: 32 segments per video              2048→512→32→1        │
                      test:  one per 16-frame clip              trained with          ▼
                                                                MIL ranking loss   frame-level ROC AUC

                         ┌────────────── Part 2: real-time tracking (runs on any video) ────────────────┐
camera / video  ──►  YOLOv8n (people, 320 px)  ──►  ByteTrack (persistent IDs)  ──►  rules  ──►  events JSON
                     23.8 FPS on M1 CPU             same person = same ID           loitering > 20 s
                                                                                     crowd ≥ 8 for ≥ 2 s
```

**Why these choices**
- **Multiple-instance learning (MIL), because labels are weak.** UCF-Crime says *whether* a training video contains a crime, never *when*. MIL treats each video as a "bag" of 32 segments and trains with a ranking loss: the highest-scoring segment of an anomalous video must outscore the highest-scoring segment of a normal video by a margin (Sultani, Chen & Shah, CVPR 2018). Two regularisers encode the priors that anomalies are *short* (sparsity) and *contiguous* (smoothness). Written from the paper in [`surveillance/model.py`](surveillance/model.py).
- **Precomputed I3D features + a small MLP.** Extracting I3D features from about 100 GB of raw video needs a GPU cluster. The community-standard features let a 1M-parameter model train in about 6 minutes on a laptop CPU.
- **Model selection on validation, evaluation on test exactly once.** 10% of *training* videos are held out (video-level labels only). The best epoch and the final model are picked there. Test frame labels are used only to report.
- **Tracking, not just detection.** Detection answers "is there a person in this frame?". Loitering needs "has *this* person been here for 20 s?", which needs identity over time, so ByteTrack goes on top of YOLO.
- **YOLOv8n at 320 px on CPU.** It was the only configuration measured to keep up with the camera on a laptop (table below). On this machine, the M1 GPU was *slower* than the CPU for one-frame-at-a-time inference.

## Data

- **UCF-Crime I3D features** (RGB, ten-crop, 2048-d) from the Hugging Face mirror [`jinmang2/ucf_crime_tencrop_i3d_seg32`](https://huggingface.co/datasets/jinmang2/ucf_crime_tencrop_i3d_seg32/tree/tushar-n). That's 1610 training videos (800 normal / 810 anomalous, 13 crime types) and 290 test videos (150 / 140) with per-frame ground truth. **Only 7.6% of test frames are anomalous**, which is why the metric is ROC AUC, not accuracy.
- **Sample video** for tracking: OpenCV's `vtest.avi` (Apache-2.0, 768×576, 10 fps, 80 s of pedestrians).
- Sources, layouts, verified counts and download commands are in [`data/README.md`](data/README.md). Nothing in `data/` is committed.

## Results

### Anomaly detection (UCF-Crime test set)

| Variant (5 seeds each) | Frame AUC | AUC inside anomalous videos only | False-alarm rate @ 0.5 |
|---|---|---|---|
| **Top-1 MIL** (the paper's loss) | **0.802 ± 0.005** | 0.597 | 2.2% |
| Top-3 MIL (average of the 3 highest segments) | 0.804 ± 0.002 | 0.597 | 3.3% |

The final model is top-1, seed 3, **picked on validation** (val video-AUC 0.925). It scores 0.803 test frame-AUC and is saved as `models/final.pt` (4 MB).

For context, as reported in the literature: Sultani et al. with C3D features got 0.754. The RTFM paper (Tian et al., ICCV 2021) lists Sultani-style MIL on I3D-RGB at about 0.78 and RTFM itself at 0.843. This re-implementation lands where an honest MIL baseline on I3D-RGB features should.

| | |
|---|---|
| ![Ablation](reports/ablation.png) | ![ROC](reports/roc.png) |
| **Top-k ablation:** the 0.002 gap is inside the seed-to-seed spread, so it's a tie on accuracy. Top-3 is more stable (std 0.002 vs 0.005)… | **ROC** of the final model over all 1.11M test frames. |
| ![Training](reports/training_curves.png) | ![Timelines](reports/timelines.png) |
| …and converges in about 20 epochs instead of about 60 (validation curves, 5 seeds). | **Score timelines**, picked by a fixed rule (25th/75th-percentile anomalous video, median normal video), not cherry-picked. The gray band is the ground truth. |

### Real-time detection + tracking (Apple M1 laptop, CPU unless noted)

| Configuration | FPS | Notes |
|---|---|---|
| **YOLOv8n @ 320 + ByteTrack + rules (default)** | **23.8** | 5.61 people tracked per frame, 47 track IDs |
| YOLOv8n @ 640 + ByteTrack + rules | 6.5 | 5.83 people tracked per frame, 40 track IDs (steadier tracks) |
| YOLOv8n @ 640, detection only | 7.3 | |
| YOLOv8s @ 640, detection only | 2.9 | |
| YOLOv8x @ 640, detection only (used in the capstone) | 0.87 | |
| YOLOv8n @ 640 on the M1 GPU (MPS), detection only | 0.99 | slower than CPU; hung in 2 earlier runs |

The **320 vs 640 tradeoff**: 320 is 3.6× faster and loses only 4% of detected people. But tracks break more often (47 IDs vs 40), and broken tracks restart the loitering timer, so on the sample video it found 6 loitering events instead of 7 and missed the one borderline crowd event. Raw numbers are in `reports/detector_fps.json` and `reports/detector_tradeoff.json`.

### Operator alerts (LLM layer)

Tracking produces timestamped events. `surveillance/summarize.py` turns them into a short triage note, so an operator reviews only the seconds that matter instead of the whole recording:

1. **Code** pads each event by 2 s, merges overlapping windows, and computes *review time* and *% of footage saved*.
2. **An LLM** (Groq `gpt-oss-20b`, with Gemini `3.5-flash-lite` as the fallback, both on free tiers) gets *only those facts as JSON* and writes 2–4 sentences. It never sees the video, and the prompt forbids anything not in the facts.
3. **Code** appends the exact review windows after the LLM text, so the actionable part never depends on the LLM. If both providers fail, or a daily cap of 200 calls is hit, a template note is used instead.

Measured on the live sample: Groq and Gemini each answered in about 0.7 s, citing the right track and times. On that 30 s clip the saving is only 5%, because one person loiters for almost the whole clip. The approach pays off on long footage with sparse events (e.g., 28 s to review out of 100 s = 72% saved, in `tests/test_summarize.py`).

### From capstone to rebuild

| | 2025 capstone version | This rebuild |
|---|---|---|
| Scope | Research prototype: scripts plus a report and poster | Installable, tested package with documented commands and a live demo |
| Anomaly evaluation | AUC averaged over pairs of videos | Standard pooled frame-level AUC, 5 seeds, validation-based selection: **0.802 ± 0.005** |
| Anomaly model | MLP scorer on I3D features | My own MIL implementation from the paper, on public I3D-RGB features, with a top-k ablation |
| Detection | Pretrained YOLOv8 and Faster R-CNN | YOLOv8n + ByteTrack, FPS benchmarked per configuration on real hardware |
| Events / alerts | Designed in the report | Loitering + crowd events from tracks (unit-tested), plus LLM operator notes |
| Tests | — | 21 |

The same trained models scored with the capstone's per-pair metric give 0.786–0.790, which isn't comparable to published results. `tests/test_evaluate.py` shows why: per-pair averaging can't see a normal frame in one video outscoring an anomalous frame in another. A single alarm threshold across cameras has to handle exactly that case.

## Built under constraints

Every number above was produced on free or personal hardware. These limits shaped the design. Each row says what I did about one, and what lifting it would unlock.

| Constraint | What I did about it | What lifting it would unlock |
|---|---|---|
| **No GPU for development**: one Apple M1 laptop, 16 GB RAM | Small models (1M-parameter MLP, YOLOv8n), CPU-first code. Trained 10 models (~6 min each) on CPU. Benchmarked every detector config instead of assuming. | YOLOv8s/m at 640 px in real time; feature extraction from raw video |
| **The M1 GPU (MPS) was unusable**: it hung twice and ran 7× slower than CPU for frame-by-frame inference | Defaulted to CPU, and isolated each benchmark config in its own process so a hang can't block the rest | Batched GPU inference |
| **About 10–14 GB of free disk** (it filled up completely mid-project) | Read features straight from the zips (no 10 GB extraction), cached crop-averaged arrays (~1 GB), deleted the zips once cached | The raw UCF-Crime videos (~100 GB) |
| **The original data was lost**, and raw video was too big | Rebuilt on public precomputed I3D features (RGB only, no optical flow), with provenance documented | A two-stream (RGB + flow) model; end-to-end scoring of new videos |
| **Free hosting**: Hugging Face ZeroGPU, a GPU only per request within a daily quota, plus a forced torch ≤ 2.13 | Uploads capped at 30 s; only detection/tracking runs inside the GPU slot (video re-encoding stays outside); package dependencies declared as ranges so they install alongside the host's PyTorch | Always-on GPU, longer videos, concurrent users |
| **Free LLM tiers**: rate limits, models retired without notice (Llama 3.3 vanished from Groq; Gemini 2.5 Flash closed to new users; Gemini 3.8 Flash timed out at 20 s) | Provider-agnostic OpenAI-compatible client, Groq → Gemini → template fallback, model names overridable by env var, daily call cap | A dedicated, SLA-backed model endpoint |
| **No labelled event data** for loitering/crowding | Hand-set, documented thresholds; event logic unit-tested with synthetic tracks | Tuned and measured alert precision/recall |

**The result under these limits:** a benchmarked anomaly detector at 0.80 frame-AUC, comparable to published MIL baselines, plus a tracker that runs in real time on a laptop CPU and at 50–96 FPS on a free shared GPU. Every number is reproducible from the commands below.

## Project structure

```
surveillance/
  config.py       all paths, constants and thresholds (single source of truth)
  download.py     fetch the UCF-Crime feature zips (resumable)
  data.py         read features from the zips, crop-average, cache, assert counts/shapes/alignment
  model.py        MLP scorer + MIL ranking loss (top-k option)
  train.py        paired-bag training, epoch selection on validation video-AUC
  evaluate.py     pooled frame-level AUC, anomalous-only AUC, false-alarm rate, the 2025 metric
  experiment.py   seeds x variants grid: train, then evaluate on test once
  report.py       aggregate runs -> reports/metrics.json + charts, pick final model on validation
  detect.py       YOLOv8 + ByteTrack + EventDetector (loitering, crowd), streaming
  benchmark.py    detector FPS per config, one subprocess each (isolates hangs)
  export_scores.py  final model's per-clip test scores -> reports/test_scores.json (for the demo)
  summarize.py    events -> review windows + % footage saved (code) + LLM operator note (Groq -> Gemini -> template)
space/            Hugging Face Space wrapper: app.py (Gradio), requirements.txt, packages.txt, README.md
tests/            21 tests: data, loss, metrics, event rules, alert summaries
models/final.pt   the selected anomaly model (4 MB)
reports/          metrics.json, test_scores.json, detector_fps.json, detector_tradeoff.json, charts
data/README.md    data sources and download instructions (data itself not committed)
pyproject.toml    makes `surveillance` pip-installable (dependency ranges); requirements.txt = exact pins
```

## Run locally

Python 3.12. These are the exact commands run in this rebuild.

```bash
pip install -r requirements-dev.txt          # runtime: requirements.txt; dev adds matplotlib + pytest
python -m pytest                             # 21 tests (one is skipped if the dataset isn't downloaded)
```

Real-time tracking on the sample video (no dataset needed):

```bash
mkdir -p data/samples && curl -L -o data/samples/vtest.avi https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi
python -m surveillance.detect --save-video    # -> outputs/vtest_events.json, outputs/vtest_tracked.mp4
python -m surveillance.benchmark              # -> reports/detector_fps.json
```

Reproduce the anomaly results (about 10 GB download; the zips can be deleted after caching):

```bash
python -m surveillance.download               # feature zips + ground truth -> data/ucf_crime/
python -m surveillance.data                   # build caches, print verified counts
python -m surveillance.experiment --k 1       # 5 seeds, ~6 min each on an M1 CPU
python -m surveillance.experiment --k 3
python -m surveillance.report                 # -> reports/metrics.json, charts, models/final.pt
```

## Deployment

**Hugging Face Spaces (Gradio SDK, free ZeroGPU hardware):** https://huggingface.co/spaces/vedpawar/ai-surveillance-demo

```
Space repo (text files only: space/*)          GitHub repo (this one)
  app.py ── pip install ai-surveillance @ git+https://github.com/...@main
         ── fetches reports/test_scores.json + metrics.json (raw GitHub URLs)
         ── fetches the sample video from OpenCV
```

- **Why the Space holds only a wrapper:** Hugging Face rejects binary files committed with plain git, and this repo has PNGs and `final.pt` in its history. The Space installs the package from GitHub, so GitHub stays the single source of truth.
- **Tabs:** *Live tracking* (upload ≤ 30 s, or use the sample, to get the tracked video + events), *Anomaly explorer* (score vs. ground truth for all 290 test videos, precomputed from `final.pt`), *About & constraints*.
- **GPU use:** only detection + tracking run inside `@spaces.GPU`. The same code falls back to CPU anywhere else.
- **To update:** push to GitHub, set the new commit SHA in `space/requirements.txt`, and upload that file to the Space. The package is pinned to an exact commit, so the Space always runs known code, and the changed line forces Hugging Face to reinstall instead of reusing a cached build.
- Local run of the same app: `pip install -e . gradio matplotlib && python space/app.py`.

## Limitations

- **Localization is weak.** Within crime videos, the AUC for *when* the crime happens is only 0.60. Most of the 0.80 comes from separating crime videos from normal ones. That's typical of weakly-supervised MIL, and it's the main thing newer methods (RTFM, MGFN) improve on.
- **RGB only.** This feature release has no optical-flow stream, so motion is captured only through I3D's 3D convolutions.
- **The anomaly model can't score a new raw video yet.** That needs the *same* I3D-ResNet50 feature extractor the dataset used. A different extractor would create a training/serving mismatch that silently breaks the model.
- **Unofficial data mirror.** The features come from a single-contributor Hugging Face upload. Counts, shapes and ground-truth alignment are all verified on load, but the files aren't checksummed against an official release.
- **Event rules are hand-set** (20 s, 8 people, 2 s) and evaluated on a single sample video, with no labelled ground truth for events. Some low-confidence false positives get through (e.g., a sign detected as a person at 0.30).
- **Hardware-specific numbers.** All FPS figures are from one M1 laptop.
- **Licensing:** Ultralytics YOLOv8 is AGPL-3.0, which matters if this is ever deployed as a service.

## Learning log: mistakes and what I learned

Built in layers; each records what went wrong and the lesson.

### Layer 1 - VIT capstone (2024–25, report dated April 2025; Ultralytics YOLOv8, torchvision Faster R-CNN, PyTorch)

*What I built:* a capstone report and poster designing a three-model surveillance system (YOLOv8, Faster R-CNN and an I3D-feature anomaly model) with alerts, a dashboard, edge deployment and a cost/ROI analysis, plus prototype scripts. It gave me the problem framing and the system design. The rebuild raised the engineering and evaluation to a professional standard.

| # | What I took forward | Then → now | Lesson |
|---|---|---|---|
| 1.1 | **Describe the model precisely** | "I3D two-stream model" → "an MLP on precomputed I3D-RGB features" | Name exactly what you trained. A precise description is still a good project. |
| 1.2 | **Use the benchmark's evaluation protocol** | AUC averaged over video pairs → pooled frame-level AUC, fixed seeds, model selection on validation only | The protocol is part of the result. |
| 1.3 | **Every number needs a script** | Planned accuracy and FPS figures in the report → every metric traces to a file in `reports/` | If a number can't be regenerated, it isn't a result yet. |
| 1.4 | **Write portable code** | Hard-coded dataset paths and `.cuda()` → `config.py` and CPU-first code | Paths and devices go in config; the code shouldn't assume one machine. |
| 1.5 | **Stream video** | Collecting every frame's results in memory → `stream=True` | A camera never stops, so memory has to stay flat. |
| 1.6 | **Scope the write-up to what runs** | Alerts, dashboard and edge deployment designed in the report → events and LLM alerts shipped; the rest listed as future work | Ship a slice end to end, then extend it. |

### Layer 2 - Putting it on GitHub (Nov 8–10, 2025)

*What I built:* a first public repo with the prototype scripts, a README and the architecture diagram.

| # | Lesson | What it taught me |
|---|---|---|
| 2.1 | **Push from the working source and test a fresh clone** | Moving code between documents and repos can silently break formatting. A clean clone catches that. |
| 2.2 | **A repo is reproducible only if a stranger can run it from the README** | It needs pinned requirements, data instructions and paths that match the README. |
| 2.3 | **Record data provenance the day you download it** | The capstone's feature zip was shared privately and is no longer available, so the rebuild uses a documented public release. |

### Layer 3 - The rebuild (November 2026)

*What I did:* rebuilt the project from scratch as a tested package on public data, with my own implementation, standard evaluation and measured speeds.

| # | Decision | Why |
|---|---|---|
| 3.1 | Public Hugging Face feature mirror, documented in `data/README.md` with sizes, layouts and verified counts | Anyone can reproduce the results. |
| 3.2 | Read features straight from the zips, crop-average them, cache them, and **assert counts, shapes and frame alignment** | The shape check caught on its first run that train is `(crops, 32 segments, dim)` but test is `(clips, crops, dim)`. Averaging the wrong axis would have silently produced garbage. |
| 3.3 | MIL ranking loss written from the paper, with sources credited | My own implementation, precisely described. |
| 3.4 | Validation split from training videos (video labels only); test evaluated once per model | No selection on test. |
| 3.5 | Pooled frame-level AUC, 5 seeds, mean ± std, plus the capstone's per-pair metric reproduced for comparison | A number comparable to the literature, with honest uncertainty. |
| 3.6 | Top-k ablation, reported as a tie | 0.804 vs 0.802 is inside the noise; the real gain is stability and faster convergence. |
| 3.7 | YOLOv8n + ByteTrack + tested rule-based events, streaming | Turns detection into alerts an operator can use, with flat memory. |
| 3.8 | FPS benchmark per config in isolated processes; 320 px CPU as the default | Real numbers from real hardware. Showed that YOLOv8x runs at 0.87 FPS on a laptop CPU and that MPS was slower than CPU. |
| 3.9 | `config.py`, pinned requirements, tests, clean-install check | Runs on any machine from the README. |

*Known gaps I chose not to fix:* RGB-only features (no flow stream), no end-to-end anomaly scoring on raw video (needs the matching I3D extractor), weak temporal localization (0.60), event thresholds not validated against labelled events, an unofficial data mirror.

### Layer 4 - Deploying under constraints (November 2026, Hugging Face Spaces, Gradio)

*What I built:* a public demo on Hugging Face's free ZeroGPU tier, plus packaging so the Space installs the tested code from GitHub.

| # | Mistake | What actually happened | Lesson |
|---|---|---|---|
| 4.1 | **Used exact pins as package dependencies** | `pyproject.toml` read `requirements.txt`, so the package demanded `torch==2.15.0`. ZeroGPU forces `torch<=2.13.0`, and the build failed with `ResolutionImpossible`. Fixed with ranges (`torch>=2.2`), verified by resolving the Space's exact install for Linux locally, and re-running the tests on torch 2.13 (same 0.8032 AUC). | A *package* declares compatible ranges; an *app* pins exact versions in a lock file. Don't use one file for both. |
| 4.2 | **Assumed ZeroGPU would just run a CPU app** | The build passed, then the runtime stopped the app with `No @spaces.GPU function detected during startup`. Fixed by wrapping only detection + tracking in `@spaces.GPU`, with a CPU fallback. | Read the platform's runtime contract, not just its build docs. |
| 4.3 | **Let the disk fill up mid-project** | With 10 GB of zips plus caches on a nearly full laptop, commands failed with `ENOSPC`. The zips weren't needed once the caches existed. | Know which artifacts are raw inputs and which are derived, and delete what can be re-derived or re-downloaded. |
| 4.4 | **Nearly shipped a video browsers can't play** (caught locally) | OpenCV writes MPEG-4 Part 2 (`mp4v`), which browsers won't play. The app re-encodes to H.264 with ffmpeg, which `packages.txt` installs on the Space. | Test the demo in a real browser before deploying. |
| 4.5 | **Hard-coded LLM model names from memory** | The first test returned 404 from both providers: Groq no longer serves Llama 3.3, and Gemini 2.5 Flash is closed to new users. The fallback chain still produced a (template) answer. Fixed by checking each provider's `/models` list and picking current, fast models. | Hosted models get retired; verify against the live model list, make names configurable, and always have a non-LLM fallback. |
| 4.6 | **Installed the package from a moving branch (`@main`)** | After adding `summarize.py`, the Space crashed with `ImportError: cannot import name 'summarize'`. The requirements line hadn't changed, so Hugging Face reused a cached install of the old package. Fixed by pinning to a commit SHA. | Pin deployments to exact versions; a moving reference plus a build cache means you don't know what code is running. |
| 4.7 | **Reserved far more GPU time than needed** | `@spaces.GPU(duration=120)`: the first live run worked, the next one failed. Each call reserves its full duration from the visitor's daily ZeroGPU quota, so a second 120 s reservation most likely didn't fit. With `duration=30`, two back-to-back runs passed (50 and 96 FPS on CUDA). I also added an automatic CPU fallback and `show_error=True`, so a visitor always gets a result and a readable reason. | Size resource reservations to the real workload, and design shared-quota features to degrade gracefully instead of failing. |

### Key takeaways

1. **A number without a script is a claim, not a result.** Every metric here traces back to a file in `reports/`.
2. **The evaluation protocol is part of the result.** The same model scores 0.79 or 0.80 depending on how you average, and only one of those is comparable to papers.
3. **Assert your data's shape before you trust it.** One assertion caught a train/test layout mismatch that would have silently ruined training.
4. **"Real time" depends on the hardware.** YOLOv8x at 640 px runs at 0.87 FPS on a laptop CPU; YOLOv8n at 320 px runs at 23.8 FPS. Resolution and model size are the knobs, and each has a measurable cost.
5. **A reproducible number beats an impressive one.** Anyone can regenerate 0.80 from this repo, and it holds up under questioning.
6. **Constraints are design inputs.** No GPU, a nearly full disk and free hosting decided the model sizes, the data format and the deployment architecture. Naming them, and what lifting each would unlock, is part of the result.

## Future improvements

- Run the alert LLM locally (a small open model on the Space's GPU), so no event data leaves the deployment: the privacy-first option for real CCTV.
- Score raw video end to end using the matching I3D-ResNet50 extractor.
- Try a stronger weakly-supervised method (RTFM-style feature magnitudes) to improve localization.
- Label a few events on real footage to tune and evaluate the alert rules.

## Credits

- Sultani, Chen & Shah, *Real-world Anomaly Detection in Surveillance Videos*, CVPR 2018 (the method and the UCF-Crime dataset).
- Tian et al., *RTFM*, ICCV 2021 (I3D feature release and comparison numbers).
- The `jinmang2` Hugging Face feature upload; OpenCV's `vtest.avi`.
- Ultralytics YOLOv8 (AGPL-3.0) and ByteTrack (Zhang et al., ECCV 2022).
- The 2025 version began as a VIT capstone project (School of Computer Science and Engineering, April 2025), whose prototype built on an open-source PyTorch re-implementation of Sultani et al. This rebuild is a complete rewrite.
