---
title: AI Surveillance Demo
emoji: 🎥
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Real-time person tracking + UCF-Crime anomaly detection
---

# AI Surveillance demo

Live person tracking (YOLOv8n + ByteTrack) with loitering and crowd alerts, plus an explorer for a weakly-supervised
anomaly detector (MIL) benchmarked on UCF-Crime (test frame-level AUC 0.802 ± 0.005 over 5 seeds).

Runs on the free CPU tier, no GPU. See the "About & constraints" tab.

Source, method, results and the learning log:
https://github.com/VedPawar1410/ai-autonomous-surveillance-system

This folder holds only the Space wrapper. The `surveillance` package is installed from that GitHub repo
(see `requirements.txt`). License is AGPL-3.0 because the demo uses Ultralytics YOLOv8.
