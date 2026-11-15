"""Turn tracking events into a short operator alert: which seconds of footage to review, and why.

The numbers (review windows, % of footage saved) are computed here in plain code. The LLM only phrases a
2-4 sentence note from those facts, and the code appends the exact review windows afterwards, so the
actionable part never depends on the LLM.

Providers (OpenAI-compatible chat API, tried in order; set via environment variables):
  GROQ_API_KEY   -> Groq    (primary)
  GEMINI_API_KEY -> Gemini  (fallback)
If neither works, a deterministic template summary is returned.
"""
import json
import os
import urllib.request

PAD_S = 2.0  # seconds of context added before/after each event window

PROVIDERS = [
    {"name": "groq", "env": "GROQ_API_KEY", "url": "https://api.groq.com/openai/v1/chat/completions",
     "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")},
    {"name": "gemini", "env": "GEMINI_API_KEY",
     "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
     "model": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")},
]
# Model names verified against each provider's /models endpoint in Nov 2026 - providers retire models often
# (Llama 3.3 had vanished from Groq; gemini-2.5-flash was closed to new users; gemini-3.8-flash timed out
# at 20 s), hence a fast "lite" fallback, env overrides, and the template as the last resort.

SYSTEM_PROMPT = (
    "You write short alerts for a CCTV operator. You receive JSON facts produced by a person tracker. "
    "Use ONLY these facts: never invent people, actions, objects or causes that are not in the JSON "
    "(you cannot see the video). Write 2-4 plain sentences: what was detected, when (mm:ss), and that the "
    "operator only needs to review the listed windows. No markdown headings, no bullet lists."
)


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def review_windows(events, video_s: float, pad: float = PAD_S):
    """Padded, merged [start, end] intervals covering every event."""
    spans = sorted((max(0.0, e["start_s"] - pad), min(video_s, e.get("end_s", e["start_s"]) + pad)) for e in events)
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(round(s, 1), round(e, 1)) for s, e in merged]


def build_facts(summary: dict) -> dict:
    video_s = summary["frames"] / summary["video_fps"]
    windows = review_windows(summary["events"], video_s)
    review_s = sum(e - s for s, e in windows)
    return {
        "video_length": mmss(video_s),
        "video_seconds": round(video_s, 1),
        "people_tracked_per_frame": summary.get("tracked_people_per_frame"),
        "events": [{k: (mmss(v) if k in ("start_s", "end_s", "flagged_at_s") else v) for k, v in e.items()}
                   for e in summary["events"]],
        "review_windows": [f"{mmss(s)}-{mmss(e)}" for s, e in windows],
        "review_seconds": round(review_s, 1),
        "footage_saved_pct": round(100 * (1 - review_s / video_s), 1) if video_s else 0.0,
    }


def template_summary(facts: dict) -> str:
    n = len(facts["events"])
    if n == 0:
        return f"No loitering or crowd events in {facts['video_length']} of footage. Nothing to review."
    kinds = sorted({e["type"] for e in facts["events"]})
    return (f"{n} event(s) detected ({', '.join(kinds)}) in {facts['video_length']} of footage. "
            f"Review {facts['review_seconds']:.0f} s instead of {facts['video_seconds']:.0f} s.")


def _call(provider: dict, facts: dict, timeout: float = 20.0) -> str:
    body = json.dumps({
        "model": provider["model"],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": json.dumps(facts)}],
        "temperature": 0.2,
        "max_tokens": 400,
    }).encode()
    req = urllib.request.Request(provider["url"], data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {os.environ[provider['env']]}",
        "User-Agent": "ai-surveillance-demo"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    if not text or not text.strip():
        raise ValueError("empty completion")
    return text.strip()


def summarize(summary: dict, providers=PROVIDERS):
    """Returns (alert_text, source, errors). source = provider name or 'template'."""
    facts = build_facts(summary)
    footer = ""
    if facts["review_windows"]:
        footer = (f"\n\n**Review:** {', '.join(facts['review_windows'])} "
                  f"({facts['review_seconds']:.0f} s of {facts['video_seconds']:.0f} s, "
                  f"**{facts['footage_saved_pct']:.0f}% less footage to watch**)")
    errors = []
    if facts["events"]:  # no point paying for an LLM call to say "nothing happened"
        for p in providers:
            if not os.environ.get(p["env"]):
                continue
            try:
                return _call(p, facts) + footer, p["name"], errors
            except Exception as e:  # network, auth, quota, bad model name -> try the next provider
                errors.append(f"{p['name']}: {type(e).__name__}: {str(e)[:120]}")
    return template_summary(facts) + footer, "template", errors
