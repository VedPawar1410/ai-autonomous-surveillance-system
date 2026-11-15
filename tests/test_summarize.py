from surveillance import summarize as S

EV = [{"type": "loitering", "track_id": 3, "start_s": 10.0, "end_s": 20.0},
      {"type": "loitering", "track_id": 4, "start_s": 21.0, "end_s": 25.0},
      {"type": "crowd", "start_s": 70.0, "end_s": 75.0, "peak_count": 9}]
SUMMARY = {"frames": 1000, "video_fps": 10.0, "tracked_people_per_frame": 3.0, "events": EV}  # 100 s video


def test_review_windows_pad_merge_and_clip():
    assert S.review_windows(EV, 100) == [(8.0, 27.0), (68.0, 77.0)]     # first two overlap after padding
    assert S.review_windows([{"type": "x", "start_s": 0.5, "end_s": 99.5}], 100) == [(0.0, 100.0)]


def test_facts_compute_footage_saved():
    f = S.build_facts(SUMMARY)
    assert f["review_seconds"] == 28.0 and f["footage_saved_pct"] == 72.0
    assert f["review_windows"] == ["0:08-0:27", "1:08-1:17"]


def test_no_events_means_no_llm_call(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(S, "_call", lambda p, f: (_ for _ in ()).throw(AssertionError("should not call LLM")))
    text, source, _ = S.summarize({**SUMMARY, "events": []})
    assert source == "template" and "Nothing to review" in text


def test_falls_back_to_next_provider_then_template(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "y")

    def fake_call(p, facts):
        if p["name"] == "groq":
            raise TimeoutError("down")
        return "gemini note"
    monkeypatch.setattr(S, "_call", fake_call)
    text, source, errors = S.summarize(SUMMARY)
    assert source == "gemini" and text.startswith("gemini note") and "72% less footage" in text
    assert errors and errors[0].startswith("groq")

    monkeypatch.setattr(S, "_call", lambda p, f: (_ for _ in ()).throw(TimeoutError("down")))
    text, source, errors = S.summarize(SUMMARY)
    assert source == "template" and len(errors) == 2 and "Review:" in text
