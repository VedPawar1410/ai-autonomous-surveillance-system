from surveillance.detect import EventDetector


def feed(det, frames):
    for i, ids in enumerate(frames):
        det.update(i, ids)
    return det.finish()


def test_loitering_fires_once_after_threshold():
    det = EventDetector(fps=1, loiter_s=5, crowd_n=99)
    events = feed(det, [[7]] * 10)                     # person 7 visible for 10 s
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == "loitering" and ev["track_id"] == 7
    assert ev["start_s"] == 0 and ev["flagged_at_s"] == 5 and ev["end_s"] == 9


def test_no_loitering_for_short_visits():
    det = EventDetector(fps=1, loiter_s=5, crowd_n=99)
    assert feed(det, [[1], [1], [1], [2], [2]]) == []


def test_crowd_needs_count_and_duration():
    det = EventDetector(fps=1, loiter_s=99, crowd_n=3, crowd_s=2)
    big, small = [1, 2, 3, 4], [1]
    events = feed(det, [big, big, small, big, big, big, big, small])   # first burst too short (1 s)
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == "crowd" and ev["start_s"] == 3 and ev["end_s"] == 7 and ev["peak_count"] == 4


def test_crowd_still_open_at_end_of_video_gets_closed():
    det = EventDetector(fps=1, loiter_s=99, crowd_n=2, crowd_s=1)
    events = feed(det, [[1, 2]] * 4)
    assert events[0]["end_s"] == 3
