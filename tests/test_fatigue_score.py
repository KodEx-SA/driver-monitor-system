"""
test_fatigue_score.py

Tests for src/scoring/fatigue_score.py.

Uses a fixed baseline EAR and feeds in sequences of (avg_ear, mar) pairs
with explicit timestamps, so behavior is deterministic and doesn't
depend on real time passing.
"""

import pytest

from src.scoring.fatigue_score import FatigueScorer

BASELINE_EAR = 0.30


def test_score_zero_when_eyes_open_and_no_yawns():
    scorer = FatigueScorer()
    for t in range(10):
        result = scorer.update(avg_ear=BASELINE_EAR, baseline_ear=BASELINE_EAR, mar=0.1, timestamp=float(t))
    assert result.score == pytest.approx(0.0)
    assert result.perclos == pytest.approx(0.0)
    assert result.yawn_count_in_window == 0


def test_score_high_when_eyes_sustained_closed():
    scorer = FatigueScorer()
    # EAR far below baseline (eyes closed) for a sustained run of frames.
    closed_ear = BASELINE_EAR * 0.1
    for t in range(20):
        result = scorer.update(avg_ear=closed_ear, baseline_ear=BASELINE_EAR, mar=0.1, timestamp=float(t))
    assert result.perclos == pytest.approx(1.0)
    assert result.score == pytest.approx(100.0)


def test_yawn_counted_once_per_event_not_per_frame():
    scorer = FatigueScorer()
    # Mouth opens past threshold and STAYS open for several frames —
    # should count as exactly one yawn event, not five.
    for t in range(5):
        result = scorer.update(avg_ear=BASELINE_EAR, baseline_ear=BASELINE_EAR, mar=0.8, timestamp=float(t))
    assert result.yawn_count_in_window == 1


def test_yawn_score_contribution_is_capped():
    scorer = FatigueScorer()
    # Simulate many separate yawn events (mouth opens and closes repeatedly)
    # within the yawn window — the score contribution from yawns alone
    # should never exceed YAWN_SCORE_CAP, even with eyes wide open (no PERCLOS).
    t = 0.0
    for _ in range(10):
        scorer.update(avg_ear=BASELINE_EAR, baseline_ear=BASELINE_EAR, mar=0.8, timestamp=t)
        t += 0.1
        scorer.update(avg_ear=BASELINE_EAR, baseline_ear=BASELINE_EAR, mar=0.1, timestamp=t)
        t += 0.1

    result = scorer.update(avg_ear=BASELINE_EAR, baseline_ear=BASELINE_EAR, mar=0.1, timestamp=t)
    assert result.perclos == pytest.approx(0.0)
    assert result.score <= 30.0  # YAWN_SCORE_CAP
