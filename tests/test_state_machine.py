"""
test_state_machine.py

Tests for src/scoring/state_machine.py — specifically the hysteresis
behavior, since that's the part most likely to have subtle bugs.
"""

from src.scoring.state_machine import FatigueState, FatigueStateMachine


def test_starts_in_normal_state():
    sm = FatigueStateMachine()
    assert sm.state == FatigueState.NORMAL


def test_stays_normal_for_low_score():
    sm = FatigueStateMachine(warning_threshold=40.0, critical_threshold=70.0, sustain_seconds=3.0)
    state = sm.update(score=10.0, timestamp=0.0)
    assert state == FatigueState.NORMAL


def test_does_not_escalate_immediately_on_single_high_reading():
    sm = FatigueStateMachine(warning_threshold=40.0, critical_threshold=70.0, sustain_seconds=3.0)
    # One high reading shouldn't be enough on its own — hysteresis requires
    # it to sustain for `sustain_seconds`.
    state = sm.update(score=80.0, timestamp=0.0)
    assert state == FatigueState.NORMAL


def test_escalates_after_sustained_high_score():
    sm = FatigueStateMachine(warning_threshold=40.0, critical_threshold=70.0, sustain_seconds=3.0)
    sm.update(score=80.0, timestamp=0.0)
    sm.update(score=80.0, timestamp=1.0)
    state = sm.update(score=80.0, timestamp=3.5)  # 3.5s since first high reading
    assert state == FatigueState.CRITICAL


def test_a_single_dip_back_to_normal_resets_the_pending_timer():
    sm = FatigueStateMachine(warning_threshold=40.0, critical_threshold=70.0, sustain_seconds=3.0)
    sm.update(score=80.0, timestamp=0.0)
    sm.update(score=80.0, timestamp=1.0)
    # A brief dip back to Normal-range before the sustain window completes...
    sm.update(score=10.0, timestamp=2.0)
    # ...means the escalation clock must restart from here.
    state = sm.update(score=80.0, timestamp=2.5)  # only 0.5s since the restart
    assert state == FatigueState.NORMAL


def test_de_escalation_also_requires_sustained_evidence():
    sm = FatigueStateMachine(warning_threshold=40.0, critical_threshold=70.0, sustain_seconds=3.0)
    # Escalate to Critical first.
    sm.update(score=80.0, timestamp=0.0)
    sm.update(score=80.0, timestamp=3.5)
    assert sm.state == FatigueState.CRITICAL

    # A single good reading shouldn't immediately drop back to Normal.
    state = sm.update(score=5.0, timestamp=4.0)
    assert state == FatigueState.CRITICAL

    # But sustained good readings should.
    state = sm.update(score=5.0, timestamp=7.5)
    assert state == FatigueState.NORMAL


def test_warning_threshold_reached_before_critical():
    sm = FatigueStateMachine(warning_threshold=40.0, critical_threshold=70.0, sustain_seconds=3.0)
    sm.update(score=50.0, timestamp=0.0)
    state = sm.update(score=50.0, timestamp=3.5)
    assert state == FatigueState.WARNING


def test_microsleep_bypasses_hysteresis_immediately():
    sm = FatigueStateMachine(warning_threshold=40.0, critical_threshold=70.0, sustain_seconds=3.0)
    # A low score (wouldn't normally trigger anything) but a sustained
    # closure duration past the microsleep threshold — should jump
    # straight to Critical on the very first call, no waiting.
    state = sm.update(score=5.0, continuous_closed_seconds=1.6, timestamp=0.0)
    assert state == FatigueState.CRITICAL


def test_short_closure_does_not_trigger_microsleep_path():
    sm = FatigueStateMachine(warning_threshold=40.0, critical_threshold=70.0, sustain_seconds=3.0)
    # Below MICROSLEEP_SECONDS (1.5 by default) — should fall through to
    # normal score-based classification instead.
    state = sm.update(score=5.0, continuous_closed_seconds=0.5, timestamp=0.0)
    assert state == FatigueState.NORMAL
