# 🚗 Driver Drowsiness & Fatigue Monitoring System

Real-time driver fatigue detection using computer vision - built to catch the moment attention starts slipping, before it becomes a crash statistic.

> Most accidents don't happen because people can't drive. They happen because people are tired, distracted, or losing focus. This project uses your device's camera to detect the early physical signs of fatigue — eye closure, yawning, and loss of focus — and raise an alert before it's too late.

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Features](#features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Dashboard](#dashboard)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why This Exists

Fatigue impairs reaction time the same way alcohol does, but it creeps up gradually and drivers routinely misjudge how tired they are. Cameras are cheap, on-device CV models are fast enough to run without a GPU, and the warning signs of drowsiness — slow blinks, yawning, a drifting gaze, a nodding head — are all visible on the face well before a driver consciously notices anything wrong.

This system watches for those signs in real time and turns them into a single, understandable risk signal.

---

## Features

| Category | What it does |
|---|---|
| 👁️ Eye Closure Detection | Tracks eye landmarks and flags prolonged or excessive eye closure |
| 🥱 Yawn Detection | Measures mouth aspect ratio to catch yawning |
| 👀 Gaze / Attention Tracking | Detects when the driver looks away from the road for too long |
| 📊 Fatigue Score | Combines all signals into one real-time score, not isolated alarms |
| 🔔 Tiered Alerts | Escalating alerts (gentle → urgent) instead of one jarring alarm |
| 📁 Event Logging | Every alert-worthy event is logged with a timestamp and context |
| 📉 Trends Dashboard | Visualizes fatigue patterns across a session or over time |
| 📤 Data Export | Export logged driving/fatigue data for further analysis |

---

## How It Works

The system is built around a simple idea: **no single signal should be trusted on its own.** A closed eye could be a blink. A tilted head could be a shoulder check. Fatigue detection only becomes reliable when several weak signals are combined and smoothed over time.

**1. Face & Landmark Tracking**
MediaPipe Face Mesh locates ~468 facial landmarks per frame, from which we extract the eye, mouth, and head-pose reference points.

**2. Eye Aspect Ratio (EAR)**
A geometric ratio of eye height to eye width. It drops sharply when eyes close. Rather than a fixed threshold, each session calibrates a short personal baseline first (a few seconds of normal blinking), since eye shape varies a lot between people.

**3. PERCLOS (Percentage of Eye Closure)**
Instead of counting "how many frames in a row were the eyes closed," we track the *percentage of time eyes were mostly closed over a rolling window* (e.g. the last 60–90 seconds). This is the same class of metric used in real driver-monitoring research — it's far more resistant to noise than a single-frame trigger.

**4. Mouth Aspect Ratio (MAR)**
Same idea as EAR, applied to the mouth, to detect yawning.

**5. Head Pose Estimation**
Landmark positions are used to estimate head pitch/yaw. This catches head-nodding (an early microsleep sign) and gaze-away-from-road behavior without needing a separate detector.

**6. Fatigue Score**
A weighted composite of PERCLOS, yawn frequency, head-nod frequency, and blink rate — smoothed and decaying over time — produces one 0–100 risk score, rather than four independent alarms fighting for attention.

**7. State Machine**
Risk moves through `Normal → Warning → Critical` states with hysteresis: it takes sustained evidence to escalate, and a cooldown to de-escalate. This avoids alert flapping from one noisy frame.

---

## Architecture

Built with clean separation of concerns so each piece can be tested, replaced, or reused independently — no single file should need to know how the others work internally.

```
┌─────────────────┐
│  Camera Capture  │   grabs frames, nothing else
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Face Landmarker  │   wraps MediaPipe, outputs raw landmarks
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Feature Extractor│   EAR, MAR, head pose — pure math, no I/O
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Fatigue Scorer    │   combines features → single risk score + state
└────────┬─────────┘
         │
   ┌─────┴─────┐
┌──▼───┐   ┌───▼────┐
│Alerts │   │ Logger  │   side effects live here, isolated from logic
└───────┘   └───┬────┘
                │
          ┌─────▼─────┐
          │ Dashboard  │   reads logs, never touches the live pipeline
          └────────────┘
```

**Guiding principles:**
- **Single responsibility** — each module does one job (capture ≠ detection ≠ scoring ≠ alerting).
- **Pure functions where possible** — feature extraction (EAR/MAR/head pose) takes landmarks in, returns numbers out, with no side effects. Easy to unit test without a camera.
- **Side effects isolated** — anything that touches hardware, disk, or UI (camera, alerts, logging, dashboard) is kept out of the core detection logic.
- **Config over hardcoding** — thresholds, window sizes, and alert timings live in one config file, not scattered through the code.

---

## Project Structure

```
driver-drowsiness-monitor/
├── src/
│   ├── capture/
│   │   └── camera.py            # camera I/O only
│   ├── detection/
│   │   ├── face_mesh.py         # MediaPipe wrapper
│   │   └── features.py          # EAR, MAR, head pose (pure functions)
│   ├── scoring/
│   │   ├── fatigue_score.py     # combines features into risk score
│   │   └── state_machine.py     # Normal → Warning → Critical logic
│   ├── alerts/
│   │   └── alert_manager.py     # sound/visual alerts, tiered escalation
│   ├── logging/
│   │   └── event_logger.py      # writes events to SQLite/CSV
│   ├── dashboard/
│   │   └── app.py               # Streamlit dashboard (reads logs only)
│   └── config.py                # all thresholds & constants, one place
├── data/
│   └── logs.db                  # session event log
├── tests/
│   └── ...                      # unit tests for pure logic (features, scoring)
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Face & landmark detection | **MediaPipe** | CPU-optimized, runs well without a GPU |
| Video I/O & drawing | **OpenCV** | Standard, fast, well-documented |
| Numeric computation | **NumPy** | Vector math for EAR/MAR/head pose |
| Live monitoring window | **OpenCV `imshow`** (not Streamlit) | Streamlit's rerun model fights a live video loop; a plain OpenCV window stays smooth on modest hardware |
| Event storage | **SQLite** | Cheap to query even as logs grow, unlike re-loading a growing CSV into memory every refresh |
| Dashboard | **Streamlit** + **Pandas** | Great fit for the *offline* trends/export view, where rerun-per-interaction isn't a problem |

> Note: the live camera loop and the dashboard are intentionally split into separate processes/entry points — they have very different performance needs and shouldn't compete for the same CPU cycles.

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/KodEx-SA/driver-drowsiness-monitor.git
cd driver-drowsiness-monitor

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the live monitor
python src/main.py

# 5. Run the dashboard (separate terminal)
streamlit run src/dashboard/app.py
```

On first run, sit normally and blink a few times when prompted — this calibrates your personal baseline EAR instead of relying on a fixed threshold.

---

## Configuration

All tunable values live in `src/config.py` — nothing is hardcoded inside detection logic:

```python
PERCLOS_WINDOW_SECONDS = 60       # rolling window for eye-closure percentage
EAR_CALIBRATION_SECONDS = 15      # baseline calibration duration
YAWN_MAR_THRESHOLD = 0.6
HEAD_NOD_PITCH_THRESHOLD = 15     # degrees
ALERT_COOLDOWN_SECONDS = 5
FRAME_SKIP = 2                    # process every Nth frame for performance
```

---

## Dashboard

The Streamlit dashboard reads only from the event log — it never touches the live camera pipeline. It shows:

- Fatigue score over the course of a session
- Frequency of eye-closure, yawn, and gaze-away events
- Trends across multiple sessions/days
- CSV export of raw event data

---

## Roadmap

- [ ] Head-pose based nod detection (earlier microsleep signal than eye closure)
- [ ] Composite fatigue score blending PERCLOS + yawn rate + nod rate + blink rate
- [ ] State machine with hysteresis to reduce alert flapping
- [ ] Progressive alert escalation (chime → alarm → optional haptic hook)
- [ ] Session context logging (lighting, time of day, trip duration)
- [ ] Low-light/IR camera handling with confidence fallback
- [ ] Unit tests for all pure feature-extraction functions

---

## Contributing

This project is being built incrementally, one clear, well-scoped step at a time — favoring readable, testable modules over clever shortcuts. If you'd like to contribute, please keep new logic inside the appropriate layer (detection, scoring, alerts, logging) and avoid mixing side effects into pure calculation functions.

---

## License

MIT License — free to use, modify, and build on.

---

Built by **Ashley Motsie** ([KodEx-SA](https://ashleydevhub.vercel.app))
