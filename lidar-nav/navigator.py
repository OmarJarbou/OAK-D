"""
navigator.py — Follow the Gap Method (FTG) obstacle avoidance.

Algorithm (per scan cycle):
  1. Restrict scan to the forward working arc  (±SCAN_HALF_W°)
  2. Emergency STOP — if any reading < STOP_MM
  3. If nothing within DANGER_MM → straight ahead (CENTER)
  4. Find the closest point  P  in the working arc
  5. Safety bubble: zero out every point whose physical distance to P
     is less than BUBBLE_RADIUS_MM
       angular_radius = arcsin( min(BUBBLE_R / dist(P), 1) )
     → thin obstacles (chair legs, poles) become invisible, revealing
       the gap beside them
  6. Collect all contiguous non-zero sequences  → "gaps"
  7. Select the largest gap
  8. Steer toward the angular midpoint of that gap

Why midpoint (not max-distance point):
  A walker moves slowly — midpoint keeps the path centred in the gap
  which feels natural and avoids hugging one wall.

Steering output (CMD:ANGLE scale):
  -100  = full left      0  = straight ahead      +100  = full right

──────────────────────────────────────────────────────────────────────────
FRONT_HEADING calibration:
  Face a flat wall directly ahead, run a quick scan print, note which
  sensor angle gives the minimum distance — set FRONT_HEADING to that.
  Common values:  0° (connector faces backward)  /  180° (connector forward)
──────────────────────────────────────────────────────────────────────────
"""

import math

# ── Mounting ────────────────────────────────────────────────────────────
FRONT_HEADING    = 0      # sensor angle that equals "forward"

# ── Working arc ─────────────────────────────────────────────────────────
SCAN_HALF_W      = 100    # ±100° → 200° forward hemisphere (no reverse)

# ── Distance thresholds (mm) ────────────────────────────────────────────
STOP_MM          = 350    # < 35 cm  → emergency stop  (runs BEFORE FTG)
DANGER_MM        = 800    # < 80 cm  → FTG activates;  above this = CENTER

# ── Safety bubble ───────────────────────────────────────────────────────
BUBBLE_RADIUS_MM = 280    # physical radius of bubble around the closest point
                           # ≈ half the walker width + small margin
                           # bigger = safer but harder to fit through doorways

# ── Gap quality ─────────────────────────────────────────────────────────
MIN_GAP_POINTS   = 8      # ignore gaps narrower than this many scan readings

# ── Steering output ─────────────────────────────────────────────────────
MAX_STEER        = 90     # max |angle| in CMD:ANGLE scale
MIN_STEER        = 6      # deadband — suppress micro-jitter


# ── Internal helpers ────────────────────────────────────────────────────

def _rel(sensor_angle: int) -> float:
    """Signed angle relative to FRONT_HEADING. Right = positive."""
    return (sensor_angle - FRONT_HEADING + 180) % 360 - 180


def _in_working_arc(sensor_angle: int) -> bool:
    return abs(_rel(sensor_angle)) <= SCAN_HALF_W


def _apply_bubble(working: list, closest_angle: int, closest_dist: float) -> list:
    """
    Zero out every point within the safety bubble around the closest obstacle.
    Angular radius of the bubble (degrees):
        β = arcsin( min( BUBBLE_RADIUS_MM / dist, 1 ) )
    """
    beta = math.degrees(math.asin(min(BUBBLE_RADIUS_MM / closest_dist, 1.0)))
    masked = []
    for (angle, dist) in working:
        ang_diff = abs((angle - closest_angle + 180) % 360 - 180)
        masked.append((angle, 0.0 if ang_diff <= beta else dist))
    return masked


def _find_gaps(masked: list) -> list:
    """Return list of gaps; each gap = [(angle, dist), ...] with dist > 0."""
    gaps, current = [], []
    for (angle, dist) in masked:
        if dist > 0:
            current.append((angle, dist))
        else:
            if len(current) >= MIN_GAP_POINTS:
                gaps.append(current)
            current = []
    if len(current) >= MIN_GAP_POINTS:
        gaps.append(current)
    return gaps


def _gap_midpoint_angle(gap: list) -> float:
    """Angular midpoint of the gap — keeps the walker centred in the opening."""
    angles = [a for a, _ in gap]
    return (angles[0] + angles[-1]) / 2.0


# ── Public API ──────────────────────────────────────────────────────────

def decide(scan: dict) -> tuple:
    """
    Returns (action, angle):
      ('STOP',   0)   emergency — obstacle < STOP_MM anywhere in working arc
      ('STEER',  n)   steer n   (n in -100..+100)
      ('CENTER', 0)   path clear — go straight
      ('NODATA', 0)   no scan data yet
    """
    if not scan:
        return ('NODATA', 0)

    # ── 1. Build working arc sorted left → right ────────────────────
    working = sorted(
        [(a, d) for a, d in scan.items() if _in_working_arc(a) and d > 0],
        key=lambda x: _rel(x[0])
    )

    if not working:
        return ('CENTER', 0)

    min_dist = min(d for _, d in working)

    # ── 2. Emergency stop ────────────────────────────────────────────
    if min_dist < STOP_MM:
        return ('STOP', 0)

    # ── 3. Path clear ────────────────────────────────────────────────
    if min_dist > DANGER_MM:
        return ('CENTER', 0)

    # ── 4. Closest point ─────────────────────────────────────────────
    closest_angle, closest_dist = min(working, key=lambda x: x[1])

    # ── 5. Apply safety bubble ───────────────────────────────────────
    masked = _apply_bubble(working, closest_angle, closest_dist)

    # ── 6. Find gaps ─────────────────────────────────────────────────
    gaps = _find_gaps(masked)

    if not gaps:
        return ('STOP', 0)      # completely surrounded — no gap found

    # ── 7. Largest gap ───────────────────────────────────────────────
    best_gap = max(gaps, key=len)

    # ── 8. Steer toward gap midpoint ─────────────────────────────────
    target_angle = _gap_midpoint_angle(best_gap)
    rel    = _rel(target_angle)                      # degrees, right=positive
    steer  = int(rel / SCAN_HALF_W * MAX_STEER)
    steer  = max(-MAX_STEER, min(MAX_STEER, steer))

    if abs(steer) < MIN_STEER:
        return ('CENTER', 0)

    return ('STEER', steer)
