"""
navigator.py — Pure obstacle-avoidance logic.

Converts a LiDAR scan dict {angle_int: dist_mm} into a steering decision.

──────────────────────────────────────────────────────────────────────────
COORDINATE SYSTEM (top-down, sensor as origin):

  FRONT_HEADING degrees  →  directly in front of the walker
  FRONT_HEADING + 90     →  right side
  FRONT_HEADING + 270    →  left side   (same as FRONT_HEADING - 90)

Default: FRONT_HEADING = 0
  → works when the RPLIDAR connector faces backward and 180° = front.
  → set to 180 when connector faces forward, or 90/270 for sideways mount.

Measure once: hold a wall directly in front of the walker, note which
sensor angle gives the minimum distance, and set FRONT_HEADING to that.
──────────────────────────────────────────────────────────────────────────

Return value of decide():
  ('STOP',   0)          emergency — obstacle < STOP_MM in front
  ('STEER',  n)          steer  n  (-100..+100, neg=left, pos=right)
  ('CENTER', 0)          path clear — go straight
  ('NODATA', 0)          no scan data yet
"""

# ── Mounting ───────────────────────────────────────────────────────────
FRONT_HEADING  = 180    # sensor angle that points directly FORWARD

# ── Arc widths (half-widths in degrees) ───────────────────────────────
FRONT_HALF_W   = 35     # ±35° → 70° total front arc
SIDE_HALF_W    = 40     # ±40° → 80° total side arc (for free-space check)

# ── Distance thresholds (mm) ──────────────────────────────────────────
STOP_MM        = 350    # < 35 cm  → emergency stop
DANGER_MM      = 750    # < 75 cm  → start steering away

# ── Steering output ───────────────────────────────────────────────────
MAX_STEER      = 90     # maximum |angle| sent (out of 100)
MIN_STEER      = 15     # minimum steering strength to avoid micro-jitter
# Deadband: if the closest front obstacle is more than this many degrees
# off-centre in the front arc, steer directly away from it instead of
# consulting the side arcs.
DIRECTIONAL_DEADBAND_DEG = 12


# ── Helpers ───────────────────────────────────────────────────────────

def _in_arc(sensor_angle: int, logical_center_deg: int, half_width: int) -> bool:
    center = (FRONT_HEADING + logical_center_deg) % 360
    lo = (center - half_width) % 360
    hi = (center + half_width) % 360
    a  = sensor_angle % 360
    if lo <= hi:
        return lo <= a <= hi
    return a >= lo or a <= hi          # arc wraps around 0°


def _arc_min(scan: dict, logical_center_deg: int, half_width: int) -> float:
    distances = [
        dist for angle, dist in scan.items()
        if _in_arc(angle, logical_center_deg, half_width)
    ]
    return min(distances) if distances else float('inf')


def _arc_points(scan: dict, logical_center_deg: int, half_width: int) -> list:
    """Return [(sensor_angle, dist), ...] for points inside the arc."""
    return [
        (angle, dist) for angle, dist in scan.items()
        if _in_arc(angle, logical_center_deg, half_width)
    ]


# ── Main decision function ─────────────────────────────────────────────

def decide(scan: dict) -> tuple:
    """
    Returns (action, angle):
      action : 'STOP' | 'STEER' | 'CENTER' | 'NODATA'
      angle  : int  -100..+100   (negative=steer left, positive=steer right)
    """
    if not scan:
        return ('NODATA', 0)

    front_points = _arc_points(scan, 0, FRONT_HALF_W)

    if not front_points:
        return ('CENTER', 0)

    front_min_dist  = min(d for _, d in front_points)
    front_min_angle = min(front_points, key=lambda x: x[1])[0]

    # Relative angle of the closest obstacle inside the front arc
    # positive → to the right of centre, negative → to the left
    center_sensor = (FRONT_HEADING) % 360
    rel = (front_min_angle - center_sensor + 180) % 360 - 180

    # ── Emergency stop ───────────────────────────────────────────────
    if front_min_dist < STOP_MM:
        return ('STOP', 0)

    # ── Obstacle in danger zone ──────────────────────────────────────
    if front_min_dist < DANGER_MM:
        # Steering strength scales with proximity (0.0 → 1.0)
        proximity = (DANGER_MM - front_min_dist) / (DANGER_MM - STOP_MM)
        strength  = int(proximity * MAX_STEER)
        strength  = max(strength, MIN_STEER)

        if abs(rel) > DIRECTIONAL_DEADBAND_DEG:
            # Obstacle clearly off-centre: steer directly away from it
            # rel > 0  → obstacle on the right → steer LEFT (negative)
            # rel < 0  → obstacle on the left  → steer RIGHT (positive)
            direction = +1 if rel > 0 else -1
            return ('STEER', direction * strength)

        # Obstacle roughly centred — consult side arcs for free space
        # logical 90° = right, logical 270° = left
        right_min = _arc_min(scan,  90, SIDE_HALF_W)
        left_min  = _arc_min(scan, 270, SIDE_HALF_W)

        direction = -1 if right_min >= left_min else +1
        return ('STEER', direction * strength)

    # ── Path clear ───────────────────────────────────────────────────
    return ('CENTER', 0)
