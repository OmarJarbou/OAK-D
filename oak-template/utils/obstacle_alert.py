import time

ZONE_PRIORITY = {"GREEN": 0, "YELLOW": 1, "RED": 2}

class ObstacleAlert:
    """
    Controls alert frequency per zone.
    RED fires at most every red_cooldown seconds.
    YELLOW fires at most every yellow_cooldown seconds.
    GREEN never fires (no action needed).
    """
    def __init__(self, red_cooldown: float = 1.0, yellow_cooldown: float = 3.0,
                 debounce_frames: int = 3):
        self.cooldowns        = {"RED": red_cooldown, "YELLOW": yellow_cooldown}
        self.last_alert       = {"RED": 0.0, "YELLOW": 0.0}
        self.active           = {"RED": False, "YELLOW": False}
        self._candidate       = "GREEN"
        self._candidate_count = 0
        self._committed       = "GREEN"
        self.debounce_frames  = debounce_frames

    def should_alert(self, zone: str, distance_m: float) -> bool:
        if zone == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate       = zone
            self._candidate_count = 1

        # Commit immediately if zone is MORE dangerous than current — no debounce wait
        escalating = ZONE_PRIORITY[zone] > ZONE_PRIORITY[self._committed]
        if escalating or self._candidate_count >= self.debounce_frames:
            self._committed = zone

        committed = self._committed
        now = time.time()

        for z in ("RED", "YELLOW"):
            if z != committed and self.active[z]:
                self.active[z]     = False
                self.last_alert[z] = 0.0

        if committed == "GREEN":
            return False

        self.active[committed] = True
        if now - self.last_alert[committed] >= self.cooldowns[committed]:
            self.last_alert[committed] = now
            return True
        return False

    @property
    def committed_zone(self):
        return self._committed