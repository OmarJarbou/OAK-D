# utils/navigator.py
"""
Smart Walker v8.0 — Navigator
الدماغ البسيط: 3 أرقام داخل → قرار واحد خارج.

Input:  left_z, center_z, right_z  (mm من SpatialLocationCalculator)
Output: command string → "FREE" | "STOP" | "GO:L1" | "GO:R1" | "GO:L2" | "GO:R2"
"""

from dataclasses import dataclass
from typing import Optional
from utils.config import Config


@dataclass
class NavResult:
    command: str          # القرار النهائي (بعد التثبيت)
    raw_command: str      # القرار الخام (قبل التثبيت)
    left_z: float
    center_z: float
    right_z: float
    reason: str
    confirm_count: int


class Navigator:
    """
    منطق القرار:

    1. إذا center آمن وبعيد → FREE
    2. إذا center فيه حاجز:
       - شوف يمين أو يسار أيهم أوسع
       - إذا الفرق كبير → L2/R2 (دوران أكثر)
       - إذا الفرق صغير → L1/R1 (ميلان خفيف)
    3. إذا كل شي ضيق → STOP

    Stability: لازم CONFIRM_FRAMES متتالية لنفس القرار قبل الإرسال.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._last_stable: str = "NONE"
        self._candidate: str = "NONE"
        self._candidate_count: int = 0
        self._exit_free_count: int = 0

    def update(self, left_z: float, center_z: float, right_z: float) -> NavResult:
        cfg = self.cfg

        # ── اختر القرار الخام ───────────────────────────────────
        raw = self._decide(left_z, center_z, right_z)
        reason = self._build_reason(raw, left_z, center_z, right_z)

        # ── Stability: لازم N frames متتالية ─────────────────────
        stable = self._stabilize(raw)

        return NavResult(
            command=stable,
            raw_command=raw,
            left_z=left_z,
            center_z=center_z,
            right_z=right_z,
            reason=reason,
            confirm_count=self._candidate_count,
        )

    def _decide(self, left_z: float, center_z: float, right_z: float) -> str:
        cfg = self.cfg

        # ── 1. Center آمن تماماً → FREE ──────────────────────────
        if center_z >= cfg.FREE_MM:
            return "FREE"

        # ── 2. Center مقبول (caution zone) ───────────────────────
        #    إذا اليمين واليسار كلاهم أبعد من CAUTION → FREE برضو
        if center_z >= cfg.CAUTION_MM:
            if left_z >= cfg.CAUTION_MM and right_z >= cfg.CAUTION_MM:
                return "FREE"

        # ── 3. Center مسدود أو ضيق → وجه ──────────────────────────
        #    أيهم أفضل: يمين أو يسار؟
        best_side, best_z, other_z = self._best_side(left_z, right_z)

        if best_z < cfg.DANGER_MM:
            # كل الجهات خطر → STOP
            return "STOP"

        # كم الفرق بين الجهتين؟
        gap = best_z - other_z

        # إذا الفرق كبير (> 400mm) → دوران أكثر
        if gap > 400:
            turn = "L2" if best_side == "left" else "R2"
        else:
            turn = "L1" if best_side == "left" else "R1"

        return f"GO:{turn}"

    def _best_side(self, left_z: float, right_z: float):
        """يرجع (side, best_z, other_z)"""
        if left_z >= right_z:
            return "left", left_z, right_z
        else:
            return "right", right_z, left_z

    def _stabilize(self, raw: str) -> str:
        """
        Counter بسيط:
        - إذا نفس القرار تكرر CONFIRM_FRAMES مرات → قبله
        - إذا في FREE وجاء قرار مختلف، لازم EXIT_FREE_FRAMES مرات
        """
        cfg = self.cfg

        # تتبع المرشح الحالي
        if raw == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = raw
            self._candidate_count = 1

        # منطق الخروج من FREE (أصعب شوي عشان ما نخرج بسبب frame واحدة)
        if self._last_stable == "FREE" and raw != "FREE":
            self._exit_free_count += 1
            if self._exit_free_count < cfg.EXIT_FREE_FRAMES:
                return "FREE"  # لسا FREE
        else:
            self._exit_free_count = 0

        # قبول القرار بعد CONFIRM_FRAMES
        if self._candidate_count >= cfg.CONFIRM_FRAMES:
            self._last_stable = raw
            return raw

        return self._last_stable

    def _build_reason(self, cmd: str, left_z: float, center_z: float, right_z: float) -> str:
        return (
            f"{cmd} | "
            f"L={left_z/1000:.2f}m "
            f"C={center_z/1000:.2f}m "
            f"R={right_z/1000:.2f}m"
        )

    def reset(self):
        self._last_stable = "NONE"
        self._candidate = "NONE"
        self._candidate_count = 0
        self._exit_free_count = 0