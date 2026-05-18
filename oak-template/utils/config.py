# utils/config.py
"""
Smart Walker v8.0 — Minimal Config
كل الأرقام هون، مش في أي مكان ثاني.
"""

import os
from dataclasses import dataclass


@dataclass
class Config:

    # ── Arduino ──────────────────────────────────────────────
    ARDUINO_PORT: str = "MOCK"
    ARDUINO_BAUD: int = 9600

    # ── Depth Thresholds (mm) ─────────────────────────────────
    # أقل من هيك → خطر → حاول تتحاشى
    DANGER_MM: float = 700.0
    # أقل من هيك → ابدأ تختار اتجاه
    CAUTION_MM: float = 1200.0
    # أبعد من هيك في الـ center → FREE (امشي براحتك)
    FREE_MM: float = 1400.0

    # ── Stability (anti-flicker) ──────────────────────────────
    # كم frame لازم نشوف نفس القرار قبل ما نبعته
    CONFIRM_FRAMES: int = 3
    # كم frame سيئة قبل ما نغير من FREE لـ GO/STOP
    EXIT_FREE_FRAMES: int = 4

    # ── ROI vertical band (normalized 0→1) ───────────────────
    # نتجاهل السقف والأرضية، نركز على "جسم الإنسان والكرسي"
    ROI_Y_TOP: float = 0.30   # 30% من الفوق → نتجاهل
    ROI_Y_BOT: float = 0.80   # 20% من التحت → نتجاهل (أرضية)

    # ── Features ─────────────────────────────────────────────
    DEBUG_DISPLAY: bool = True
    USE_TTS: bool = False

    # ── TTS Cooldown ─────────────────────────────────────────
    TTS_COOLDOWN_S: float = 2.5

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            ARDUINO_PORT=os.getenv("ARDUINO_PORT", "MOCK"),
            ARDUINO_BAUD=int(os.getenv("ARDUINO_BAUD", "9600")),
            DANGER_MM=float(os.getenv("DANGER_MM", "700")),
            CAUTION_MM=float(os.getenv("CAUTION_MM", "1200")),
            FREE_MM=float(os.getenv("FREE_MM", "1400")),
            CONFIRM_FRAMES=int(os.getenv("CONFIRM_FRAMES", "3")),
            EXIT_FREE_FRAMES=int(os.getenv("EXIT_FREE_FRAMES", "4")),
            ROI_Y_TOP=float(os.getenv("ROI_Y_TOP", "0.30")),
            ROI_Y_BOT=float(os.getenv("ROI_Y_BOT", "0.80")),
            DEBUG_DISPLAY=os.getenv("DEBUG_DISPLAY", "1") == "1",
            USE_TTS=os.getenv("USE_TTS", "0") == "1",
            TTS_COOLDOWN_S=float(os.getenv("TTS_COOLDOWN_S", "2.5")),
        )