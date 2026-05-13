# utils/tts_player.py
"""
Pre-cached WAV audio player for Smart Walker TTS.

Strategy:
  1. On first run, generate .wav files for all known phrases using espeak-ng.
  2. On playback, use `aplay` (non-blocking subprocess) for ~50ms latency.
  3. If a phrase has no cached file, fall back to espeak-ng directly.

Usage:
    player = TTSPlayer(sounds_dir="/home/lama/OAK-D/sounds")
    player.pregenerate()        # call once at startup
    player.play("Stop")         # non-blocking, <100ms
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Optional


# All fixed phrases that need pre-cached audio.
PHRASE_MAP: dict[str, str] = {
    "Stop":             "stop",
    "Free mode":        "free_mode",
    "Go forward":       "go_forward",
    "Slight left":      "slight_left",
    "Turn left":        "turn_left",
    "Hard left":        "hard_left",
    "Slight right":     "slight_right",
    "Turn right":       "turn_right",
    "Hard right":       "hard_right",
    "System authorized": "system_authorized",
    "System locked":    "system_locked",
}

# espeak-ng voice/speed settings (match original main.py settings)
_ESPEAK_VOICE = "en+f3"
_ESPEAK_SPEED = "135"


class TTSPlayer:
    """Non-blocking TTS player using pre-cached WAV files."""

    def __init__(self, sounds_dir: str = "/home/lama/OAK-D/sounds") -> None:
        self._dir = Path(sounds_dir)
        self._cache: dict[str, Path] = {}   # phrase → wav path
        self._lock = threading.Lock()
        self._current_proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]

    def pregenerate(self) -> None:
        """
        Generate .wav files for every phrase in PHRASE_MAP if they don't exist.
        Safe to call multiple times — skips files that already exist.
        """
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"[TTS] Cannot create sounds dir {self._dir}: {e}")
            return

        for phrase, stem in PHRASE_MAP.items():
            wav_path = self._dir / f"{stem}.wav"
            if not wav_path.exists():
                try:
                    # espeak-ng --stdout → pipe into WAV file via sox or direct wav output
                    # espeak-ng supports direct WAV output with -w flag
                    result = subprocess.run(
                        [
                            "espeak-ng",
                            "-v", _ESPEAK_VOICE,
                            "-s", _ESPEAK_SPEED,
                            "-w", str(wav_path),
                            phrase,
                        ],
                        capture_output=True,
                        timeout=10.0,
                    )
                    if result.returncode == 0 and wav_path.exists():
                        print(f"[TTS] Generated: {wav_path.name}")
                    else:
                        print(
                            f"[TTS] Failed to generate {wav_path.name}: "
                            f"{result.stderr.decode(errors='replace').strip()}"
                        )
                        # Remove partial file if any
                        wav_path.unlink(missing_ok=True)
                except FileNotFoundError:
                    print("[TTS] espeak-ng not found — cannot pre-generate audio")
                    return
                except Exception as e:
                    print(f"[TTS] Generation error for '{phrase}': {e}")
            else:
                print(f"[TTS] Cached: {wav_path.name}")

            if wav_path.exists():
                self._cache[phrase] = wav_path

        print(f"[TTS] Pre-cache complete: {len(self._cache)}/{len(PHRASE_MAP)} phrases ready")

    def play(self, text: str) -> None:
        """
        Play audio for `text`. Non-blocking.
        - Uses pre-cached WAV via `aplay` if available (~50ms latency).
        - Falls back to espeak-ng directly otherwise (~300-500ms latency).
        """
        wav_path = self._cache.get(text)

        if wav_path is not None and wav_path.exists():
            self._play_wav(wav_path)
        else:
            self._play_espeak(text)

    def _play_wav(self, wav_path: Path) -> None:
        """Play a WAV file with aplay in a daemon thread (non-blocking)."""
        def _run() -> None:
            try:
                # Kill any previously playing clip so we never queue up sounds
                with self._lock:
                    if self._current_proc is not None:
                        try:
                            self._current_proc.terminate()
                        except Exception:
                            pass
                    proc = subprocess.Popen(
                        ["aplay", "-q", str(wav_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._current_proc = proc
                proc.wait()
                with self._lock:
                    if self._current_proc is proc:
                        self._current_proc = None
            except FileNotFoundError:
                # aplay not available — fall back
                self._play_espeak_sync(str(wav_path))
            except Exception as e:
                print(f"[TTS] aplay error: {e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _play_espeak(self, text: str) -> None:
        """Fallback: play via espeak-ng directly in a daemon thread."""
        def _run() -> None:
            try:
                os.system(
                    f'espeak-ng -v {_ESPEAK_VOICE} -s {_ESPEAK_SPEED} '
                    f'"{text}" >/dev/null 2>&1'
                )
            except Exception as e:
                print(f"[TTS] espeak-ng error: {e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _play_espeak_sync(self, text: str) -> None:
        """Synchronous espeak-ng fallback (used inside _play_wav thread only)."""
        try:
            os.system(
                f'espeak-ng -v {_ESPEAK_VOICE} -s {_ESPEAK_SPEED} '
                f'"{text}" >/dev/null 2>&1'
            )
        except Exception as e:
            print(f"[TTS] espeak-ng fallback error: {e}")
