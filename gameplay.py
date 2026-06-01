from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pygame

from audio_analysis import AnalysisResult, NoteEvent, convert_song_to_rate_wav, convert_song_to_wav

DEFAULT_KEYBINDS = {
    "lane_a": "a",
    "lane_s": "s",
    "lane_d": "d",
    "lane_f": "f",
    "lane_space": "space",
    "speed_down": "j",
    "speed_up": "l",
    "pause": "h",
}

SPEED_STEPS = [0.12, 0.16, 0.2, 0.26, 0.34, 0.44, 0.56, 0.7, 0.86, 1.0, 1.16, 1.34, 1.56, 1.82, 2.14, 2.5]
MAX_CUE_VOLUME = 2.5
MAX_SONG_VOLUME = 1.0
MIN_SONG_VOLUME = 0.0
SONG_VOLUME_STEP = 0.05
CUE_VOLUME_STEP = 0.05
SIMULTANEOUS_TOLERANCE_S = 0.014
APPROACH_CUE_LEAD_S = 0.08


@dataclass
class GameplayResult:
    score: int
    hits: int
    misses: int
    max_combo: int
    bpm: float
    perfect_hits: int
    good_hits: int
    ok_hits: int
    failed: bool


class GameplayCueBank:
    def __init__(
        self,
        assets_root: str | Path,
        cue_type: str = "both",
        cue_volume: float = 0.1,
        cue_panning: bool = False,
    ):
        self.assets_root = Path(assets_root)
        self.cue_type = cue_type
        self.cue_gain = 2.15
        self.cue_volume = max(0.0, min(MAX_CUE_VOLUME, float(cue_volume)))
        self.cue_panning = bool(cue_panning)
        self._lane_pan = {
            "A": -1.0,
            "S": -0.55,
            "D": 0.55,
            "F": 1.0,
            "SPACE": 0.0,
        }
        self._lane_sounds = {
            "A": self._load_sound("a.mp3"),
            "S": self._load_sound("s.mp3"),
            "D": self._load_sound("d.mp3"),
            "F": self._load_sound("f.mp3"),
            "SPACE": self._load_sound("space.mp3"),
        }
        self._hit_sound = self._load_sound("hit.mp3")
        self._miss_sound = self._load_sound("miss.mp3")
        self._combo_sound = self._load_sound("combo.mp3")
        if self._hit_sound is None:
            self._hit_sound = self._make_tone(920.0, 1120.0, 0.055)
        if self._miss_sound is None:
            self._miss_sound = self._make_tone(240.0, 105.0, 0.16)
        if self._combo_sound is None:
            self._combo_sound = self._make_tone(660.0, 1320.0, 0.13)

    def _load_sound(self, filename: str) -> pygame.mixer.Sound | None:
        path = self.assets_root / "sounds" / "gameplay sounds" / filename
        if not path.exists():
            return None
        try:
            return pygame.mixer.Sound(path.as_posix())
        except pygame.error:
            return None

    def _make_tone(self, start_hz: float, end_hz: float, duration_s: float) -> pygame.mixer.Sound | None:
        mixer_init = pygame.mixer.get_init()
        if mixer_init is None:
            return None
        sample_rate, _, channels = mixer_init
        sample_count = max(1, int(sample_rate * duration_s))
        frequencies = np.linspace(start_hz, end_hz, sample_count, dtype=np.float64)
        phase = np.cumsum((2.0 * np.pi * frequencies) / sample_rate)
        envelope = np.linspace(1.0, 0.0, sample_count, dtype=np.float64)
        waveform = (np.sin(phase) * envelope * 11000.0).astype(np.int16)
        samples = np.column_stack([waveform] * channels) if channels > 1 else waveform
        try:
            return pygame.sndarray.make_sound(np.ascontiguousarray(samples))
        except pygame.error:
            return None

    def set_mode(self, cue_type: str) -> None:
        self.cue_type = cue_type

    def set_master_volume(self, cue_volume: float) -> None:
        self.cue_volume = max(0.0, min(MAX_CUE_VOLUME, float(cue_volume)))

    def set_cue_panning(self, enabled: bool) -> None:
        self.cue_panning = bool(enabled)

    def audio_enabled(self) -> bool:
        return self.cue_type in {"both", "audio", "audio_letters", "audio_only"}

    def lane_labels_visible(self) -> bool:
        return self.cue_type in {"both", "visual", "audio_letters", "letters_only"}

    def _scaled_volume(self, volume: float) -> float:
        return max(0.0, min(1.0, volume * self.cue_gain * self.cue_volume))

    def _channel_volume(self, lane: str | None, volume: float) -> tuple[float, float]:
        base = self._scaled_volume(volume)
        if not self.cue_panning or lane is None:
            return base, base
        pan = float(self._lane_pan.get(lane, 0.0))
        pan = max(-1.0, min(1.0, pan))
        if pan < 0.0:
            return base, base * (1.0 + pan)
        if pan > 0.0:
            return base * (1.0 - pan), base
        return base, base

    def _play_sound(self, sound: pygame.mixer.Sound, volume: float, lane: str | None = None) -> None:
        channel = sound.play()
        if channel is None:
            return
        left, right = self._channel_volume(lane, volume)
        channel.set_volume(left, right)

    def play_lane(self, lane: str, volume: float = 0.5) -> None:
        if not self.audio_enabled():
            return
        sound = self._lane_sounds.get(lane)
        if sound is None:
            return
        self._play_sound(sound, volume, lane=lane)

    def play_hit(self, lane: str) -> None:
        if not self.audio_enabled():
            return
        self.play_lane(lane, volume=0.78)
        if self._hit_sound is not None:
            self._play_sound(self._hit_sound, 1.0, lane=None)

    def play_miss(self, volume: float = 1.0) -> None:
        if not self.audio_enabled():
            return
        if self._miss_sound is not None:
            self._play_sound(self._miss_sound, volume, lane=None)

    def play_error(self, volume: float = 1.0) -> None:
        self.play_miss(volume=volume)

    def play_combo(self) -> None:
        if not self.audio_enabled():
            return
        if self._combo_sound is not None:
            self._play_sound(self._combo_sound, 1.0, lane=None)


class GameSession:
    def __init__(
        self,
        screen: pygame.Surface,
        assets_root: str | Path,
        song_path: str | Path,
        analysis: AnalysisResult,
        announce: Callable[[str], None],
        announce_lane: Callable[[str], None] | None = None,
        cue_type: str = "both",
        cue_volume: float = 0.1,
        cue_panning: bool = False,
        speak_letters: bool = True,
        music_enabled: bool = True,
        starting_song_volume: float = 1.0,
        reduced_inputs: bool = False,
        difficulty_level: int = 5,
        input_latency_s: float = 0.0,
        initial_speed_index: int | None = None,
        keybinds: dict[str, str] | None = None,
    ):
        self.screen = screen
        self.assets_root = Path(assets_root)
        self.song_path = Path(song_path)
        self._playback_path = self.song_path
        self._base_playback_path = self.song_path
        self._temp_playback_file: Path | None = None
        self.analysis = analysis
        self.announce = announce
        self.announce_lane = announce_lane or announce
        self.cue_type = cue_type
        self.speak_letters = bool(speak_letters)
        self.reduced_inputs = bool(reduced_inputs)
        self.difficulty_level = max(1, min(10, int(difficulty_level)))
        self.input_latency_s = max(0.0, min(0.75, float(input_latency_s)))
        self.keybinds = self._normalize_keybinds(keybinds)

        self.hit_window = 0.18
        self.space_hit_window = 0.15
        self.perfect_window = 0.07
        self.good_window = 0.12
        self.approach_lead_s = 0.62
        self.speed_steps = list(SPEED_STEPS)
        self.speed_index = self.speed_steps.index(1.0)
        if initial_speed_index is not None:
            self.speed_index = max(0, min(len(self.speed_steps) - 1, int(initial_speed_index)))
        self._timeline_scale = 1.0
        self._playback_start_chart_time_s = 0.0
        self.score = 0
        self.hits = 0
        self.misses = 0
        self.combo = 0
        self.max_combo = 0
        self.perfect_hits = 0
        self.good_hits = 0
        self.ok_hits = 0
        self.paused = False
        self.finished = False
        self.failed = False
        self.complete_announced = False
        self.last_judgment = ""
        self.last_judgment_until = 0.0
        self.health_max = 150.0
        self.health = self.health_max
        self.fail_miss_penalty = 1.9
        self.fail_strike_penalty = 1.2
        self.fail_hit_recovery = 3.2
        self.lane_colors = self._lane_colors_for_mode()
        self.lanes = self._lanes_for_mode()
        self.lane_key_map = self._lane_key_map_for_mode()
        self._apply_difficulty_to_notes()
        self._apply_input_mode_to_notes()
        self.last_press_s = {lane: -9.0 for lane in self.lanes}
        self.input_cooldown_s = 0.03
        self.cues = GameplayCueBank(
            self.assets_root,
            cue_type=cue_type,
            cue_volume=cue_volume,
            cue_panning=cue_panning,
        )
        self.music_enabled = bool(music_enabled)
        self.song_volume = max(MIN_SONG_VOLUME, min(MAX_SONG_VOLUME, float(starting_song_volume)))
        self._fallback_song_time_s = 0.0
        self._fallback_last_tick_ms = pygame.time.get_ticks()
        self._configure_tempo_windows()
        self._start_audio()

    def _normalize_keybinds(self, keybinds: dict[str, str] | None) -> dict[str, str]:
        normalized = dict(DEFAULT_KEYBINDS)
        if not isinstance(keybinds, dict):
            return normalized
        for action, value in keybinds.items():
            if action not in normalized or not isinstance(value, str):
                continue
            key_name = value.strip().lower()
            if key_name:
                normalized[action] = key_name
        return normalized

    def _key_code(self, action: str) -> int:
        key_name = self.keybinds.get(action, DEFAULT_KEYBINDS[action])
        try:
            return pygame.key.key_code(key_name)
        except ValueError:
            return pygame.key.key_code(DEFAULT_KEYBINDS[action])

    def _lanes_for_mode(self) -> list[str]:
        if self.reduced_inputs:
            return ["S", "D", "SPACE"]
        return ["A", "S", "D", "F", "SPACE"]

    def _lane_colors_for_mode(self) -> dict[str, tuple[int, int, int]]:
        if self.reduced_inputs:
            return {
                "S": (130, 230, 160),
                "D": (255, 220, 120),
                "SPACE": (210, 210, 210),
            }
        return {
            "A": (120, 200, 255),
            "S": (130, 230, 160),
            "D": (255, 220, 120),
            "F": (255, 160, 120),
            "SPACE": (210, 210, 210),
        }

    def _lane_key_map_for_mode(self) -> dict[int, str]:
        key_a = self._key_code("lane_a")
        key_s = self._key_code("lane_s")
        key_d = self._key_code("lane_d")
        key_f = self._key_code("lane_f")
        key_space = self._key_code("lane_space")
        if self.reduced_inputs:
            return {
                key_a: "S",
                key_s: "S",
                key_d: "D",
                key_f: "D",
                key_space: "SPACE",
            }
        return {
            key_a: "A",
            key_s: "S",
            key_d: "D",
            key_f: "F",
            key_space: "SPACE",
        }

    def _map_lane_for_mode(self, lane: str) -> str:
        if not self.reduced_inputs:
            return lane
        if lane in {"A", "S"}:
            return "S"
        if lane in {"D", "F"}:
            return "D"
        return "SPACE"

    def _apply_input_mode_to_notes(self) -> None:
        for note in self.analysis.notes:
            note.lane = self._map_lane_for_mode(note.lane)
            note.hit = False
            note.judged = False
            note.approach_cued = False

    def _difficulty_note_keep_ratio(self) -> float:
        if self.difficulty_level <= 1:
            return 0.36
        if self.difficulty_level == 2:
            return 0.48
        if self.difficulty_level == 3:
            return 0.62
        if self.difficulty_level == 4:
            return 0.8
        return 1.0

    def _apply_difficulty_to_notes(self) -> None:
        ratio = self._difficulty_note_keep_ratio()
        if ratio >= 0.999:
            return
        source = list(self.analysis.notes)
        kept: list[NoteEvent] = []
        carry = 0.0
        for note in source:
            if note.lane == "SPACE":
                kept.append(note)
                continue
            carry += ratio
            if carry >= 1.0:
                kept.append(note)
                carry -= 1.0
        if not kept and source:
            kept.append(source[0])
        self.analysis.notes = kept

    def _difficulty_speed_multiplier(self) -> float:
        offset = self.difficulty_level - 5
        if offset == 0:
            return 1.0
        if offset > 0:
            return 1.0 + (offset * 0.07)
        return max(0.72, 1.0 + (offset * 0.05))

    def _approach_cue_stride(self) -> int:
        if self.difficulty_level <= 1:
            return 5
        if self.difficulty_level == 2:
            return 4
        if self.difficulty_level in {3, 4}:
            return 3
        if self.difficulty_level in {5, 6}:
            return 2
        return 1

    def _visual_cues_enabled(self) -> bool:
        return self.difficulty_level >= 3

    def _hud_guidance_enabled(self) -> bool:
        return self.difficulty_level >= 5

    def _configure_tempo_windows(self) -> None:
        beat_s = 60.0 / max(1e-6, self.analysis.bpm)
        # Calibration absorbs input delay, so hits can require precise timing.
        self.hit_window = float(np.clip(beat_s * 0.10, 0.065, 0.10))
        self.space_hit_window = float(np.clip(self.hit_window * 0.95, 0.06, 0.095))
        self.perfect_window = float(np.clip(self.hit_window * 0.34, 0.02, 0.035))
        self.good_window = float(np.clip(self.hit_window * 0.68, 0.04, 0.065))
        self.approach_lead_s = APPROACH_CUE_LEAD_S
        if self.difficulty_level < 5:
            scale = 1.0 + ((5 - self.difficulty_level) * 0.05)
            self.hit_window *= scale
            self.space_hit_window *= scale
            self.perfect_window *= scale
            self.good_window *= scale
        elif self.difficulty_level > 5:
            scale = max(0.78, 1.0 - ((self.difficulty_level - 5) * 0.04))
            self.hit_window *= scale
            self.space_hit_window *= scale
            self.perfect_window *= scale
            self.good_window *= scale

    def _reset_music_stream(self) -> None:
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        try:
            unload = getattr(pygame.mixer.music, "unload", None)
            if callable(unload):
                unload()
        except pygame.error:
            pass

    def _speed_multiplier(self) -> float:
        base = self.speed_steps[self.speed_index]
        return float(np.clip(base * self._difficulty_speed_multiplier(), 0.1, 3.0))

    def _timeline_rate(self) -> float:
        return self._speed_multiplier()

    def _speed_text(self) -> str:
        return f"{self._timeline_scale:.2f}x"

    def _change_speed(self, delta: int) -> None:
        current_chart_time = self._song_time()
        old_rate = self._timeline_scale
        prev = self.speed_index
        self.speed_index = max(0, min(len(self.speed_steps) - 1, self.speed_index + delta))
        if self.speed_index == prev:
            edge = "lowest" if delta < 0 else "highest"
            self.announce(f"Speed {edge}. {self._speed_text()}")
            return
        new_rate = self._timeline_rate()
        if abs(new_rate - old_rate) > 1e-4:
            self._switch_playback_rate(current_chart_time, new_rate)
        self.announce(f"Speed {self._speed_text()}")

    def _spoken_lane_enabled(self) -> bool:
        return self.speak_letters

    def _lane_speech_text(self, lane: str) -> str:
        if lane == "SPACE":
            return "Space"
        return lane

    def _apply_miss_penalty(self, penalty: float) -> None:
        self.health = max(0.0, self.health - penalty)
        if self.health <= 0.0 and not self.failed:
            self.failed = True
            self.finished = True
            self.announce("Failed out. Too many misses.")

    def _start_audio(self) -> None:
        self._reset_music_stream()
        self._base_playback_path = self._resolve_playback_path(self.song_path)
        requested_rate = self._timeline_rate()
        self._playback_path, self._timeline_scale = self._resolve_rate_playback_path(
            self._base_playback_path,
            requested_rate,
        )
        self._playback_start_chart_time_s = 0.0
        load_error: Exception | None = None
        try:
            pygame.mixer.music.load(self._playback_path.as_posix())
        except pygame.error as exc:
            load_error = exc
            # Secondary fallback: force conversion from the original selected song.
            try:
                forced = convert_song_to_wav(self.song_path, self.song_path.parent / ".converted")
                pygame.mixer.music.load(forced.as_posix())
                self._playback_path = forced
                self._timeline_scale = 1.0
            except Exception as forced_exc:
                raise pygame.error(
                    f"Game could not load audio file: {self._playback_path}. "
                    f"Fallback conversion failed: {forced_exc}"
                ) from load_error
        self._apply_song_volume()
        pygame.mixer.music.play()
        self._fallback_song_time_s = 0.0
        self._fallback_last_tick_ms = pygame.time.get_ticks()
        self.announce(
            f"Song started. Detected type {self.analysis.genre}. "
            f"Lane profile {self.analysis.lane_profile}. "
            f"Difficulty {self.difficulty_level}. "
            f"Shanra events detected {self.analysis.shanra_hits}."
        )

    def _resolve_playback_path(self, path: Path) -> Path:
        cache_dir = path.parent / ".converted"
        if path.suffix.lower() not in {".wav", ".ogg", ".mp3"}:
            try:
                return convert_song_to_wav(path, cache_dir)
            except Exception:
                pass
        try:
            pygame.mixer.Sound(path.as_posix())
            return path
        except pygame.error:
            pass

        try:
            converted = convert_song_to_wav(path, cache_dir)
            pygame.mixer.Sound(converted.as_posix())
            return converted
        except Exception as exc:
            raise pygame.error(f"Unsupported playback format: {exc}") from exc

    def _resolve_rate_playback_path(self, base_path: Path, rate: float) -> tuple[Path, float]:
        if abs(rate - 1.0) < 1e-4:
            return base_path, 1.0
        try:
            converted = convert_song_to_rate_wav(base_path, rate, cache_dir=base_path.parent / ".converted")
            return converted, rate
        except Exception:
            return base_path, 1.0

    def _switch_playback_rate(self, chart_time_s: float, new_rate: float) -> None:
        was_paused = self.paused
        self._playback_path, self._timeline_scale = self._resolve_rate_playback_path(
            self._base_playback_path,
            new_rate,
        )
        self._playback_start_chart_time_s = max(0.0, float(chart_time_s))
        try:
            pygame.mixer.music.load(self._playback_path.as_posix())
            self._apply_song_volume()
            pygame.mixer.music.play()
        except pygame.error:
            self._timeline_scale = 1.0
            self._playback_path = self._base_playback_path
            self._playback_start_chart_time_s = max(0.0, float(chart_time_s))
            pygame.mixer.music.load(self._playback_path.as_posix())
            self._apply_song_volume()
            pygame.mixer.music.play()

        if not was_paused:
            # Position in the playback file timeline, derived from chart timeline.
            target_playback_pos_s = self._playback_start_chart_time_s / max(1e-6, self._timeline_scale)
            if target_playback_pos_s > 0.01:
                try:
                    pygame.mixer.music.set_pos(target_playback_pos_s)
                except pygame.error:
                    pass
        if was_paused:
            pygame.mixer.music.pause()
        self._fallback_song_time_s = self._playback_start_chart_time_s
        self._fallback_last_tick_ms = pygame.time.get_ticks()

    def _song_time(self) -> float:
        if self.paused:
            return self._fallback_song_time_s
        pos_ms = pygame.mixer.music.get_pos()
        if pos_ms >= 0:
            playback_s = pos_ms / 1000.0
            self._fallback_song_time_s = (
                self._playback_start_chart_time_s + (playback_s * self._timeline_scale)
            )
            self._fallback_last_tick_ms = pygame.time.get_ticks()
            return self._fallback_song_time_s
        now_tick = pygame.time.get_ticks()
        delta_s = max(0.0, (now_tick - self._fallback_last_tick_ms) / 1000.0)
        self._fallback_last_tick_ms = now_tick
        self._fallback_song_time_s += delta_s * self._timeline_scale
        return self._fallback_song_time_s

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            pygame.mixer.music.pause()
            self.announce("Paused")
        else:
            pygame.mixer.music.unpause()
            self._fallback_last_tick_ms = pygame.time.get_ticks()
            self.announce("Resumed")

    def _apply_song_volume(self) -> None:
        target = self.song_volume if self.music_enabled else 0.0
        pygame.mixer.music.set_volume(max(MIN_SONG_VOLUME, min(MAX_SONG_VOLUME, target)))

    def _change_song_volume(self, delta: float) -> None:
        previous = self.song_volume
        updated = max(MIN_SONG_VOLUME, min(MAX_SONG_VOLUME, previous + delta))
        if abs(updated - previous) < 1e-6:
            edge = "minimum" if delta < 0 else "maximum"
            self.announce(f"Song volume {edge}. {int(round(updated * 100.0))} percent")
            return
        self.song_volume = updated
        self._apply_song_volume()
        self.announce(f"Song volume {int(round(updated * 100.0))} percent")

    def _change_cue_volume(self, delta: float) -> None:
        previous = self.cues.cue_volume
        updated = max(0.0, min(MAX_CUE_VOLUME, previous + delta))
        if abs(updated - previous) < 1e-6:
            edge = "minimum" if delta < 0 else "maximum"
            self.announce(f"Cue volume {edge}. {int(round(updated * 100.0))} percent")
            return
        self.cues.set_master_volume(updated)
        self.announce(f"Cue volume {int(round(updated * 100.0))} percent")

    def _judge_lane_hit(self, lane: str) -> None:
        now = max(0.0, self._song_time() - self.input_latency_s)
        lane_window = self.space_hit_window if lane == "SPACE" else self.hit_window
        best: NoteEvent | None = None
        best_delta = lane_window + 1.0
        for note in self.analysis.notes:
            if note.judged or note.lane != lane:
                continue
            if abs(note.time_s - now) > lane_window:
                continue
            delta = abs(note.time_s - now)
            if lane == "SPACE":
                signed_delta = note.time_s - now
                if signed_delta < -0.06:
                    delta += 0.008
            if delta < best_delta:
                best_delta = delta
                best = note
        if best is None and lane != "SPACE":
            best, best_delta = self._nearest_simultaneous_pitched_cluster(now, lane_window)
        if best is not None and best_delta <= lane_window:
            for note in self._simultaneous_cluster(best):
                note.judged = True
                note.hit = True
            if best_delta <= self.perfect_window:
                points = 150
                judgment = "PERFECT"
                self.perfect_hits += 1
            elif best_delta <= self.good_window:
                points = 110
                judgment = "GOOD"
                self.good_hits += 1
            else:
                points = 80
                judgment = "OK"
                self.ok_hits += 1
            self.hits += 1
            self.score += int(points + self.combo * 2)
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            self.health = min(self.health_max, self.health + self.fail_hit_recovery)
            self.cues.play_hit(lane)
            self.last_judgment = judgment
            self.last_judgment_until = now + 0.45
            if self.combo > 0 and self.combo % 10 == 0:
                self.cues.play_combo()
                self.announce(f"Combo {self.combo}")
        else:
            self.misses += 1
            self.score = max(0, self.score - 15)
            self.combo = 0
            self._apply_miss_penalty(self.fail_strike_penalty)
            self.cues.play_miss()
            self.last_judgment = "MISS"
            self.last_judgment_until = now + 0.45

    def _simultaneous_cluster(self, anchor: NoteEvent) -> list[NoteEvent]:
        cluster: list[NoteEvent] = [anchor]
        if anchor.lane == "SPACE":
            return cluster
        for note in self.analysis.notes:
            if note is anchor or note.judged or note.lane == "SPACE":
                continue
            if abs(note.time_s - anchor.time_s) <= SIMULTANEOUS_TOLERANCE_S:
                cluster.append(note)
        return cluster

    def _nearest_simultaneous_pitched_cluster(self, now: float, window: float) -> tuple[NoteEvent | None, float]:
        best: NoteEvent | None = None
        best_delta = window + 1.0
        for note in self.analysis.notes:
            if note.judged or note.lane == "SPACE":
                continue
            delta = abs(note.time_s - now)
            if delta > window:
                continue
            cluster_size = 0
            for other in self.analysis.notes:
                if other.judged or other.lane == "SPACE":
                    continue
                if abs(other.time_s - note.time_s) <= SIMULTANEOUS_TOLERANCE_S:
                    cluster_size += 1
            if cluster_size < 2:
                continue
            if delta < best_delta:
                best = note
                best_delta = delta
        return best, best_delta

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type != pygame.KEYDOWN:
            return False

        if event.key == pygame.K_DOWN:
            self._change_song_volume(-SONG_VOLUME_STEP)
            return True
        if event.key == pygame.K_UP:
            self._change_song_volume(SONG_VOLUME_STEP)
            return True
        if event.key == pygame.K_LEFTBRACKET:
            self._change_cue_volume(-CUE_VOLUME_STEP)
            return True
        if event.key == pygame.K_RIGHTBRACKET:
            self._change_cue_volume(CUE_VOLUME_STEP)
            return True
        if event.key == self._key_code("pause"):
            self._toggle_pause()
            return True
        if event.key == self._key_code("speed_down"):
            self._change_speed(-1)
            return True
        if event.key == self._key_code("speed_up"):
            self._change_speed(1)
            return True

        if self.paused:
            return True

        lane = self.lane_key_map.get(event.key)
        if lane is not None:
            now = self._song_time()
            if now - self.last_press_s[lane] < self.input_cooldown_s:
                return True
            self.last_press_s[lane] = now
            self._judge_lane_hit(lane)
            return True
        return False

    def update(self) -> bool:
        if self.finished:
            return True
        if not self.paused:
            now = self._song_time()
            stride = self._approach_cue_stride()
            for idx, note in enumerate(self.analysis.notes):
                if note.judged:
                    continue
                if (not note.approach_cued) and (0.0 <= (note.time_s - now) <= self.approach_lead_s):
                    note.approach_cued = True
                    if idx % stride == 0:
                        self.cues.play_lane(note.lane, volume=0.45)
                    if self._spoken_lane_enabled() and idx % stride == 0:
                        self.announce_lane(self._lane_speech_text(note.lane))
                miss_window = self.space_hit_window if note.lane == "SPACE" else self.hit_window
                if now - note.time_s > miss_window:
                    missed_cluster = self._simultaneous_cluster(note)
                    for missed in missed_cluster:
                        missed.judged = True
                        missed.hit = False
                    self.misses += 1
                    self.combo = 0
                    self._apply_miss_penalty(self.fail_miss_penalty)
                    self.cues.play_miss()
                    if self.failed:
                        break
            if now >= self.analysis.duration_s + 0.3:
                self.finished = True
        return self.finished

    def draw(self, font: pygame.font.Font, small_font: pygame.font.Font) -> None:
        w, h = self.screen.get_size()
        self.screen.fill((0, 0, 0))
        hit_line_y = int(h * 0.82)

        lane_x = []
        for i in range(len(self.lanes)):
            x = int((i + 1) * (w / (len(self.lanes) + 1)))
            lane_x.append(x)
            lane = self.lanes[i]
            color = self.lane_colors.get(lane, (200, 200, 200))
            dim = (max(80, color[0] // 3), max(80, color[1] // 3), max(80, color[2] // 3))
            pygame.draw.line(self.screen, dim, (x, 110), (x, h - 40), 2)
            if self._visual_cues_enabled() and self.cues.lane_labels_visible():
                lbl = small_font.render(lane, True, color)
                self.screen.blit(lbl, lbl.get_rect(center=(x, h - 18)))

        pygame.draw.line(self.screen, (255, 255, 255), (40, hit_line_y), (w - 40, hit_line_y), 2)
        pygame.draw.rect(self.screen, (80, 80, 80), pygame.Rect(40, hit_line_y - 18, w - 80, 36), 1)

        now = self._song_time()
        px_per_s = 260 * self._timeline_scale
        lane_index = {lane: i for i, lane in enumerate(self.lanes)}

        for note in self.analysis.notes:
            if note.judged and (now - note.time_s) > 0.6:
                continue
            dt = note.time_s - now
            y = int(hit_line_y - dt * px_per_s)
            if y < 70 or y > h - 20:
                continue
            x = lane_x[lane_index[note.lane]]
            color = self.lane_colors.get(note.lane, (255, 255, 255))
            if note.judged and note.hit:
                color = (120, 220, 120)
            if note.judged and not note.hit:
                color = (220, 90, 90)
            pygame.draw.circle(self.screen, color, (x, y), 12)

        mode_text = "Reduced Inputs" if self.reduced_inputs else "Full Inputs"
        title = font.render(f"Rithmatic Play ({mode_text})", True, (255, 255, 255))
        self.screen.blit(title, (20, 16))

        hud = small_font.render(
            (
                f"Score: {self.score}  Hits: {self.hits}  Misses: {self.misses}  "
                f"Combo: {self.combo}  Speed: {self._speed_text()}  "
                f"Song Vol: {int(round(self.song_volume * 100.0))}%  "
                f"Type: {self.analysis.genre}  H:Pause"
            ),
            True,
            (220, 220, 220),
        )
        self.screen.blit(hud, (20, 54))

        health_w = 280
        health_h = 16
        health_x = w - health_w - 26
        health_y = 20
        pygame.draw.rect(self.screen, (120, 120, 120), pygame.Rect(health_x, health_y, health_w, health_h), 1)
        fill_w = int((self.health / max(1.0, self.health_max)) * (health_w - 2))
        fill_color = (120, 220, 120) if self.health >= 45 else (255, 210, 120) if self.health >= 20 else (255, 120, 120)
        pygame.draw.rect(self.screen, fill_color, pygame.Rect(health_x + 1, health_y + 1, max(0, fill_w), health_h - 2))
        hp_text = small_font.render(f"Health {int(self.health)}", True, (235, 235, 235))
        self.screen.blit(hp_text, (health_x, health_y + 20))

        next_note = next((n for n in self.analysis.notes if not n.judged), None)
        if self._hud_guidance_enabled() and next_note is not None:
            lead = max(0.0, next_note.time_s - now)
            next_text = small_font.render(
                f"Next: {self._lane_speech_text(next_note.lane)} in {lead:.2f}s",
                True,
                (235, 235, 235),
            )
            self.screen.blit(next_text, (20, 88))

        if self._visual_cues_enabled() and self.last_judgment and now <= self.last_judgment_until:
            grade_color = {
                "PERFECT": (120, 240, 150),
                "GOOD": (190, 240, 120),
                "OK": (255, 210, 120),
                "MISS": (255, 120, 120),
            }.get(self.last_judgment, (230, 230, 230))
            grade = font.render(self.last_judgment, True, grade_color)
            self.screen.blit(grade, grade.get_rect(center=(w // 2, 86)))

        if self.paused:
            paused = font.render("PAUSED", True, (255, 255, 255))
            self.screen.blit(paused, paused.get_rect(center=(w // 2, h // 2)))
        if self.failed:
            failed = font.render("FAILED", True, (255, 120, 120))
            self.screen.blit(failed, failed.get_rect(center=(w // 2, h // 2 - 72)))

    def finish(self, announce_complete: bool = True) -> GameplayResult:
        self._reset_music_stream()
        if self._temp_playback_file is not None:
            try:
                self._temp_playback_file.unlink(missing_ok=True)
            except OSError:
                pass
            self._temp_playback_file = None
        if announce_complete and not self.complete_announced and not self.failed:
            self.announce("Song complete")
            self.complete_announced = True
        return GameplayResult(
            score=self.score,
            hits=self.hits,
            misses=self.misses,
            max_combo=self.max_combo,
            bpm=self.analysis.bpm,
            perfect_hits=self.perfect_hits,
            good_hits=self.good_hits,
            ok_hits=self.ok_hits,
            failed=self.failed,
        )
