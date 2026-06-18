from __future__ import annotations

from dataclasses import dataclass
from math import gcd, log2
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

try:
    import librosa
except Exception:  # pragma: no cover - optional dependency
    librosa = None

try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover - optional dependency
    imageio_ffmpeg = None


LANES = ["A", "S", "D", "F"]
TARGET_SR = 22050
ONSET_FRAME = 1024
ONSET_HOP = 256
MAX_TEMPO_ANALYSIS_SECONDS = 180.0
MAX_FEATURE_ANALYSIS_SECONDS = 45.0
MAX_NOTE_ANALYSIS_SECONDS = 300.0
MAX_AUTOCORR_FRAMES = 8192
MAX_LIBROSA_SECONDS = 20.0
INTRO_SCAN_MAX_SECONDS = 120.0
INTRO_WINDOW_SECONDS = 2.2
INTRO_HOP_SECONDS = 0.55


@dataclass
class NoteEvent:
    time_s: float
    lane: str
    is_drum: bool = False
    hit: bool = False
    judged: bool = False
    hit_cued: bool = False


@dataclass
class AnalysisResult:
    bpm: float
    duration_s: float
    notes: list[NoteEvent]
    genre: str
    lane_profile: str
    onset_density: float
    shanra_hits: int
    shanra_density: float
    start_offset_s: float = 0.0


@dataclass
class PitchedEvent:
    time_s: float
    midi: float
    centroid_hz: float
    transient: float
    harmonic_ratio: float
    power: float
    melodic_score: float = 0.0


def _hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * log2(max(hz, 1e-9) / 440.0)


def _normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-8:
        y = y / peak
    return y


def _ffmpeg_binary() -> str | None:
    env_path = os.environ.get("FFMPEG_BINARY")
    if env_path:
        return env_path
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    return None


def _convert_with_ffmpeg(src: Path, dst: Path, sample_rate: int = 44100, channels: int = 2) -> Path:
    ffmpeg_bin = _ffmpeg_binary()
    if ffmpeg_bin is None:
        raise RuntimeError(
            "ffmpeg is not available. Install it with winget (`winget install Gyan.FFmpeg`) "
            "or install the Python package `imageio-ffmpeg`."
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        src.as_posix(),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        dst.as_posix(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg conversion failed: {detail}")
    return dst


def _atempo_filters(rate: float) -> str:
    # ffmpeg atempo accepts 0.5..2.0, so chain filters for wider ratios.
    remaining = float(rate)
    parts: list[str] = []
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def _clean_cached_stem(path: Path) -> str:
    parts = path.stem.split(".")
    cleaned: list[str] = []
    for part in parts:
        if part == "auto":
            continue
        if part.startswith("rate") and part[4:].isdigit():
            continue
        cleaned.append(part)
    if not cleaned:
        return path.stem
    return ".".join(cleaned)


def _load_audio(song_path: Path) -> tuple[np.ndarray, int]:
    try:
        y, sr = sf.read(song_path.as_posix(), always_2d=False)
    except Exception:
        converted = convert_song_to_wav(song_path, song_path.parent)
        y, sr = sf.read(converted.as_posix(), always_2d=False)

    y = _normalize(y)
    if sr != TARGET_SR and y.size > 0:
        common = gcd(int(sr), TARGET_SR)
        up = TARGET_SR // common
        down = int(sr) // common
        y = resample_poly(y, up=up, down=down).astype(np.float32)
        sr = TARGET_SR
    return y, int(sr)


def convert_song_to_wav(song_path: str | Path, cache_dir: str | Path | None = None) -> Path:
    src = Path(song_path)
    if cache_dir is None:
        cache_root = src.parent
    else:
        cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    safe_suffix = src.suffix.lower().lstrip(".") or "bin"
    out_path = cache_root / f"{src.stem}.{safe_suffix}.auto.wav"
    def _valid_audio_file(path: Path) -> bool:
        try:
            info = sf.info(path.as_posix())
        except Exception:
            return False
        if getattr(info, "frames", 0) <= 0:
            return False
        if getattr(info, "samplerate", 0) <= 0:
            return False
        return True

    try:
        if out_path.exists() and out_path.stat().st_mtime >= src.stat().st_mtime:
            if _valid_audio_file(out_path):
                return out_path
            try:
                out_path.unlink(missing_ok=True)
            except OSError:
                pass
    except OSError:
        pass

    try:
        y, sr = sf.read(src.as_posix(), always_2d=False)
        sf.write(out_path.as_posix(), y, sr)
        if _valid_audio_file(out_path):
            return out_path
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
    except Exception:
        pass

    try:
        converted = _convert_with_ffmpeg(src, out_path, sample_rate=44100, channels=2)
        if _valid_audio_file(converted):
            return converted
        raise RuntimeError("converted WAV failed validation")
    except Exception as exc:
        raise RuntimeError(f"Could not decode and convert: {src.name}. {exc}") from exc


def convert_song_to_rate_wav(
    song_path: str | Path,
    rate: float,
    cache_dir: str | Path | None = None,
) -> Path:
    rate = float(np.clip(float(rate), 0.1, 3.0))
    src = Path(song_path)
    if src.suffix.lower() == ".wav" and src.exists():
        base_wav = src
    else:
        base_wav = convert_song_to_wav(src, cache_dir=cache_dir)
    if abs(rate - 1.0) < 1e-4:
        return base_wav

    rate_tag = f"rate{int(round(rate * 1000.0)):04d}"
    stem = _clean_cached_stem(base_wav)
    out_path = base_wav.with_name(f"{stem}.{rate_tag}.auto.wav")
    def _valid_audio_file(path: Path) -> bool:
        try:
            info = sf.info(path.as_posix())
        except Exception:
            return False
        if getattr(info, "frames", 0) <= 0:
            return False
        if getattr(info, "samplerate", 0) <= 0:
            return False
        return True

    try:
        if out_path.exists() and out_path.stat().st_mtime >= base_wav.stat().st_mtime:
            if _valid_audio_file(out_path):
                return out_path
            try:
                out_path.unlink(missing_ok=True)
            except OSError:
                pass
    except OSError:
        pass

    ffmpeg_bin = _ffmpeg_binary()
    if ffmpeg_bin is None:
        raise RuntimeError(
            "ffmpeg is required for slow/fast rate conversion. "
            "Install it with winget (`winget install Gyan.FFmpeg`) "
            "or install `imageio-ffmpeg`."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        base_wav.as_posix(),
        "-vn",
        "-filter:a",
        _atempo_filters(rate),
        "-ac",
        "2",
        "-ar",
        "44100",
        "-f",
        "wav",
        out_path.as_posix(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg rate conversion failed: {detail}")
    if not _valid_audio_file(out_path):
        raise RuntimeError("ffmpeg rate conversion produced invalid WAV output")
    return out_path


def trim_song_to_start(
    song_path: str | Path,
    start_offset_s: float,
    cache_dir: str | Path | None = None,
) -> Path:
    src = Path(song_path)
    offset = max(0.0, float(start_offset_s))
    if offset < 0.12:
        return src

    if src.suffix.lower() == ".wav" and src.exists():
        base_wav = src
    else:
        base_wav = convert_song_to_wav(src, cache_dir=cache_dir)

    start_tag = f"start{int(round(offset * 1000.0)):06d}"
    stem = _clean_cached_stem(base_wav)
    out_path = base_wav.with_name(f"{stem}.{start_tag}.auto.wav")

    try:
        if out_path.exists() and out_path.stat().st_mtime >= base_wav.stat().st_mtime:
            return out_path
    except OSError:
        pass

    y, sr = sf.read(base_wav.as_posix(), always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    start_sample = int(round(offset * max(1, int(sr))))
    start_sample = max(0, min(len(y), start_sample))
    trimmed = y[start_sample:]
    if len(trimmed) <= 8:
        return base_wav
    sf.write(out_path.as_posix(), trimmed, int(sr))
    return out_path


def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    if x.size == 0 or width <= 1:
        return x
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(x, kernel, mode="same")


def _slice_for_analysis(y: np.ndarray, sr: int, max_seconds: float) -> np.ndarray:
    if y.size == 0 or sr <= 0 or max_seconds <= 0:
        return y
    max_samples = int(max_seconds * sr)
    if max_samples <= 0 or y.size <= max_samples:
        return y
    start = (y.size - max_samples) // 2
    end = start + max_samples
    return y[start:end]


def _head_for_analysis(y: np.ndarray, sr: int, max_seconds: float) -> np.ndarray:
    if y.size == 0 or sr <= 0 or max_seconds <= 0:
        return y
    max_samples = int(max_seconds * sr)
    if max_samples <= 0 or y.size <= max_samples:
        return y
    return y[:max_samples]


def _rms_envelope(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if y.size < frame:
        return np.array([], dtype=np.float32)
    n = 1 + (len(y) - frame) // hop
    env = np.empty(n, dtype=np.float32)
    for i in range(n):
        start = i * hop
        seg = y[start : start + frame]
        env[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    return env


def _likely_music_video_name(name: str) -> bool:
    if not name:
        return False
    lowered = name.lower()
    hints = ("music video", "official video", "lyric video", "video", "visualizer")
    return any(token in lowered for token in hints)


def _normalized_rank(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float32)
    ranks[order] = np.arange(values.size, dtype=np.float32)
    return ranks / float(values.size - 1)


def _smooth_series(values: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size < 3:
        return values.copy()
    win = max(3, min(int(window), int(values.size)))
    if win % 2 == 0:
        win -= 1
    win = max(3, win)
    pad = win // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _window_music_score(seg: np.ndarray, sr: int) -> tuple[float, float]:
    if seg.size < 1024 or sr <= 0:
        return (0.0, 0.0)
    seg = np.asarray(seg, dtype=np.float32)
    rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    zcr = float(np.mean(np.signbit(seg[:-1]) != np.signbit(seg[1:]))) if seg.size >= 2 else 0.0

    env = _rms_envelope(seg, 1024, 256)
    periodicity = 0.0
    novelty_strength = 0.0
    if env.size >= 10:
        diff = np.maximum(np.diff(env, prepend=env[0]), 0.0)
        novelty_strength = float(np.mean(diff))
        novelty = diff - float(np.mean(diff))
        if novelty.size >= 10:
            n = int(novelty.size)
            fft_n = 1 << int(np.ceil(np.log2(max(2, (2 * n) - 1))))
            spec = np.fft.rfft(novelty, n=fft_n)
            acf = np.fft.irfft(spec * np.conj(spec), n=fft_n)[:n]
            frame_rate = sr / 256.0
            lag_min = max(1, int(frame_rate * 60.0 / 210.0))
            lag_max = min(n - 1, int(frame_rate * 60.0 / 68.0))
            if lag_max > lag_min:
                peak = float(np.max(acf[lag_min : lag_max + 1]))
                periodicity = float(np.clip(peak / (acf[0] + 1e-9), 0.0, 1.0))

    window = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * window))
    if spec.size <= 1:
        return (0.0, rms)
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
    total = float(np.sum(spec) + 1e-9)
    centroid_hz = float(np.sum(freqs * spec) / total)
    log_spec = np.log(spec + 1e-9)
    flatness = float(np.exp(np.mean(log_spec)) / (np.mean(spec) + 1e-9))
    harmonic_ratio = float(np.clip(1.0 - flatness, 0.0, 1.0))

    tonal_balance = float(np.clip((centroid_hz - 240.0) / 2800.0, 0.0, 1.0))
    energy_score = float(np.clip((rms - 0.006) / 0.065, 0.0, 1.0))
    novelty_score = float(np.clip(novelty_strength * 34.0, 0.0, 1.0))
    speech_penalty = float(np.clip(zcr * 7.5, 0.0, 1.0))
    score = (
        (0.38 * harmonic_ratio)
        + (0.34 * periodicity)
        + (0.14 * novelty_score)
        + (0.12 * tonal_balance)
        + (0.12 * energy_score)
        - (0.12 * speech_penalty)
    )
    return (float(np.clip(score, 0.0, 1.0)), rms)


def _detect_song_start_offset(y: np.ndarray, sr: int, source_name: str = "") -> float:
    if y.size < max(1024, sr // 2) or sr <= 0:
        return 0.0
    clip = _head_for_analysis(y, sr, INTRO_SCAN_MAX_SECONDS)
    if clip.size < max(1024, sr):
        return 0.0
    win = max(1024, int(INTRO_WINDOW_SECONDS * sr))
    hop = max(256, int(INTRO_HOP_SECONDS * sr))
    if clip.size < win + hop:
        return 0.0

    scores: list[float] = []
    energies: list[float] = []
    times: list[float] = []
    for start in range(0, max(0, clip.size - win), hop):
        seg = clip[start : start + win]
        score, rms = _window_music_score(seg, sr)
        times.append(float(start) / float(sr))
        scores.append(score)
        energies.append(rms)
    if len(scores) < 6:
        return 0.0

    score_arr = _moving_average(np.asarray(scores, dtype=np.float32), width=5)
    energy_arr = _moving_average(np.asarray(energies, dtype=np.float32), width=5)
    intro_windows = min(len(score_arr), max(4, int(8.0 / INTRO_HOP_SECONDS)))
    base_score = float(np.median(score_arr[:intro_windows]))
    base_energy = float(np.median(energy_arr[:intro_windows]))
    is_video_like = _likely_music_video_name(source_name)
    clip_duration_s = float(len(clip) / float(sr))
    if is_video_like:
        # Music videos can have long spoken intros before the actual track.
        max_video_skip_s = min(95.0, max(30.0, (len(y) / sr) * 0.5))
        max_skip_s = min(max(12.0, clip_duration_s - INTRO_WINDOW_SECONDS), max_video_skip_s)
    else:
        max_skip_s = min(45.0, max(8.0, (len(y) / sr) * 0.38))
    max_idx = min(len(score_arr) - 4, int(max_skip_s / INTRO_HOP_SECONDS))
    sustain = max(3, int(3.3 / INTRO_HOP_SECONDS))

    energy_rank = _normalized_rank(energy_arr)
    if is_video_like:
        jump_needed = 0.08
        activate_score = 0.46
        sustain_score = 0.43
        min_energy_rank = 0.60
    else:
        # Non-video tracks commonly have intentional musical intros; be conservative.
        jump_needed = 0.22
        activate_score = 0.62
        sustain_score = 0.59
        min_energy_rank = 0.82
        max_skip_s = min(max_skip_s, 16.0)

    for i in range(1, max(2, max_idx)):
        if times[i] < 0.9:
            continue
        tail = score_arr[i : i + sustain]
        if tail.size < sustain:
            break
        if score_arr[i] < activate_score:
            continue
        if float(np.mean(tail)) < sustain_score:
            continue
        strong_motion = float(np.mean(np.maximum(np.diff(score_arr[max(0, i - 2) : i + 1]), 0.0))) >= 0.01
        score_jump = (score_arr[i] - base_score) >= jump_needed
        energy_jump = (energy_arr[i] >= (base_energy * 2.2 + 0.014)) or (energy_rank[i] >= min_energy_rank)
        if not is_video_like and (base_score >= 0.44 or base_energy >= 0.028):
            continue
        if (score_jump and energy_jump) or (energy_jump and strong_motion and score_arr[i] >= 0.57):
            return float(times[i])

    # Silence/noise intros are still common, so keep a soft fallback gate.
    frame = 2048
    rms_hop = 512
    env = _rms_envelope(clip, frame, rms_hop)
    if env.size < 16:
        return 0.0
    base_frames = min(len(env), max(8, int(8.0 * sr / rms_hop)))
    noise_floor = float(np.percentile(env[:base_frames], 70))
    gate = max(0.012, noise_floor * 2.2)
    strong_gate = max(0.022, noise_floor * 3.1)
    sustain_frames = max(8, int(2.2 * sr / rms_hop))
    max_fallback_idx = min(len(env) - 1, int(max_skip_s * sr / rms_hop))
    fallback_score_gate = max(0.42, base_score + 0.08) if is_video_like else 0.0
    fallback_sustain_gate = max(0.40, base_score + 0.06) if is_video_like else 0.0
    fallback_score_sustain = max(2, int(2.2 / INTRO_HOP_SECONDS))
    for i in range(0, max_fallback_idx):
        if env[i] < gate:
            continue
        t_s = float((i * rms_hop) / sr)
        if is_video_like and t_s < 1.2:
            continue
        if is_video_like and score_arr.size:
            score_idx = min(score_arr.size - 1, max(0, int(round(t_s / INTRO_HOP_SECONDS))))
            score_now = float(score_arr[score_idx])
            score_tail = score_arr[score_idx : score_idx + fallback_score_sustain]
            if score_tail.size < max(1, fallback_score_sustain // 2):
                break
            if score_now < fallback_score_gate:
                continue
            if float(np.mean(score_tail)) < fallback_sustain_gate:
                continue
        window = env[i : i + sustain_frames]
        if window.size < sustain_frames // 2:
            break
        active_ratio = float(np.mean(window >= gate))
        strong_ratio = float(np.mean(window >= strong_gate))
        if active_ratio >= 0.62 and strong_ratio >= 0.20:
            return float((i * rms_hop) / sr)
    return 0.0


def _dedupe_times(times: np.ndarray, min_gap_s: float = 0.04) -> np.ndarray:
    if times.size == 0:
        return times
    deduped = [float(times[0])]
    for t in times[1:]:
        if float(t) - deduped[-1] >= min_gap_s:
            deduped.append(float(t))
    return np.asarray(deduped, dtype=float)


def _tempo_spacing(bpm: float, onset_density: float) -> tuple[float, float, float]:
    beat_s = 60.0 / max(1e-6, bpm)
    # Slow songs need wider spacing so charts match groove instead of machine-gun onsets.
    global_gap_s = float(np.clip(beat_s * 0.55, 0.13, 0.4))
    lane_gap_s = float(np.clip(beat_s * 0.78, 0.18, 0.58))
    space_gap_s = float(np.clip(beat_s * 0.9, 0.22, 0.72))

    # If the detector reports very sparse material, keep spacing wide and musical.
    if onset_density <= 1.2:
        global_gap_s = max(global_gap_s, 0.25)
        lane_gap_s = max(lane_gap_s, 0.34)
        space_gap_s = max(space_gap_s, 0.42)
    return global_gap_s, lane_gap_s, space_gap_s


def _rhythm_subdivision(bpm: float, onset_density: float) -> int:
    # Keep patterns readable: beat-only for slower/sparser tracks, half-beat for denser material.
    if bpm < 88.0 and onset_density < 2.2:
        return 1
    return 2


def _quantize_pitched_points(
    midi_points: list[PitchedEvent],
    beat_times: np.ndarray,
    bpm: float,
    onset_density: float,
) -> list[PitchedEvent]:
    if not midi_points:
        return []
    beat_s = 60.0 / max(1e-6, bpm)
    step = beat_s / float(_rhythm_subdivision(bpm, onset_density))
    anchor = float(beat_times[0]) if beat_times.size else 0.0

    by_slot: dict[int, tuple[float, float, float, PitchedEvent]] = {}
    for event in midi_points:
        slot = int(round((float(event.time_s) - anchor) / max(1e-6, step)))
        qt = anchor + slot * step
        if qt < 0.0:
            continue
        err = abs(float(event.time_s) - qt)
        snapped = PitchedEvent(
            time_s=float(qt),
            midi=float(event.midi),
            centroid_hz=float(event.centroid_hz),
            transient=float(event.transient),
            harmonic_ratio=float(event.harmonic_ratio),
            power=float(event.power),
            melodic_score=float(event.melodic_score),
        )
        prev = by_slot.get(slot)
        if prev is None:
            by_slot[slot] = (err, -snapped.melodic_score, -snapped.harmonic_ratio, snapped)
            continue
        better_timing = err < prev[0]
        similar_timing = abs(err - prev[0]) <= 0.014
        better_melody = snapped.melodic_score > -prev[1]
        if better_timing or (similar_timing and better_melody):
            by_slot[slot] = (err, -snapped.melodic_score, -snapped.harmonic_ratio, snapped)

    quantized = [entry[3] for entry in by_slot.values()]
    quantized.sort(key=lambda x: x.time_s)
    return quantized


def _space_lane_times(
    space_candidates: list[float],
    beat_times: np.ndarray,
    bpm: float,
    onset_density: float,
    percussive_drive: float,
    vocal_bias: float,
    harmonic_ratio_global: float,
) -> np.ndarray:
    if beat_times.size:
        beat_s = 60.0 / max(1e-6, bpm)
        tol = beat_s * 0.22
        snapped: set[float] = set()
        for t in space_candidates:
            idx = int(np.searchsorted(beat_times, t))
            nearest: list[float] = []
            if idx < beat_times.size:
                nearest.append(float(beat_times[idx]))
            if idx > 0:
                nearest.append(float(beat_times[idx - 1]))
            if not nearest:
                continue
            bt = min(nearest, key=lambda x: abs(x - float(t)))
            if abs(bt - float(t)) <= tol:
                snapped.add(bt)

        all_beats = [float(bt) for bt in beat_times.tolist()]
        if not all_beats:
            return np.array([], dtype=float)

        # Build a percussive backbone: dense when percussion is clear, sparse for vocal/melodic songs.
        sparse_penalty = 0.18 if vocal_bias >= 0.56 else 0.0
        drive = float(np.clip(percussive_drive - sparse_penalty, 0.0, 1.0))
        if onset_density >= 3.2 and bpm >= 120.0:
            drive = min(1.0, drive + 0.08)

        if drive < 0.24 and harmonic_ratio_global >= 0.62:
            # Piano/string dominant tracks should not force a drum pulse.
            backbone: list[float] = []
        elif drive >= 0.62:
            backbone = all_beats
        elif drive >= 0.42:
            backbone = all_beats[::2]
        elif drive >= 0.24:
            backbone = all_beats[::4]
        else:
            # Mostly melodic/vocal material: keep only strong downbeats.
            backbone = all_beats[::6]

        merged = sorted(set(backbone) | snapped)
        return np.asarray(merged, dtype=float)

    if not space_candidates:
        return np.array([], dtype=float)
    return np.asarray(sorted(space_candidates), dtype=float)


def _max_notes_per_beat(
    bpm: float,
    onset_density: float,
    percussive_drive: float,
    harmonic_ratio_global: float,
    vocal_bias: float,
) -> int:
    if percussive_drive <= 0.20 and (harmonic_ratio_global >= 0.64 or vocal_bias >= 0.56):
        return 2
    if percussive_drive <= 0.34:
        return 3
    if bpm < 90.0 or onset_density < 1.8:
        return 2
    if bpm < 130.0:
        return 3
    return 4


def _thin_pitched_events(
    events: list[PitchedEvent],
    bpm: float,
    onset_density: float,
    percussive_drive: float,
    harmonic_ratio_global: float,
    vocal_bias: float,
) -> list[PitchedEvent]:
    if len(events) <= 4:
        return events

    beat_s = 60.0 / max(1e-6, bpm)
    # Harmonic/melodic tracks should use wider note spacing than percussive tracks.
    if percussive_drive <= 0.20 and (harmonic_ratio_global >= 0.64 or vocal_bias >= 0.56):
        min_gap = float(np.clip(beat_s * 1.04, 0.32, 0.9))
    elif percussive_drive <= 0.34:
        min_gap = float(np.clip(beat_s * 0.68, 0.22, 0.58))
    else:
        min_gap = float(np.clip(beat_s * 0.52, 0.16, 0.44))

    if onset_density >= 4.0:
        min_gap *= 0.92

    kept: list[PitchedEvent] = []
    i = 0
    while i < len(events):
        start_t = events[i].time_s
        j = i + 1
        while j < len(events) and (events[j].time_s - start_t) < min_gap:
            j += 1
        group = events[i:j]
        chosen = max(
            group,
            key=lambda e: (e.melodic_score, e.harmonic_ratio, e.power, -abs(e.transient - 0.03)),
        )
        kept.append(chosen)
        i = j
    return kept


def _is_melodic_track(percussive_drive: float, harmonic_ratio_global: float, vocal_bias: float) -> bool:
    return (
        percussive_drive <= 0.22
        and harmonic_ratio_global >= 0.62
        and vocal_bias <= 0.52
    )


def _refine_melodic_events(events: list[PitchedEvent], bpm: float) -> list[PitchedEvent]:
    if len(events) < 3:
        return events
    beat_s = 60.0 / max(1e-6, bpm)
    local_span = float(np.clip(beat_s * 1.2, 0.36, 0.95))

    refined = list(events)
    for i in range(1, len(refined) - 1):
        prev_e = refined[i - 1]
        cur = refined[i]
        next_e = refined[i + 1]
        if (cur.time_s - prev_e.time_s) > local_span or (next_e.time_s - cur.time_s) > local_span:
            continue
        local_median = float(np.median([prev_e.midi, cur.midi, next_e.midi]))
        if abs(cur.midi - local_median) >= 7.5 and cur.melodic_score < 0.84:
            refined[i] = PitchedEvent(
                time_s=cur.time_s,
                midi=local_median,
                centroid_hz=cur.centroid_hz,
                transient=cur.transient,
                harmonic_ratio=cur.harmonic_ratio,
                power=cur.power,
                melodic_score=cur.melodic_score,
            )

    # Remove isolated noisy attacks that do not belong to a nearby melodic contour.
    kept: list[PitchedEvent] = []
    for i, cur in enumerate(refined):
        near_left = i > 0 and (cur.time_s - refined[i - 1].time_s) <= local_span and abs(cur.midi - refined[i - 1].midi) <= 5.5
        near_right = i < len(refined) - 1 and (refined[i + 1].time_s - cur.time_s) <= local_span and abs(refined[i + 1].midi - cur.midi) <= 5.5
        if near_left or near_right or cur.melodic_score >= 0.86:
            kept.append(cur)
    return kept


def _onset_times(y: np.ndarray, sr: int) -> np.ndarray:
    max_librosa_samples = int(MAX_LIBROSA_SECONDS * sr) if sr > 0 else 0
    can_use_librosa = max_librosa_samples <= 0 or y.size <= max_librosa_samples
    if librosa is not None and can_use_librosa and y.size >= ONSET_FRAME * 2:
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=ONSET_HOP)
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=ONSET_HOP,
                units="frames",
                backtrack=True,
                pre_max=3,
                post_max=3,
                pre_avg=3,
                post_avg=5,
                delta=0.07,
                wait=2,
            )
            times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=ONSET_HOP)
            return _dedupe_times(np.asarray(times, dtype=float))
        except Exception:
            pass

    env = _rms_envelope(y, ONSET_FRAME, ONSET_HOP)
    if env.size == 0:
        return np.array([], dtype=float)

    diff = np.diff(env, prepend=env[0])
    novelty = np.maximum(diff, 0.0)
    novelty = _moving_average(novelty, width=5)
    threshold = float(np.median(novelty) + 0.5 * np.std(novelty))
    min_gap_frames = max(1, int(0.06 * sr / ONSET_HOP))

    peaks: list[int] = []
    last_peak = -min_gap_frames
    for i in range(1, len(novelty) - 1):
        if i - last_peak < min_gap_frames:
            continue
        if novelty[i] < threshold:
            continue
        if novelty[i] >= novelty[i - 1] and novelty[i] >= novelty[i + 1]:
            peaks.append(i)
            last_peak = i

    return np.asarray([(p * ONSET_HOP) / sr for p in peaks], dtype=float)


def _estimate_bpm(y: np.ndarray, sr: int) -> float:
    max_librosa_samples = int(MAX_LIBROSA_SECONDS * sr) if sr > 0 else 0
    can_use_librosa = max_librosa_samples <= 0 or y.size <= max_librosa_samples
    if librosa is not None and can_use_librosa and y.size >= ONSET_FRAME * 4:
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=ONSET_HOP)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=ONSET_HOP)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if tempo.size else 120.0
            return float(np.clip(float(tempo), 60.0, 220.0))
        except Exception:
            pass

    env = _rms_envelope(y, ONSET_FRAME, ONSET_HOP)
    if env.size < 8:
        return 120.0
    diff = np.diff(env, prepend=env[0])
    novelty = np.maximum(diff, 0.0)
    novelty = _moving_average(novelty, width=5)
    novelty = novelty - float(np.mean(novelty))

    frame_rate = sr / ONSET_HOP
    if novelty.size > MAX_AUTOCORR_FRAMES:
        step = int(np.ceil(novelty.size / MAX_AUTOCORR_FRAMES))
        novelty = novelty[::step]
        frame_rate = frame_rate / step

    if novelty.size < 8:
        return 120.0

    n = int(novelty.size)
    fft_n = 1 << int(np.ceil(np.log2(max(2, (2 * n) - 1))))
    spec = np.fft.rfft(novelty, n=fft_n)
    acf = np.fft.irfft(spec * np.conj(spec), n=fft_n)[:n]
    lag_min = max(1, int(frame_rate * 60.0 / 220.0))
    lag_max = max(lag_min + 1, int(frame_rate * 60.0 / 65.0))
    lag_max = min(lag_max, len(acf) - 1)
    if lag_max <= lag_min:
        return 120.0
    segment = acf[lag_min : lag_max + 1]
    if segment.size == 0:
        return 120.0
    lag = lag_min + int(np.argmax(segment))
    if lag <= 0:
        return 120.0
    bpm = 60.0 * frame_rate / lag
    return float(np.clip(bpm, 65.0, 220.0))


def _beat_times(y: np.ndarray, sr: int, duration_s: float, bpm: float, onsets: np.ndarray) -> np.ndarray:
    if duration_s <= 0.0:
        return np.array([], dtype=float)

    max_librosa_samples = int(MAX_LIBROSA_SECONDS * sr) if sr > 0 else 0
    can_use_librosa = max_librosa_samples <= 0 or y.size <= max_librosa_samples
    if librosa is not None and can_use_librosa and y.size >= ONSET_FRAME * 4:
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=ONSET_HOP)
            _, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=ONSET_HOP,
                start_bpm=max(60.0, min(220.0, bpm)),
                units="frames",
            )
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=ONSET_HOP)
            if len(beat_times) > 0:
                return _dedupe_times(np.asarray(beat_times, dtype=float), min_gap_s=0.09)
        except Exception:
            pass

    interval = 60.0 / max(1e-6, bpm)
    if interval <= 0.0:
        return np.array([], dtype=float)

    if onsets.size:
        anchor = float(onsets[0])
        back_count = int(anchor // interval) + 1
        start = anchor - back_count * interval
    else:
        start = 0.0

    beats: list[float] = []
    t = start
    while t < duration_s:
        if t >= 0.0:
            beats.append(float(t))
        t += interval
    return np.asarray(beats, dtype=float)


def _onset_descriptor(
    y: np.ndarray, sr: int, t: float
) -> tuple[float | None, float, float, float, float, float, float]:
    center = int(t * sr)
    half = int(0.05 * sr)
    start = max(0, center - half)
    end = min(len(y), center + half)
    if end - start < 256:
        return (None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    seg = y[start:end]
    power = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    transient = float(np.mean(np.abs(np.diff(seg))))

    window = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
    total = float(np.sum(spec) + 1e-9)
    centroid_hz = float(np.sum(freqs * spec) / total)
    log_spec = np.log(spec + 1e-9)
    flatness = float(np.exp(np.mean(log_spec)) / (np.mean(spec) + 1e-9))
    harmonic_ratio = float(np.clip(1.0 - flatness, 0.0, 1.0))

    low_ratio = float(np.sum(spec[freqs <= 240.0]) / total)
    high_ratio = float(np.sum(spec[freqs >= 1700.0]) / total)

    band = (freqs >= 82.0) & (freqs <= 1400.0)
    if not np.any(band):
        return (None, low_ratio, high_ratio, transient, power, harmonic_ratio, centroid_hz)

    band_spec = spec[band]
    band_freqs = freqs[band]
    if band_spec.size == 0:
        return (None, low_ratio, high_ratio, transient, power, harmonic_ratio, centroid_hz)

    peak_idx = int(np.argmax(band_spec))
    peak_mag = float(band_spec[peak_idx])
    if peak_mag <= float(np.mean(band_spec)) * 1.32:
        return (None, low_ratio, high_ratio, transient, power, harmonic_ratio, centroid_hz)

    midi = _hz_to_midi(float(band_freqs[peak_idx]))
    if not np.isfinite(midi) or midi < 33.0 or midi > 96.0:
        return (None, low_ratio, high_ratio, transient, power, harmonic_ratio, centroid_hz)
    return (float(midi), low_ratio, high_ratio, transient, power, harmonic_ratio, centroid_hz)


def _lane_from_midi(midi: float, q25: float, q50: float, q75: float) -> str:
    if midi >= q75:
        return "A"
    if midi >= q50:
        return "S"
    if midi >= q25:
        return "D"
    return "F"


def _detect_genre(
    bpm: float,
    onset_density: float,
    harmonic_ratio: float,
    centroid_hz: float,
    low_energy_ratio: float,
    zcr: float,
) -> tuple[str, str]:
    if bpm >= 124 and onset_density >= 2.7 and harmonic_ratio < 0.7 and centroid_hz >= 2100:
        return ("EDM", "bright")
    if 72 <= bpm <= 108 and low_energy_ratio >= 0.46 and onset_density <= 3.3:
        return ("Hip-Hop", "low")
    if bpm >= 108 and onset_density >= 2.2 and harmonic_ratio >= 0.62 and zcr >= 0.075:
        return ("Rock", "mid")
    if bpm <= 88 and onset_density <= 1.6 and harmonic_ratio >= 0.72:
        return ("Ambient", "high")
    if harmonic_ratio >= 0.76 and onset_density <= 2.3 and centroid_hz <= 2300:
        return ("Acoustic", "mid")
    return ("Pop", "mid-high")


def _vocal_weight(y: np.ndarray, sr: int) -> float:
    if y.size < 1024 or sr <= 0:
        return 0.0
    seg = _slice_for_analysis(y, sr, 20.0)
    window = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
    total = float(np.sum(spec) + 1e-9)
    vocal_band = (freqs >= 220.0) & (freqs <= 3200.0)
    low_band = freqs <= 180.0
    vocal_ratio = float(np.sum(spec[vocal_band]) / total)
    low_ratio = float(np.sum(spec[low_band]) / total)
    return float(np.clip((vocal_ratio * 1.25) - (low_ratio * 0.7), 0.0, 1.0))


def _apply_profile_shift(midi: float, lane_profile: str) -> float:
    shifts = {
        "low": -3.0,
        "mid": 0.0,
        "mid-high": 1.5,
        "bright": 2.5,
        "high": 3.0,
    }
    return midi + shifts.get(lane_profile, 0.0)


def _audio_features(y: np.ndarray, sr: int) -> tuple[float, float, float, float]:
    if y.size < 16:
        return (1800.0, 0.65, 0.35, 0.06)

    y = _slice_for_analysis(y, sr, MAX_FEATURE_ANALYSIS_SECONDS)

    zcr = float(np.mean(np.signbit(y[:-1]) != np.signbit(y[1:])))
    window = np.hanning(len(y))
    spec = np.abs(np.fft.rfft(y * window))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    mag_sum = float(np.sum(spec) + 1e-9)
    centroid_hz = float(np.sum(freqs * spec) / mag_sum)

    low_ratio = float(np.sum(spec[freqs <= 250.0]) / mag_sum)

    log_spec = np.log(spec + 1e-9)
    flatness = float(np.exp(np.mean(log_spec)) / (np.mean(spec) + 1e-9))
    harmonic_ratio = float(np.clip(1.0 - flatness, 0.0, 1.0))
    return centroid_hz, low_ratio, harmonic_ratio, zcr


def _is_shanra_event(low_ratio: float, high_ratio: float, transient: float, power: float, midi: float | None) -> bool:
    if power < 0.015 or transient < 0.018:
        return False
    if high_ratio < 0.28:
        return False
    if low_ratio > 0.42:
        return False
    if midi is not None and midi > 81.0:
        return False
    return True


def _is_space_drum_event(
    low_ratio: float,
    high_ratio: float,
    transient: float,
    power: float,
    midi: float | None,
    near_beat: bool,
    harmonic_ratio: float,
    centroid_hz: float,
) -> bool:
    if power < 0.011 or transient < 0.01:
        return False
    tonal_event = midi is not None and harmonic_ratio >= 0.52
    if tonal_event:
        return False
    kick_like = (
        near_beat
        and low_ratio >= 0.5
        and transient >= 0.014
        and power >= 0.015
        and centroid_hz <= 950.0
        and (midi is None or harmonic_ratio <= 0.42)
    )
    snare_like = (
        near_beat
        and high_ratio >= 0.24
        and low_ratio <= 0.44
        and transient >= 0.021
        and harmonic_ratio <= 0.45
        and midi is None
    )
    percussive_no_pitch = (
        midi is None
        and near_beat
        and transient >= 0.026
        and power >= 0.015
        and harmonic_ratio <= 0.35
    )
    return kick_like or snare_like or percussive_no_pitch


def _estimate_percussive_drive(
    onset_desc: list[tuple[float, float | None, float, float, float, float, float, float]],
    vocal_bias: float,
    harmonic_ratio_global: float,
) -> float:
    if not onset_desc:
        return 0.0
    percussive_votes = 0.0
    total_weight = 0.0
    for _, midi, low_ratio, high_ratio, transient, power, harmonic_ratio_local, centroid_hz_local in onset_desc:
        weight = float(np.clip(power * 24.0, 0.35, 1.4))
        total_weight += weight
        if _is_space_drum_event(
            low_ratio,
            high_ratio,
            transient,
            power,
            midi,
            near_beat=True,
            harmonic_ratio=harmonic_ratio_local,
            centroid_hz=centroid_hz_local,
        ):
            percussive_votes += weight

    if total_weight <= 1e-6:
        return 0.0
    raw = percussive_votes / total_weight
    # Vocal-forward and highly harmonic songs should bias away from drum-lane saturation.
    raw -= float(np.clip((vocal_bias - 0.5) * 0.45, 0.0, 0.22))
    raw -= float(np.clip((harmonic_ratio_global - 0.62) * 0.35, 0.0, 0.14))
    return float(np.clip(raw, 0.0, 1.0))


def _stabilize_pitch_track(midi_points: list[PitchedEvent]) -> list[PitchedEvent]:
    if len(midi_points) < 3:
        return midi_points
    stabilized = list(midi_points)
    for i in range(1, len(stabilized) - 1):
        prev_midi = stabilized[i - 1].midi
        cur = stabilized[i]
        next_midi = stabilized[i + 1].midi
        cur_midi = cur.midi
        local_median = float(np.median([prev_midi, cur_midi, next_midi]))
        if abs(cur_midi - local_median) > 9.0:
            stabilized[i] = PitchedEvent(
                time_s=cur.time_s,
                midi=local_median,
                centroid_hz=cur.centroid_hz,
                transient=cur.transient,
                harmonic_ratio=cur.harmonic_ratio,
                power=cur.power,
                melodic_score=cur.melodic_score,
            )
    return stabilized


def _score_to_lane(score: float, q23: float, q50: float, q77: float) -> str:
    if score >= q77:
        return "A"
    if score >= q50:
        return "S"
    if score >= q23:
        return "D"
    return "F"


def _map_pitched_lanes(
    pitched_events: list[PitchedEvent],
    lane_profile: str,
    percussive_drive: float,
    harmonic_ratio_global: float,
    vocal_bias: float,
) -> list[tuple[float, str]]:
    if not pitched_events:
        return []

    shifted_midi = np.asarray(
        [_apply_profile_shift(event.midi, lane_profile) for event in pitched_events],
        dtype=np.float32,
    )
    centroid = np.asarray([event.centroid_hz for event in pitched_events], dtype=np.float32)
    transient = np.asarray([event.transient for event in pitched_events], dtype=np.float32)
    harmonic = np.asarray([event.harmonic_ratio for event in pitched_events], dtype=np.float32)

    midi_rank = _normalized_rank(shifted_midi)
    centroid_rank = _normalized_rank(centroid)
    transient_rank = _normalized_rank(transient)
    harmonic_rank = _normalized_rank(harmonic)
    midi_rank_smooth = _normalized_rank(_smooth_series(shifted_midi, window=5))
    centroid_rank_smooth = _normalized_rank(_smooth_series(centroid, window=3))
    midi_spread = float(np.percentile(shifted_midi, 90) - np.percentile(shifted_midi, 10))
    melodic_track = _is_melodic_track(percussive_drive, harmonic_ratio_global, vocal_bias)
    if melodic_track:
        midi_weight = 0.84 if midi_spread >= 4.5 else 0.76
        centroid_weight = 0.06
    else:
        midi_weight = 0.66 if midi_spread >= 6.0 else 0.52
        centroid_weight = 0.16 if midi_spread >= 6.0 else 0.26
    lane_score_raw = (
        (midi_weight * midi_rank)
        + (0.08 * midi_rank_smooth)
        + (centroid_weight * centroid_rank)
        + (0.06 * centroid_rank_smooth)
        + (0.14 * harmonic_rank)
        - (0.12 * transient_rank)
    )
    lane_score_smooth = _smooth_series(lane_score_raw, window=5)
    lane_score = (0.62 * lane_score_raw) + (0.38 * lane_score_smooth)
    q23, q50, q77 = [float(v) for v in np.percentile(lane_score, [23, 50, 77])]
    mq25, mq50, mq75 = [float(v) for v in np.percentile(shifted_midi, [25, 50, 75])]

    mapped: list[tuple[float, str, float]] = []
    lane_streak = 0
    prev_lane = ""
    prev_midi = float(shifted_midi[0])
    lane_order = {"A": 0, "S": 1, "D": 2, "F": 3}
    rev_lane = ["A", "S", "D", "F"]

    for i, event in enumerate(pitched_events):
        smidi = float(shifted_midi[i])
        lane = _score_to_lane(float(lane_score[i]), q23, q50, q77)
        midi_lane = _lane_from_midi(smidi, mq25, mq50, mq75)
        lane_idx = lane_order[lane]
        midi_idx = lane_order[midi_lane]

        # Melodic/vocal-forward material should favor pitch contour over transient/centroid spikes.
        if (melodic_track or vocal_bias >= 0.56) and abs(midi_idx - lane_idx) >= 2:
            lane = midi_lane
        elif harmonic[i] >= 0.72 and abs(smidi - prev_midi) >= 1.0 and abs(midi_idx - lane_idx) >= 1:
            lane = midi_lane

        if prev_lane == lane:
            lane_streak += 1
        else:
            lane_streak = 1

        # Avoid long same-lane runs if melodic contour is moving.
        if lane_streak >= 3 and abs(smidi - prev_midi) >= 1.2:
            cur_idx = lane_order[lane]
            if smidi > prev_midi:
                lane = rev_lane[max(0, cur_idx - 1)]
            else:
                lane = rev_lane[min(3, cur_idx + 1)]
            lane_streak = 1

        if prev_lane:
            prev_idx = lane_order[prev_lane]
            cur_idx = lane_order[lane]
            lane_jump = abs(cur_idx - prev_idx)
            # Large lane jumps should usually align with large pitch jumps.
            if lane_jump >= 3 and abs(smidi - prev_midi) < 4.0:
                target = prev_idx + (2 if cur_idx > prev_idx else -2)
                lane = rev_lane[max(0, min(3, target))]

        mapped.append((float(event.time_s), lane, float(lane_score[i])))
        prev_lane = lane
        prev_midi = smidi

    lane_counts: dict[str, int] = {lane: 0 for lane in LANES}
    for _, lane, _ in mapped:
        lane_counts[lane] += 1
    dominant = max(lane_counts, key=lane_counts.get)
    sparse = min(lane_counts, key=lane_counts.get)
    if lane_counts[dominant] >= max(8, int(len(mapped) * 0.58)) and lane_counts[sparse] <= max(2, int(len(mapped) * 0.1)):
        dom_idx = lane_order[dominant]
        sparse_idx = lane_order[sparse]
        direction = -1 if sparse_idx < dom_idx else 1
        reassigned: list[tuple[float, str, float]] = []
        changed = 0
        for t, lane, score in mapped:
            if lane != dominant or changed >= int(len(mapped) * 0.16):
                reassigned.append((t, lane, score))
                continue
            target_idx = max(0, min(3, dom_idx + direction))
            if target_idx == dom_idx:
                reassigned.append((t, lane, score))
                continue
            if abs(score - q50) <= 0.2:
                reassigned.append((t, rev_lane[target_idx], score))
                changed += 1
            else:
                reassigned.append((t, lane, score))
        mapped = reassigned

    return [(t, lane) for t, lane, _ in mapped]


def analyze_song(song_path: str | Path) -> AnalysisResult:
    song_path = Path(song_path)
    y, sr = _load_audio(song_path)
    start_offset_s = _detect_song_start_offset(y, sr, source_name=song_path.name)
    if start_offset_s > 0.12:
        start_sample = int(round(start_offset_s * sr))
        start_sample = max(0, min(len(y), start_sample))
        y = y[start_sample:]
    analysis_y = _head_for_analysis(y, sr, MAX_NOTE_ANALYSIS_SECONDS)
    duration_s = float(len(analysis_y) / sr) if sr > 0 else 0.0

    tempo_y = _slice_for_analysis(analysis_y, sr, MAX_TEMPO_ANALYSIS_SECONDS)
    try:
        bpm = _estimate_bpm(tempo_y, sr)
    except Exception:
        bpm = 120.0
    try:
        onset_times = _onset_times(analysis_y, sr)
    except Exception:
        onset_times = np.array([], dtype=float)
    onset_density = len(onset_times) / max(1.0, duration_s)
    try:
        beat_times = _beat_times(analysis_y, sr, duration_s, bpm, onset_times)
    except Exception:
        beat_times = np.array([], dtype=float)
    global_gap_s, lane_gap_s, space_gap_s = _tempo_spacing(bpm, onset_density)

    try:
        centroid_hz, low_energy_ratio, harmonic_ratio, zcr = _audio_features(analysis_y, sr)
    except Exception:
        centroid_hz, low_energy_ratio, harmonic_ratio, zcr = (1800.0, 0.35, 0.5, 0.06)
    genre, lane_profile = _detect_genre(
        bpm=bpm,
        onset_density=onset_density,
        harmonic_ratio=harmonic_ratio,
        centroid_hz=centroid_hz,
        low_energy_ratio=low_energy_ratio,
        zcr=zcr,
    )
    vocal_bias = _vocal_weight(analysis_y, sr)

    onset_desc: list[tuple[float, float | None, float, float, float, float, float, float]] = []
    for t in onset_times.tolist():
        midi, low_ratio, high_ratio, transient, power, harmonic_ratio_local, centroid_hz_local = _onset_descriptor(
            analysis_y, sr, float(t)
        )
        onset_desc.append(
            (
                float(t),
                midi,
                low_ratio,
                high_ratio,
                transient,
                power,
                harmonic_ratio_local,
                centroid_hz_local,
            )
        )

    shanra_hits = 0
    space_candidates: list[float] = []
    pitched_points: list[PitchedEvent] = []

    for t, midi, low_ratio, high_ratio, transient, power, harmonic_ratio_local, centroid_hz_local in onset_desc:
        near_beat = False
        if beat_times.size:
            beat_idx = int(np.searchsorted(beat_times, t))
            if beat_idx < beat_times.size and abs(float(beat_times[beat_idx]) - t) <= 0.08:
                near_beat = True
            if beat_idx > 0 and abs(float(beat_times[beat_idx - 1]) - t) <= 0.08:
                near_beat = True
        if _is_shanra_event(low_ratio, high_ratio, transient, power, midi):
            shanra_hits += 1

        melodic_like = (
            midi is not None
            and harmonic_ratio_local >= (0.31 if vocal_bias >= 0.52 else 0.39)
            and transient <= (0.058 if vocal_bias >= 0.52 else 0.047)
            and centroid_hz_local >= 105.0
        )
        melodic_score = float(
            np.clip(
                (harmonic_ratio_local * 0.54)
                + (np.clip(power * 18.0, 0.0, 1.0) * 0.26)
                + (np.clip((centroid_hz_local - 120.0) / 2400.0, 0.0, 1.0) * 0.12)
                + ((1.0 - np.clip(transient / 0.08, 0.0, 1.0)) * 0.16),
                0.0,
                1.0,
            )
        )
        if melodic_like:
            pitched_points.append(
                PitchedEvent(
                    time_s=float(t),
                    midi=float(midi),
                    centroid_hz=float(centroid_hz_local),
                    transient=float(transient),
                    harmonic_ratio=float(harmonic_ratio_local),
                    power=float(power),
                    melodic_score=melodic_score,
                )
            )
            continue

        if _is_space_drum_event(
            low_ratio,
            high_ratio,
            transient,
            power,
            midi,
            near_beat,
            harmonic_ratio_local,
            centroid_hz_local,
        ):
            space_candidates.append(t)
            continue
        if midi is not None:
            pitched_points.append(
                PitchedEvent(
                    time_s=float(t),
                    midi=float(midi),
                    centroid_hz=float(centroid_hz_local),
                    transient=float(transient),
                    harmonic_ratio=float(harmonic_ratio_local),
                    power=float(power),
                    melodic_score=melodic_score,
                )
            )

    percussive_drive = _estimate_percussive_drive(onset_desc, vocal_bias, harmonic_ratio)
    if _is_melodic_track(percussive_drive, harmonic_ratio, vocal_bias):
        pitched_points = [
            event
            for event in pitched_points
            if event.harmonic_ratio >= 0.44 and event.transient <= 0.054
        ]
        pitched_points = _refine_melodic_events(pitched_points, bpm)
    pitched_points = _stabilize_pitch_track(pitched_points)
    pitched_points = _quantize_pitched_points(pitched_points, beat_times, bpm, onset_density)
    pitched_points = _thin_pitched_events(
        pitched_points,
        bpm=bpm,
        onset_density=onset_density,
        percussive_drive=percussive_drive,
        harmonic_ratio_global=harmonic_ratio,
        vocal_bias=vocal_bias,
    )
    space_times = _space_lane_times(
        space_candidates,
        beat_times,
        bpm,
        onset_density,
        percussive_drive=percussive_drive,
        vocal_bias=vocal_bias,
        harmonic_ratio_global=harmonic_ratio,
    )

    notes: list[NoteEvent] = []
    space_gap = max(0.07, space_gap_s)
    for t in _dedupe_times(space_times, min_gap_s=space_gap).tolist():
        notes.append(NoteEvent(time_s=float(t), lane="SPACE", is_drum=True))

    for t, lane in _map_pitched_lanes(
        pitched_points,
        lane_profile,
        percussive_drive=percussive_drive,
        harmonic_ratio_global=harmonic_ratio,
        vocal_bias=vocal_bias,
    ):
        notes.append(NoteEvent(time_s=t, lane=lane, is_drum=False))

    notes.sort(key=lambda n: n.time_s)
    deduped: list[NoteEvent] = []
    last_by_lane: dict[str, float] = {}
    beat_s = 60.0 / max(1e-6, bpm)
    max_per_beat = _max_notes_per_beat(
        bpm,
        onset_density,
        percussive_drive=percussive_drive,
        harmonic_ratio_global=harmonic_ratio,
        vocal_bias=vocal_bias,
    )
    beat_bucket: list[float] = []
    for note in notes:
        while beat_bucket and (note.time_s - beat_bucket[0]) > beat_s:
            beat_bucket.pop(0)
        if len(beat_bucket) >= max_per_beat:
            continue
        prev_lane_t = last_by_lane.get(note.lane)
        if prev_lane_t is not None and (note.time_s - prev_lane_t) < lane_gap_s:
            continue
        if deduped:
            prev = deduped[-1]
            dt = note.time_s - prev.time_s
            if dt < global_gap_s and note.lane == prev.lane:
                continue
            # Keep a minimum inter-note spacing overall; allow close different-lane accents.
            if dt < (global_gap_s * 0.65):
                continue
        deduped.append(note)
        last_by_lane[note.lane] = note.time_s
        beat_bucket.append(note.time_s)

    shanra_density = float(shanra_hits / max(1.0, duration_s))

    return AnalysisResult(
        bpm=float(bpm),
        duration_s=duration_s,
        notes=deduped,
        genre=genre,
        lane_profile=lane_profile,
        onset_density=float(onset_density),
        shanra_hits=int(shanra_hits),
        shanra_density=shanra_density,
        start_offset_s=float(start_offset_s),
    )

