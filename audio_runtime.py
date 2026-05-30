from __future__ import annotations

from pathlib import Path

import pygame

from audio_analysis import convert_song_to_wav


class AudioRuntime:
    def __init__(self, assets_root: str | Path, cache_root: str | Path):
        self.assets_root = Path(assets_root)
        self.cache_root = Path(cache_root)
        self.menu_tracks = [
            self.assets_root / "music" / "mmm1.mp3",
            self.assets_root / "music" / "mmm2.mp3",
            self.assets_root / "music" / "mmm3.mp3",
            self.assets_root / "music" / "mmm4.mp3",
        ]
        self._track_index = 0
        self._menu_active = False
        self.menu_music_enabled = True
        self.menu_sounds_enabled = True
        self.cue_volume = 0.1
        self.max_cue_volume = 2.5
        self._menu_track_cache: dict[Path, Path] = {}

        self.item_selected = self._load_sound(
            self.assets_root / "sounds" / "item selected.mp3"
        )
        self.song_selected = self._load_sound(
            self.assets_root / "sounds" / "song selected.mp3"
        )
        self._apply_menu_sound_volume()

    def _load_sound(self, path: Path) -> pygame.mixer.Sound | None:
        try:
            if path.exists():
                return pygame.mixer.Sound(path.as_posix())
        except pygame.error:
            return None
        return None

    def _play_current_menu_track(self, fade_ms: int = 600) -> None:
        if not self.menu_tracks:
            return
        path = self.menu_tracks[self._track_index % len(self.menu_tracks)]
        if not path.exists():
            self._track_index += 1
            return
        playback_path = self._resolve_music_playback_path(path)
        if playback_path is None:
            self._track_index += 1
            return
        try:
            pygame.mixer.music.load(playback_path.as_posix())
            pygame.mixer.music.play(fade_ms=fade_ms)
        except pygame.error:
            self._track_index += 1

    def _resolve_music_playback_path(self, path: Path) -> Path | None:
        cached = self._menu_track_cache.get(path)
        if cached is not None and cached.exists():
            return cached
        try:
            pygame.mixer.music.load(path.as_posix())
            self._menu_track_cache[path] = path
            return path
        except pygame.error:
            pass
        try:
            converted = convert_song_to_wav(path, self.cache_root / "menu_music_converted")
        except Exception:
            return None
        try:
            pygame.mixer.music.load(converted.as_posix())
            self._menu_track_cache[path] = converted
            return converted
        except pygame.error:
            return None

    def start_menu_music(self) -> None:
        if self._menu_active:
            return
        self._menu_active = True
        if not self.menu_music_enabled:
            return
        self._play_current_menu_track(fade_ms=600)

    def stop_menu_music(self, fade_ms: int = 450) -> None:
        self._menu_active = False
        try:
            pygame.mixer.music.fadeout(fade_ms)
        except pygame.error:
            return

    def update(self) -> None:
        if not self._menu_active:
            return
        if not self.menu_music_enabled:
            return
        try:
            if pygame.mixer.music.get_busy():
                return
        except pygame.error:
            return
        self._track_index = (self._track_index + 1) % max(1, len(self.menu_tracks))
        self._play_current_menu_track(fade_ms=450)

    def play_item_selected(self) -> None:
        if not self.menu_sounds_enabled:
            return
        if self.item_selected is not None:
            self.item_selected.play()

    def play_song_selected(self) -> None:
        if not self.menu_sounds_enabled:
            return
        if self.song_selected is not None:
            self.song_selected.play()

    def set_music_enabled(self, enabled: bool) -> None:
        self.menu_music_enabled = bool(enabled)
        if not self.menu_music_enabled:
            if not self._menu_active:
                return
            try:
                pygame.mixer.music.fadeout(250)
            except pygame.error:
                return
        elif self._menu_active:
            try:
                if pygame.mixer.music.get_busy():
                    return
            except pygame.error:
                pass
            self._play_current_menu_track(fade_ms=250)

    def set_menu_sounds_enabled(self, enabled: bool) -> None:
        self.menu_sounds_enabled = bool(enabled)

    def _apply_menu_sound_volume(self) -> None:
        menu_volume = max(0.0, min(1.0, self.cue_volume))
        for sound in (self.item_selected, self.song_selected):
            if sound is not None:
                sound.set_volume(menu_volume)

    def set_cue_volume(self, volume: float) -> None:
        self.cue_volume = max(0.0, min(self.max_cue_volume, float(volume)))
        self._apply_menu_sound_volume()
