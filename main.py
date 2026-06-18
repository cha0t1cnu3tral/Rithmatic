from __future__ import annotations

import random
import sys
from pathlib import Path

import pygame

from audio_analysis import analyze_song, convert_song_to_wav, trim_song_to_start
from audio_runtime import AudioRuntime
from gameplay import DEFAULT_KEYBINDS, SPEED_STEPS, GameSession, GameplayCueBank
from progression import (
    ACHIEVEMENT_LABELS,
    MAX_CUE_VOLUME,
    PlayerProfile,
)
from song_identity import identify_song_identity
from screenreader import ScreenReader
from tutorial import TutorialSession
from ui import MenuState, draw_centered_menu


WINDOW_SIZE = (1100, 720)
TITLE = "Rithmatic"
PLAYABLE_SONG_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
PLAYABLE_SONG_EXTS |= {
    ".aac",
    ".aiff",
    ".aif",
    ".wma",
    ".opus",
    ".m4b",
    ".m4p",
    ".oga",
    ".amr",
    ".ac3",
}
CONVERT_ONLY_EXTS = {".webm", ".mp4", ".mkv", ".mov"}
SONG_EXTS = PLAYABLE_SONG_EXTS | CONVERT_ONLY_EXTS

CUE_TYPE_ORDER = ["both", "audio", "visual"]
CUE_TYPE_LABEL = {
    "both": "Both",
    "audio": "Audio",
    "visual": "Visual",
}
GAME_MODE_ORDER = ["chart", "rithm"]
GAME_MODE_LABEL = {
    "chart": "Chart",
    "rithm": "Rithm",
}
HELP_OPTIONS = ["Controls Overview", "Sounds Practice", "Start Tutorial", "Back"]
KEYBIND_ACTIONS = [
    "lane_a",
    "lane_s",
    "lane_d",
    "lane_f",
    "lane_space",
    "speed_down",
    "speed_up",
    "pause",
]
KEYBIND_LABELS = {
    "lane_a": "Lane A",
    "lane_s": "Lane S",
    "lane_d": "Lane D",
    "lane_f": "Lane F",
    "lane_space": "Lane Space",
    "speed_down": "Speed Down",
    "speed_up": "Speed Up",
    "pause": "Pause",
}
MIN_DIFFICULTY_LEVEL = 1
MAX_DIFFICULTY_LEVEL = 10


class App:
    def _resolve_assets_dir(self) -> Path:
        candidates = [
            self.app_dir / "assets",
            Path.cwd() / "assets",
            self.resource_dir / "assets",
        ]
        for path in candidates:
            if path.exists():
                return path
        return self.app_dir / "assets"

    def _resolve_songs_dir(self) -> Path:
        primary = self.app_dir / "songs"
        try:
            primary.mkdir(parents=True, exist_ok=True)
            return primary
        except OSError:
            fallback = Path.cwd() / "songs"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def _keybind_conflict_action(self, action: str, key_name: str) -> str | None:
        for other_action, other_key in self.settings.get("keybinds", {}).items():
            if other_action == action:
                continue
            if str(other_key).strip().lower() == key_name:
                return other_action
        return None

    def __init__(self):
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=1024)
        pygame.init()
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            except pygame.error:
                pygame.mixer.init()

        if getattr(sys, "frozen", False):
            self.resource_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
            self.app_dir = Path(sys.executable).resolve().parent
        else:
            self.resource_dir = Path(__file__).resolve().parent
            self.app_dir = self.resource_dir
        self.assets_dir = self._resolve_assets_dir()
        self.songs_dir = self._resolve_songs_dir()
        self.converted_songs_dir = self.songs_dir / ".converted"
        self._normalize_unsupported_song_files()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("consolas", 60, bold=True)
        self.font_item = pygame.font.SysFont("consolas", 42)
        self.font_small = pygame.font.SysFont("consolas", 28)

        self.sr = ScreenReader()
        self.audio = AudioRuntime(self.assets_dir, self.app_dir / ".cache")
        self.profile = PlayerProfile(self.app_dir / "player_profile.json")
        self.settings = self.profile.settings()
        self.audio.set_music_enabled(self.settings["music_enabled"])
        self.audio.set_menu_sounds_enabled(self.settings["menu_sounds_enabled"])
        self.audio.set_cue_volume(self.settings.get("cue_volume", 0.1))

        self.state = "MENU"
        self.main_menu = MenuState(
            ["Songs", "Help", "Credits", "Achievements", "Tutorial", "Settings", "Calibration", "Quit"]
        )
        self.song_options, self.song_lookup = self._song_entries()
        self.song_menu = MenuState(self.song_options)
        self.help_menu = MenuState(HELP_OPTIONS[:])
        self.settings_menu = MenuState(self._settings_options())
        self.keybind_menu = MenuState(self._keybind_options())
        self.waiting_for_keybind_action: str | None = None
        self.calibration_cue_due_ms = 0
        self.calibration_cue_started_ms: int | None = None
        self.calibration_status = "Press Enter to start calibration."
        self.session: GameSession | None = None
        self.tutorial: TutorialSession | None = None
        self.current_song_name = ""
        self.last_summary = "No songs played yet."
        self.tutorial_song_guidance = False
        self.main_rects: list[pygame.Rect] = []
        self.song_rects: list[pygame.Rect] = []
        self.help_rects: list[pygame.Rect] = []
        self.help_sound_cues = GameplayCueBank(
            self.assets_dir,
            cue_type="audio",
            cue_volume=self.settings.get("cue_volume", 0.1),
            cue_panning=self.settings.get("cue_panning", False),
        )
        self._init_controllers()

        self.audio.start_menu_music()
        self._speak("Main menu. Songs selected.")

    def _speak(self, text: str, interrupt: bool = True) -> None:
        self.sr.speak(text, interrupt=interrupt)

    def _speak_lane(self, text: str) -> None:
        self.sr.speak(text, interrupt=False)

    def _song_paths(self) -> list[Path]:
        songs_dir = self.songs_dir
        songs_dir.mkdir(parents=True, exist_ok=True)
        paths = [
            p
            for p in songs_dir.iterdir()
            if p.is_file()
            and not self._is_generated_audio_cache(p)
            and p.suffix.lower() in PLAYABLE_SONG_EXTS
        ]
        return sorted(paths, key=lambda x: x.name.lower())

    def _is_generated_audio_cache(self, path: Path) -> bool:
        name = path.name.lower()
        return (
            name.endswith(".auto.wav")
            or ".rate" in name and name.endswith(".wav") and ".auto" in name
        )

    def _cleanup_generated_song_audio(self) -> tuple[int, int]:
        songs_dir = self.songs_dir
        songs_dir.mkdir(parents=True, exist_ok=True)
        deleted_files = 0
        delete_errors = 0

        for p in songs_dir.rglob("*"):
            if not p.is_file():
                continue
            if not self._is_generated_audio_cache(p):
                continue
            try:
                p.unlink()
                deleted_files += 1
            except OSError:
                delete_errors += 1

        converted_dir = songs_dir / ".converted"
        if converted_dir.exists() and converted_dir.is_dir():
            try:
                if not any(converted_dir.iterdir()):
                    converted_dir.rmdir()
            except OSError:
                pass
        return deleted_files, delete_errors

    def _normalize_unsupported_song_files(self) -> tuple[int, int, int]:
        songs_dir = self.songs_dir
        songs_dir.mkdir(parents=True, exist_ok=True)
        converted_to_wav = 0
        removed_sources = 0
        errors = 0

        for src in songs_dir.iterdir():
            if not src.is_file() or self._is_generated_audio_cache(src):
                continue
            if src.suffix.lower() not in CONVERT_ONLY_EXTS:
                continue
            target = src.with_suffix(".wav")
            try:
                if not target.exists():
                    converted = convert_song_to_wav(src, songs_dir)
                    if converted.resolve() != target.resolve():
                        converted.replace(target)
                    converted_to_wav += 1
                src.unlink(missing_ok=True)
                removed_sources += 1
            except Exception:
                errors += 1

        converted_dir = songs_dir / ".converted"
        if converted_dir.exists() and converted_dir.is_dir():
            try:
                if not any(converted_dir.iterdir()):
                    converted_dir.rmdir()
            except OSError:
                pass
        return converted_to_wav, removed_sources, errors

    def _song_entries(self) -> tuple[list[str], list[str]]:
        paths = self._song_paths()
        if not paths:
            return (["(No songs found in songs folder)"], [""])
        options: list[str] = []
        lookup: list[str] = []
        for p in paths:
            stars = self.profile.song_stars(p.name)
            star_text = f"{stars}*" if stars > 0 else "unrated"
            options.append(f"{p.name}  [{star_text}]")
            lookup.append(p.name)
        return options, lookup

    def _init_controllers(self) -> None:
        pygame.joystick.init()
        self.joysticks: dict[int, pygame.joystick.Joystick] = {}
        for device_index in range(pygame.joystick.get_count()):
            joystick = pygame.joystick.Joystick(device_index)
            joystick.init()
            self.joysticks[joystick.get_instance_id()] = joystick

    def _controller_key_events(self, event: pygame.event.Event) -> list[pygame.event.Event]:
        mapped: list[pygame.event.Event] = []
        if event.type == pygame.JOYDEVICEADDED:
            joystick = pygame.joystick.Joystick(event.device_index)
            joystick.init()
            self.joysticks[joystick.get_instance_id()] = joystick
            self._speak("Controller connected.", interrupt=False)
            return mapped
        if event.type == pygame.JOYDEVICEREMOVED:
            self.joysticks.pop(event.instance_id, None)
            self._speak("Controller disconnected.", interrupt=False)
            return mapped
        if event.type == pygame.JOYHATMOTION:
            x, y = event.value
            if y > 0:
                mapped.append(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_UP}))
            elif y < 0:
                mapped.append(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN}))
            if x > 0:
                mapped.append(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT}))
            elif x < 0:
                mapped.append(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT}))
            return mapped
        if event.type != pygame.JOYBUTTONDOWN:
            return mapped

        button_to_key = {
            0: pygame.K_RETURN,
            1: pygame.K_ESCAPE,
            2: self._bind_key_code("lane_a"),
            3: self._bind_key_code("lane_s"),
            4: self._bind_key_code("lane_d"),
            5: self._bind_key_code("lane_f"),
            6: self._bind_key_code("speed_down"),
            7: self._bind_key_code("speed_up"),
            8: self._bind_key_code("lane_space"),
            9: self._bind_key_code("pause"),
        }
        key = button_to_key.get(event.button)
        if key is not None:
            mapped.append(pygame.event.Event(pygame.KEYDOWN, {"key": key}))
        return mapped

    def _refresh_song_list(self, announce: bool = False) -> None:
        self._normalize_unsupported_song_files()
        previous_name = ""
        if self.song_lookup and 0 <= self.song_menu.selected_index < len(self.song_lookup):
            previous_name = self.song_lookup[self.song_menu.selected_index]
        self.song_options, self.song_lookup = self._song_entries()
        self.song_menu = MenuState(self.song_options)
        if previous_name and previous_name in self.song_lookup:
            self.song_menu.selected_index = self.song_lookup.index(previous_name)
        if announce:
            if self.song_lookup and self.song_lookup[0]:
                self._speak(f"{len(self.song_lookup)} songs listed")
            else:
                self._speak("No songs listed")

    def _selected_song_speech(self) -> str:
        idx = self.song_menu.selected_index
        if 0 <= idx < len(self.song_lookup):
            selected = self.song_lookup[idx]
            if selected:
                stats = self.profile.song_stats(selected)
                bpm = float(stats.get("last_bpm", 0.0) or 0.0)
                onset_density = float(stats.get("last_onset_density", 0.0) or 0.0)
                if bpm > 0.0:
                    difficulty_text = self._difficulty_label(bpm, onset_density)
                else:
                    difficulty_text = "unknown"

                last_rating = stats.get("last_rating_score")
                last_stars = int(stats.get("last_stars", 0) or 0)
                if last_rating is None:
                    rating_text = "no last rating"
                else:
                    rating_text = f"last rating {float(last_rating):.1f}, last stars {last_stars}"
                identity = self.profile.song_identity(selected)
                primary = str(identity.get("primary_genre", "")).strip()
                if primary:
                    rating_text = f"{rating_text}, genre {primary}"

                return (
                    f"{selected}. Difficulty {difficulty_text}. {rating_text}"
                )
        return self.song_menu.selected

    def _option_with_position(self, text: str, index: int, total: int) -> str:
        return f"{text}. {index + 1} of {total}"

    def _open_static_page(self, page: str, speech: str) -> None:
        self.state = page
        self._speak(speech)

    def _achievements_summary_speech(self) -> str:
        unlocked = self.profile.achievements()
        songs_played = len(self.profile.data.get("song_stats", {}))
        total_achievements = len(ACHIEVEMENT_LABELS)
        status = lambda aid: "unlocked" if aid in unlocked else "locked"
        achievement_statuses = " ".join(
            f"{label} {status(achievement_id)}."
            for achievement_id, label in ACHIEVEMENT_LABELS.items()
        )
        return (
            f"Achievements page. {len(unlocked)} of {total_achievements} unlocked. "
            f"{achievement_statuses} "
            f"Player level {self.profile.level()}. "
            f"XP {self.profile.xp()} of {self.profile.xp_to_next_level()} to next level. "
            f"Songs with records {songs_played}. "
            f"Last result {self.last_summary}. "
            f"Press Escape to return, or Enter to hear this summary again."
        )

    def _open_achievements_page(self) -> None:
        self.state = "ACHIEVEMENTS"
        self._speak(self._achievements_summary_speech())

    def _difficulty_label(self, bpm: float, onset_density: float) -> str:
        difficulty_score = bpm * 0.45 + onset_density * 22.0
        if difficulty_score < 78:
            return "easy"
        if difficulty_score < 96:
            return "medium"
        if difficulty_score < 118:
            return "hard"
        return "expert"

    def _song_count(self) -> int:
        paths = self._song_paths()
        return len(paths)

    def _tutorial_progress(self, phase: str, completed: bool) -> None:
        self.profile.set_tutorial_progress(phase, completed)

    def _display_key_name(self, key_name: str) -> str:
        cleaned = key_name.strip().lower()
        if not cleaned:
            return "?"
        if cleaned == "space":
            return "SPACE"
        if len(cleaned) == 1:
            return cleaned.upper()
        return cleaned.upper()

    def _bind_key_code(self, action: str) -> int:
        key_name = str(self.settings.get("keybinds", {}).get(action, "")).strip().lower()
        if not key_name:
            key_name = DEFAULT_KEYBINDS.get(action, "")
        try:
            return pygame.key.key_code(key_name)
        except ValueError:
            return pygame.key.key_code(DEFAULT_KEYBINDS[action])

    def _keybind_options(self) -> list[str]:
        keybinds = self.settings.get("keybinds", {})
        options = []
        for action in KEYBIND_ACTIONS:
            key_name = self._display_key_name(str(keybinds.get(action, "")))
            options.append(f"{KEYBIND_LABELS[action]}: {key_name}")
        options.append("Back")
        return options

    def _settings_options(self) -> list[str]:
        cue_label = CUE_TYPE_LABEL[self.settings["cue_type"]]
        cue_volume = max(0.0, min(MAX_CUE_VOLUME, float(self.settings.get("cue_volume", 0.1))))
        cue_volume_label = f"{int(round(cue_volume * 100.0))}%"
        song_volume = max(0.0, min(1.0, float(self.settings.get("default_song_volume", 0.5))))
        song_volume_label = f"{int(round(song_volume * 100.0))}%"
        cue_panning_label = "ON" if self.settings.get("cue_panning", False) else "OFF"
        speak_letters_label = "YES" if self.settings["speak_letters"] else "NO"
        reduced_label = "ON" if self.settings["reduced_inputs"] else "OFF"
        game_mode = self.settings.get("game_mode", "chart")
        game_mode_label = GAME_MODE_LABEL.get(game_mode, GAME_MODE_LABEL["chart"])
        difficulty_label = str(
            max(MIN_DIFFICULTY_LEVEL, min(MAX_DIFFICULTY_LEVEL, int(self.settings.get("difficulty_level", 5))))
        )
        music_label = "ON" if self.settings["music_enabled"] else "OFF"
        menu_sfx_label = "ON" if self.settings["menu_sounds_enabled"] else "OFF"
        auto_skip_intro_label = "ON" if self.settings.get("auto_skip_intro", True) else "OFF"
        speed_steps = self._song_speed_steps()
        default_speed_index = self._default_song_speed_index()
        speed_label = f"{speed_steps[default_speed_index]:.2f}x"
        return [
            f"Cues: {cue_label}",
            f"Cue Volume: {cue_volume_label}",
            f"Default Song Volume: {song_volume_label}",
            f"Cue Panning: {cue_panning_label}",
            f"Read Letters: {speak_letters_label}",
            f"Reduced Inputs: {reduced_label}",
            f"Game Mode: {game_mode_label}",
            f"Difficulty: {difficulty_label}",
            f"Menu Music: {music_label}",
            f"Menu Sounds: {menu_sfx_label}",
            f"Skip Music Video Intro: {auto_skip_intro_label}",
            f"Default Start Speed: {speed_label}",
            "Configure Keybinds",
            "Back",
        ]

    def _settings_hint(self) -> str:
        idx = self.settings_menu.selected_index
        if idx == 0:
            return "Choose cue type: Audio, Visual, or Both."
        if idx == 1:
            return "Controls all cue loudness. Default is 10 percent. Left quieter, right louder up to 250 percent."
        if idx == 2:
            return "Sets the starting song loudness for every new song. Default is 50 percent."
        if idx == 3:
            return "Cue Panning routes A and S left, D and F right. Reduced inputs uses S left and D right."
        if idx == 4:
            return "Read Letters speaks each incoming lane. Default is yes."
        if idx == 5:
            return "Reduced Inputs merges A/S and D/F so you only play S, D, and Space."
        if idx == 6:
            return "Game Mode chooses normal chart lanes or Rithm mode, where any key plays the drum rhythm."
        if idx == 7:
            return "Difficulty from 1 to 10. Default is 5. Higher is faster and denser."
        if idx == 8:
            return "Menu music toggle controls only main-menu background tracks."
        if idx == 9:
            return "Menu sounds toggle controls selection and song-select sound effects."
        if idx == 10:
            return "Skips spoken or non-song lead-ins in music videos before gameplay starts."
        if idx == 11:
            return "Sets the starting speed during song countdown. Left slower, right faster."
        if idx == 12:
            return "Open per-action key mapping for lanes, pause, and speed controls."
        return "Press Enter or Escape to go back."

    def _apply_runtime_settings(self) -> None:
        self.audio.set_music_enabled(self.settings["music_enabled"])
        self.audio.set_menu_sounds_enabled(self.settings["menu_sounds_enabled"])
        self.audio.set_cue_volume(self.settings.get("cue_volume", 0.1))
        if hasattr(self, "help_sound_cues"):
            self.help_sound_cues.set_master_volume(self.settings.get("cue_volume", 0.1))
            self.help_sound_cues.set_cue_panning(self.settings.get("cue_panning", False))

    def _save_settings(self) -> None:
        self.profile.update_settings(
            cue_type=self.settings["cue_type"],
            cue_volume=self.settings["cue_volume"],
            default_song_volume=self.settings.get("default_song_volume", 0.5),
            cue_panning=self.settings.get("cue_panning", False),
            speak_letters=self.settings["speak_letters"],
            music_enabled=self.settings["music_enabled"],
            menu_sounds_enabled=self.settings["menu_sounds_enabled"],
            auto_skip_intro=self.settings.get("auto_skip_intro", True),
            game_mode=self.settings.get("game_mode", "chart"),
            reduced_inputs=self.settings["reduced_inputs"],
            difficulty_level=self.settings["difficulty_level"],
            input_latency_s=self.settings.get("input_latency_s", 0.0),
            default_start_speed_index=self.settings["default_start_speed_index"],
            keybinds=self.settings["keybinds"],
        )
        self._apply_runtime_settings()
        self.settings_menu.options = self._settings_options()
        self.keybind_menu.options = self._keybind_options()

    def _song_speed_steps(self) -> list[float]:
        return list(SPEED_STEPS)

    def _default_song_speed_index(self) -> int:
        speed_steps = self._song_speed_steps()
        fallback_index = speed_steps.index(1.0) if 1.0 in speed_steps else len(speed_steps) // 2
        return max(
            0,
            min(len(speed_steps) - 1, int(self.settings.get("default_start_speed_index", fallback_index))),
        )

    def _run_song_start_countdown(self, selected_name: str) -> int | None:
        speed_steps = self._song_speed_steps()
        speed_index = self._default_song_speed_index()
        countdown_ms = 5000
        start_tick = pygame.time.get_ticks()
        announced_remaining = -1

        self._speak(
            f"Ready check for {selected_name}. Five second countdown. "
            f"Press {self._display_key_name(self.settings['keybinds']['speed_down'])} slower, "
            f"{self._display_key_name(self.settings['keybinds']['speed_up'])} faster, "
            "Enter start now, Escape cancel."
        )

        while True:
            now_tick = pygame.time.get_ticks()
            elapsed_ms = max(0, now_tick - start_tick)
            remaining_ms = max(0, countdown_ms - elapsed_ms)
            remaining_s = (remaining_ms + 999) // 1000
            if remaining_s != announced_remaining and remaining_s > 0:
                announced_remaining = remaining_s
                self._speak(f"Starting in {remaining_s}")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_ESCAPE:
                    self._speak("Song start canceled.")
                    return None
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self._speak("Starting now.")
                    return speed_index
                if event.key == self._bind_key_code("speed_down"):
                    previous = speed_index
                    speed_index = max(0, speed_index - 1)
                    if speed_index != previous:
                        self.audio.play_item_selected()
                        self._speak(f"Start speed {speed_steps[speed_index]:.2f}x")
                    else:
                        self._speak(f"Lowest speed. {speed_steps[speed_index]:.2f}x")
                elif event.key == self._bind_key_code("speed_up"):
                    previous = speed_index
                    speed_index = min(len(speed_steps) - 1, speed_index + 1)
                    if speed_index != previous:
                        self.audio.play_item_selected()
                        self._speak(f"Start speed {speed_steps[speed_index]:.2f}x")
                    else:
                        self._speak(f"Highest speed. {speed_steps[speed_index]:.2f}x")

            if elapsed_ms >= countdown_ms:
                return speed_index

            self.screen.fill((0, 0, 0))
            title = self.font_item.render("Get Ready", True, (255, 255, 255))
            self.screen.blit(title, title.get_rect(center=(WINDOW_SIZE[0] // 2, 160)))

            current_speed = self.font_small.render(
                f"Start Speed: {speed_steps[speed_index]:.2f}x",
                True,
                (230, 230, 230),
            )
            self.screen.blit(current_speed, current_speed.get_rect(center=(WINDOW_SIZE[0] // 2, 270)))

            controls = self.font_small.render(
                (
                    f"{self._display_key_name(self.settings['keybinds']['speed_down'])} slower   "
                    f"{self._display_key_name(self.settings['keybinds']['speed_up'])} faster   "
                    "Enter start now   Esc cancel"
                ),
                True,
                (220, 220, 220),
            )
            self.screen.blit(controls, controls.get_rect(center=(WINDOW_SIZE[0] // 2, 340)))

            countdown_label = self.font_title.render(str(max(1, remaining_s)), True, (255, 255, 255))
            self.screen.blit(countdown_label, countdown_label.get_rect(center=(WINDOW_SIZE[0] // 2, 500)))

            pygame.display.flip()
            self.clock.tick(60)

    def _start_song(self) -> None:
        if not self.song_lookup or not self.song_lookup[self.song_menu.selected_index]:
            self._speak("No songs in songs folder")
            return
        selected_name = self.song_lookup[self.song_menu.selected_index]
        song_path = self.songs_dir / selected_name
        if not song_path.exists():
            self._speak("Song file missing")
            return
        analysis_path = song_path
        if song_path.suffix.lower() in CONVERT_ONLY_EXTS:
            self._speak("Unsupported extension detected. Converting automatically.")
            try:
                analysis_path = convert_song_to_wav(song_path, self.songs_dir)
            except Exception as exc:
                self._speak(f"Conversion failed: {exc}")
                return

        self.audio.play_song_selected()
        self.audio.stop_menu_music(fade_ms=500)

        self.screen.fill((0, 0, 0))
        loading = self.font_item.render("Analyzing song...", True, (255, 255, 255))
        self.screen.blit(loading, loading.get_rect(center=(WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)))
        pygame.display.flip()
        self._speak(f"Loading {selected_name}")

        try:
            analysis = analyze_song(analysis_path)
        except Exception as exc:
            try:
                self._speak("Trying automatic conversion.")
                analysis_path = convert_song_to_wav(song_path, self.converted_songs_dir)
                analysis = analyze_song(analysis_path)
            except Exception:
                self.state = "SONGS"
                self._speak(f"Analysis failed: {exc}")
                return

        if self.settings.get("auto_skip_intro", True) and analysis.start_offset_s >= 0.75:
            try:
                skipped_intro_s = analysis.start_offset_s
                trimmed_playback = trim_song_to_start(
                    analysis_path,
                    skipped_intro_s,
                    cache_dir=self.converted_songs_dir,
                )
                trimmed_analysis = analyze_song(trimmed_playback)
                analysis_path = trimmed_playback
                analysis = trimmed_analysis
                self._speak(
                    f"Detected non-song intro. Skipping first {skipped_intro_s:.1f} seconds."
                )
            except Exception:
                pass

        identity = identify_song_identity(
            song_name=selected_name,
            analysis_genre=analysis.genre,
            bpm=analysis.bpm,
            onset_density=analysis.onset_density,
            lane_profile=analysis.lane_profile,
        )
        self.profile.set_song_identity(
            selected_name,
            {
                "primary_genre": identity.primary_genre,
                "subgenres": identity.subgenres,
                "mood_tags": identity.mood_tags,
                "energy_tier": identity.energy_tier,
            },
        )

        self._speak(
            f"Detected song type {analysis.genre}. "
            f"Using {analysis.lane_profile} lane profile. "
            f"Identity {identity.primary_genre}. "
            f"Shanra events detected {analysis.shanra_hits}."
        )
        if self.tutorial_song_guidance:
            label = self._difficulty_label(analysis.bpm, analysis.onset_density)
            self._speak(
                f"Tutorial guidance: estimated difficulty {label}. "
                f"BPM {int(analysis.bpm)} and density {analysis.onset_density:.1f}. "
                "Use J for slower and L for faster. Start slow and step up."
            )
        start_speed_index = self._run_song_start_countdown(selected_name)
        if start_speed_index is None:
            self.state = "SONGS"
            return
        try:
            self.session = GameSession(
                self.screen,
                self.assets_dir,
                analysis_path,
                analysis,
                self._speak,
                announce_lane=self._speak_lane,
                cue_type=self.settings["cue_type"],
                cue_volume=self.settings["cue_volume"],
                cue_panning=self.settings.get("cue_panning", False),
                speak_letters=self.settings["speak_letters"],
                music_enabled=True,
                starting_song_volume=self.settings.get("default_song_volume", 0.5),
                game_mode=self.settings.get("game_mode", "chart"),
                reduced_inputs=self.settings["reduced_inputs"],
                difficulty_level=self.settings["difficulty_level"],
                input_latency_s=self.settings.get("input_latency_s", 0.0),
                initial_speed_index=start_speed_index,
                keybinds=self.settings["keybinds"],
            )
        except Exception as exc:
            self.session = None
            self.state = "SONGS"
            self._speak(f"Could not start playback: {exc}")
            return
        self.current_song_name = selected_name
        self.state = "GAME"

    def _start_tutorial(self) -> None:
        self.audio.stop_menu_music(fade_ms=350)
        self.tutorial = TutorialSession(
            self.screen,
            self.assets_dir,
            lambda text: self._speak(text, interrupt=False),
            has_songs=lambda: self._song_count() > 0,
            on_progress=self._tutorial_progress,
            announce_lane=self._speak_lane,
            cue_type=self.settings["cue_type"],
            cue_volume=self.settings["cue_volume"],
            cue_panning=self.settings.get("cue_panning", False),
            speak_letters=self.settings["speak_letters"],
            reduced_inputs=self.settings["reduced_inputs"],
            difficulty_level=self.settings.get("difficulty_level", 5),
            keybinds=self.settings.get("keybinds", {}),
        )
        self.state = "TUTORIAL"

    def _handle_menu_input(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and self.main_menu.move(-1):
                self.audio.play_item_selected()
                self._speak(
                    self._option_with_position(
                        self.main_menu.selected,
                        self.main_menu.selected_index,
                        len(self.main_menu.options),
                    )
                )
            elif event.key == pygame.K_DOWN and self.main_menu.move(1):
                self.audio.play_item_selected()
                self._speak(
                    self._option_with_position(
                        self.main_menu.selected,
                        self.main_menu.selected_index,
                        len(self.main_menu.options),
                    )
                )
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self.audio.play_item_selected()
                self._activate_main_option(self.main_menu.selected)
        elif event.type == pygame.MOUSEMOTION:
            for idx, rect in enumerate(self.main_rects):
                if rect.collidepoint(event.pos) and idx != self.main_menu.selected_index:
                    self.main_menu.selected_index = idx
                    self.audio.play_item_selected()
                    self._speak(
                        self._option_with_position(
                            self.main_menu.selected,
                            self.main_menu.selected_index,
                            len(self.main_menu.options),
                        )
                    )
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for idx, rect in enumerate(self.main_rects):
                if rect.collidepoint(event.pos):
                    self.main_menu.selected_index = idx
                    self.audio.play_item_selected()
                    self._activate_main_option(self.main_menu.selected)

    def _activate_main_option(self, option: str) -> None:
        if option == "Songs":
            self.audio.stop_menu_music(fade_ms=350)
            self.state = "SONGS"
            self._refresh_song_list(announce=True)
            self._speak("Songs screen")
            self._speak(
                self._option_with_position(
                    self._selected_song_speech(),
                    self.song_menu.selected_index,
                    len(self.song_menu.options),
                )
            )
            if self.tutorial_song_guidance:
                self._speak(
                    "Tutorial guidance active. Choose a song and press Enter. "
                    "Use J for slower and L for faster during play."
                )
            return
        if option == "Help":
            self.audio.stop_menu_music(fade_ms=350)
            self.state = "HELP"
            self.help_menu.selected_index = 0
            self._speak("Help")
            self._speak(
                self._option_with_position(
                    self.help_menu.selected,
                    self.help_menu.selected_index,
                    len(self.help_menu.options),
                )
            )
            return
        if option == "Credits":
            self.audio.stop_menu_music(fade_ms=350)
            self._open_static_page("CREDITS", "Credits page")
            return
        if option == "Achievements":
            self.audio.stop_menu_music(fade_ms=350)
            self._open_achievements_page()
            return
        if option == "Tutorial":
            self._speak("Tutorial screen")
            self._start_tutorial()
            return
        if option == "Settings":
            self.audio.stop_menu_music(fade_ms=350)
            self.state = "SETTINGS"
            self.settings_menu.options = self._settings_options()
            self._speak("Settings")
            self._speak(self.settings_menu.selected)
            self._speak(self._settings_hint())
            return
        if option == "Calibration":
            self._open_calibration()
            return
        if option == "Quit":
            pygame.quit()
            sys.exit(0)

    def _open_calibration(self) -> None:
        self.audio.stop_menu_music(fade_ms=350)
        self.state = "CALIBRATION"
        self.calibration_cue_due_ms = 0
        self.calibration_cue_started_ms = None
        self.calibration_status = "Press Enter to start calibration."
        self._speak(
            "Latency calibration. Press Enter to begin. "
            "After the drum sound, press Space as quickly as possible."
        )

    def _begin_calibration_trial(self) -> None:
        self.calibration_cue_started_ms = None
        self.calibration_cue_due_ms = pygame.time.get_ticks() + random.randint(650, 1100)
        self.calibration_status = "Get ready. Wait for the drum sound."
        self._speak("Get ready. Wait for the drum sound.")

    def _update_calibration(self) -> None:
        if self.calibration_cue_due_ms <= 0 or self.calibration_cue_started_ms is not None:
            return
        now_ms = pygame.time.get_ticks()
        if now_ms < self.calibration_cue_due_ms:
            return
        self.calibration_cue_due_ms = 0
        self.calibration_cue_started_ms = now_ms
        self.calibration_status = "Drum played. Press Space now."
        self.help_sound_cues.play_lane("SPACE", volume=0.95)

    def _handle_calibration_input(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.state = "MENU"
            self.audio.start_menu_music()
            self._speak("Main menu")
            return
        if event.key == pygame.K_RETURN:
            self._begin_calibration_trial()
            return
        if event.key != pygame.K_SPACE:
            return
        if self.calibration_cue_started_ms is None:
            self.calibration_cue_due_ms = 0
            self.calibration_status = "Too soon. Press Enter to try again."
            self._speak("Too soon. Press Enter to try again.")
            return
        latency_s = max(0.0, (pygame.time.get_ticks() - self.calibration_cue_started_ms) / 1000.0)
        calibrated = self.profile.record_latency_calibration(latency_s)
        self.settings = self.profile.settings()
        self.calibration_cue_started_ms = None
        latency_ms = int(round(calibrated * 1000.0))
        self.calibration_status = (
            f"Calibrated input latency: {latency_ms} milliseconds. "
            "Press Enter to recalibrate or Escape to return."
        )
        self._speak(self.calibration_status)

    def _handle_songs_input(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.state = "MENU"
                self.audio.start_menu_music()
                self._speak("Main menu")
            elif event.key == pygame.K_r:
                self._refresh_song_list(announce=True)
            elif event.key == pygame.K_c:
                converted, removed, normalize_errors = self._normalize_unsupported_song_files()
                deleted, errors = self._cleanup_generated_song_audio()
                self._refresh_song_list(announce=False)
                total_errors = normalize_errors + errors
                if converted == 0 and removed == 0 and deleted == 0 and total_errors == 0:
                    self._speak("No unsupported or generated files found.")
                elif total_errors == 0:
                    self._speak(
                        f"Cleanup complete. Converted {converted}, removed {removed} unsupported files, "
                        f"and removed {deleted} generated files."
                    )
                else:
                    self._speak(
                        f"Cleanup complete with errors. Converted {converted}, removed {removed} unsupported files, "
                        f"removed {deleted} generated files. {total_errors} failed."
                    )
            elif event.key == pygame.K_UP and self.song_menu.move(-1):
                self.audio.play_item_selected()
                self._speak(
                    self._option_with_position(
                        self._selected_song_speech(),
                        self.song_menu.selected_index,
                        len(self.song_menu.options),
                    )
                )
            elif event.key == pygame.K_DOWN and self.song_menu.move(1):
                self.audio.play_item_selected()
                self._speak(
                    self._option_with_position(
                        self._selected_song_speech(),
                        self.song_menu.selected_index,
                        len(self.song_menu.options),
                    )
                )
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._start_song()
        elif event.type == pygame.MOUSEMOTION:
            for idx, rect in enumerate(self.song_rects):
                if rect.collidepoint(event.pos) and idx != self.song_menu.selected_index:
                    self.song_menu.selected_index = idx
                    self.audio.play_item_selected()
                    self._speak(
                        self._option_with_position(
                            self._selected_song_speech(),
                            self.song_menu.selected_index,
                            len(self.song_menu.options),
                        )
                    )
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for idx, rect in enumerate(self.song_rects):
                if rect.collidepoint(event.pos):
                    self.song_menu.selected_index = idx
                    self._start_song()

    def _handle_game_input(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.session:
                self.session.finish(announce_complete=False)
            self.session = None
            self.state = "SONGS"
            self._speak("Returned to songs")
            return
        if self.session:
            self.session.handle_event(event)

    def _handle_achievements_input(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.state = "MENU"
            self.audio.start_menu_music()
            self._speak("Main menu")
            return
        if event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self._speak(self._achievements_summary_speech())

    def _handle_help_input(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.state = "MENU"
                self.audio.start_menu_music()
                self._speak("Main menu")
            elif event.key == pygame.K_UP and self.help_menu.move(-1):
                self.audio.play_item_selected()
                self._speak(
                    self._option_with_position(
                        self.help_menu.selected,
                        self.help_menu.selected_index,
                        len(self.help_menu.options),
                    )
                )
            elif event.key == pygame.K_DOWN and self.help_menu.move(1):
                self.audio.play_item_selected()
                self._speak(
                    self._option_with_position(
                        self.help_menu.selected,
                        self.help_menu.selected_index,
                        len(self.help_menu.options),
                    )
                )
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self.audio.play_item_selected()
                self._activate_help_option(self.help_menu.selected)
            return
        if event.type == pygame.MOUSEMOTION:
            for idx, rect in enumerate(self.help_rects):
                if rect.collidepoint(event.pos) and idx != self.help_menu.selected_index:
                    self.help_menu.selected_index = idx
                    self.audio.play_item_selected()
                    self._speak(
                        self._option_with_position(
                            self.help_menu.selected,
                            self.help_menu.selected_index,
                            len(self.help_menu.options),
                        )
                    )
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for idx, rect in enumerate(self.help_rects):
                if rect.collidepoint(event.pos):
                    self.help_menu.selected_index = idx
                    self.audio.play_item_selected()
                    self._activate_help_option(self.help_menu.selected)

    def _activate_help_option(self, option: str) -> None:
        if option == "Controls Overview":
            lane_keys = self.settings["keybinds"]
            lane_text = (
                f"{self._display_key_name(lane_keys['lane_a'])} "
                f"{self._display_key_name(lane_keys['lane_s'])} "
                f"{self._display_key_name(lane_keys['lane_d'])} "
                f"{self._display_key_name(lane_keys['lane_f'])}"
            )
            self._speak(
                f"Controls. {lane_text} for pitched lanes. "
                f"{self._display_key_name(lane_keys['lane_space'])} for drum lane. "
                f"{self._display_key_name(lane_keys['speed_down'])} slows speed. "
                f"{self._display_key_name(lane_keys['speed_up'])} increases speed. "
                f"{self._display_key_name(lane_keys['pause'])} pauses. "
                "Down arrow lowers song volume, up arrow raises song volume. "
                "Left bracket lowers cue volume, right bracket raises cue volume. "
                "Reduced Inputs merges to S, D, and Space. "
                "Rithm mode lets any key play to the drum rhythm. "
                "Songs screen supports refresh with R and cache cleanup with C."
            )
            return
        if option == "Sounds Practice":
            self.state = "HELP_SOUNDS"
            self._speak("Sounds practice. Press lane keys to hear cues. Escape to return.")
            return
        if option == "Start Tutorial":
            self._start_tutorial()
            return
        if option == "Back":
            self.state = "MENU"
            self.audio.start_menu_music()
            self._speak("Main menu")

    def _handle_help_sounds_input(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.state = "HELP"
            self._speak("Help")
            self._speak(
                self._option_with_position(
                    self.help_menu.selected,
                    self.help_menu.selected_index,
                    len(self.help_menu.options),
                )
            )
            return
        lane_map = {
            self._bind_key_code("lane_a"): "A",
            self._bind_key_code("lane_s"): "S",
            self._bind_key_code("lane_d"): "D",
            self._bind_key_code("lane_f"): "F",
            self._bind_key_code("lane_space"): "SPACE",
        }
        lane = lane_map.get(event.key)
        if lane is None:
            return
        if self.settings["reduced_inputs"] and lane in {"A", "F"}:
            lane = "S" if lane == "A" else "D"
        self.help_sound_cues.play_lane(lane, volume=0.95)
        self._speak_lane("Space" if lane == "SPACE" else lane)

    def _handle_settings_input(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.state = "MENU"
            self.audio.start_menu_music()
            self._speak("Main menu")
            return
        if event.key == pygame.K_UP and self.settings_menu.move(-1):
            self.audio.play_item_selected()
            self._speak(self.settings_menu.selected)
            self._speak(self._settings_hint())
            return
        if event.key == pygame.K_DOWN and self.settings_menu.move(1):
            self.audio.play_item_selected()
            self._speak(self.settings_menu.selected)
            self._speak(self._settings_hint())
            return
        if event.key not in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_LEFT, pygame.K_RIGHT):
            return

        idx = self.settings_menu.selected_index
        if idx == 0:
            current = self.settings["cue_type"]
            current_idx = CUE_TYPE_ORDER.index(current)
            if event.key == pygame.K_LEFT:
                next_idx = (current_idx - 1) % len(CUE_TYPE_ORDER)
            else:
                next_idx = (current_idx + 1) % len(CUE_TYPE_ORDER)
            self.settings["cue_type"] = CUE_TYPE_ORDER[next_idx]
            self._save_settings()
            self.audio.play_item_selected()
            self._speak(f"Cues {CUE_TYPE_LABEL[self.settings['cue_type']]}")
            self._speak(self.settings_menu.selected)
            return
        if idx == 1:
            previous = max(0.0, min(MAX_CUE_VOLUME, float(self.settings.get("cue_volume", 0.1))))
            step = 0.05
            if event.key == pygame.K_LEFT:
                updated = max(0.0, previous - step)
            elif event.key == pygame.K_RIGHT:
                updated = min(MAX_CUE_VOLUME, previous + step)
            else:
                updated = previous + step
                if updated > MAX_CUE_VOLUME:
                    updated = 0.0
            self.settings["cue_volume"] = round(updated, 3)
            self._save_settings()
            self.audio.play_item_selected()
            self._speak(f"Cue volume {int(round(updated * 100.0))} percent")
            self._speak(self.settings_menu.selected)
            return
        if idx == 2:
            previous = max(0.0, min(1.0, float(self.settings.get("default_song_volume", 0.5))))
            step = 0.05
            if event.key == pygame.K_LEFT:
                updated = max(0.0, previous - step)
            elif event.key == pygame.K_RIGHT:
                updated = min(1.0, previous + step)
            else:
                updated = previous + step
                if updated > 1.0:
                    updated = 0.0
            self.settings["default_song_volume"] = round(updated, 3)
            self._save_settings()
            self.audio.play_item_selected()
            self._speak(f"Default song volume {int(round(updated * 100.0))} percent")
            self._speak(self.settings_menu.selected)
            return
        if idx == 3:
            if event.key == pygame.K_LEFT:
                self.settings["cue_panning"] = False
            elif event.key == pygame.K_RIGHT:
                self.settings["cue_panning"] = True
            else:
                self.settings["cue_panning"] = not self.settings.get("cue_panning", False)
            self._save_settings()
            self.audio.play_item_selected()
            self._speak("Cue panning on" if self.settings.get("cue_panning", False) else "Cue panning off")
            self._speak(self.settings_menu.selected)
            return
        if idx == 4:
            if event.key == pygame.K_LEFT:
                self.settings["speak_letters"] = False
            elif event.key == pygame.K_RIGHT:
                self.settings["speak_letters"] = True
            else:
                self.settings["speak_letters"] = not self.settings["speak_letters"]
            self._save_settings()
            self.audio.play_item_selected()
            self._speak("Read letters on" if self.settings["speak_letters"] else "Read letters off")
            self._speak(self.settings_menu.selected)
            return
        if idx == 5:
            if event.key == pygame.K_LEFT:
                self.settings["reduced_inputs"] = False
            elif event.key == pygame.K_RIGHT:
                self.settings["reduced_inputs"] = True
            else:
                self.settings["reduced_inputs"] = not self.settings["reduced_inputs"]
            self._save_settings()
            self.audio.play_item_selected()
            self._speak(
                "Reduced inputs on" if self.settings["reduced_inputs"] else "Reduced inputs off"
            )
            self._speak(self.settings_menu.selected)
            return
        if idx == 6:
            current = self.settings.get("game_mode", "chart")
            if current not in GAME_MODE_ORDER:
                current = "chart"
            current_idx = GAME_MODE_ORDER.index(current)
            if event.key == pygame.K_LEFT:
                next_idx = (current_idx - 1) % len(GAME_MODE_ORDER)
            else:
                next_idx = (current_idx + 1) % len(GAME_MODE_ORDER)
            self.settings["game_mode"] = GAME_MODE_ORDER[next_idx]
            self._save_settings()
            self.audio.play_item_selected()
            self._speak(f"Game mode {GAME_MODE_LABEL[self.settings['game_mode']]}")
            self._speak(self.settings_menu.selected)
            return
        if idx == 7:
            previous = int(self.settings.get("difficulty_level", 5))
            if event.key == pygame.K_LEFT:
                updated = max(MIN_DIFFICULTY_LEVEL, previous - 1)
            elif event.key == pygame.K_RIGHT:
                updated = min(MAX_DIFFICULTY_LEVEL, previous + 1)
            else:
                updated = previous + 1
                if updated > MAX_DIFFICULTY_LEVEL:
                    updated = MIN_DIFFICULTY_LEVEL
            self.settings["difficulty_level"] = updated
            self._save_settings()
            self.audio.play_item_selected()
            self._speak(f"Difficulty {updated}")
            self._speak(self.settings_menu.selected)
            return
        if idx == 8:
            if event.key == pygame.K_LEFT:
                self.settings["music_enabled"] = False
            elif event.key == pygame.K_RIGHT:
                self.settings["music_enabled"] = True
            else:
                self.settings["music_enabled"] = not self.settings["music_enabled"]
            self._save_settings()
            self.audio.play_item_selected()
            self._speak("Menu music on" if self.settings["music_enabled"] else "Menu music off")
            self._speak(self.settings_menu.selected)
            return
        if idx == 9:
            if event.key == pygame.K_LEFT:
                self.settings["menu_sounds_enabled"] = False
            elif event.key == pygame.K_RIGHT:
                self.settings["menu_sounds_enabled"] = True
            else:
                self.settings["menu_sounds_enabled"] = not self.settings["menu_sounds_enabled"]
            self._save_settings()
            self.audio.play_item_selected()
            self._speak(
                "Menu sounds on" if self.settings["menu_sounds_enabled"] else "Menu sounds off"
            )
            self._speak(self.settings_menu.selected)
            return
        if idx == 10:
            if event.key == pygame.K_LEFT:
                self.settings["auto_skip_intro"] = False
            elif event.key == pygame.K_RIGHT:
                self.settings["auto_skip_intro"] = True
            else:
                self.settings["auto_skip_intro"] = not self.settings.get("auto_skip_intro", True)
            self._save_settings()
            self.audio.play_item_selected()
            self._speak(
                "Skip music video intro on"
                if self.settings.get("auto_skip_intro", True)
                else "Skip music video intro off"
            )
            self._speak(self.settings_menu.selected)
            return
        if idx == 11:
            speed_steps = self._song_speed_steps()
            previous = self._default_song_speed_index()
            if event.key == pygame.K_LEFT:
                updated = max(0, previous - 1)
            elif event.key == pygame.K_RIGHT:
                updated = min(len(speed_steps) - 1, previous + 1)
            else:
                updated = (previous + 1) % len(speed_steps)
            self.settings["default_start_speed_index"] = updated
            self._save_settings()
            self.audio.play_item_selected()
            self._speak(f"Default start speed {speed_steps[updated]:.2f}x")
            self._speak(self.settings_menu.selected)
            return
        if idx == 12:
            self.state = "KEYBINDS"
            self.waiting_for_keybind_action = None
            self.keybind_menu.options = self._keybind_options()
            self._speak("Keybind settings")
            self._speak(self.keybind_menu.selected)
            return
        if idx == 13:
            self.state = "MENU"
            self.audio.start_menu_music()
            self._speak("Main menu")

    def _handle_keybinds_input(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return

        if self.waiting_for_keybind_action:
            if event.key == pygame.K_ESCAPE:
                self.waiting_for_keybind_action = None
                self._speak("Key change canceled")
                self._speak(self.keybind_menu.selected)
                return
            action = self.waiting_for_keybind_action
            self.waiting_for_keybind_action = None
            key_name = pygame.key.name(event.key).strip().lower()
            if not key_name:
                self._speak("That key is not supported.")
                return
            conflict_action = self._keybind_conflict_action(action, key_name)
            if conflict_action:
                self._speak(
                    f"{self._display_key_name(key_name)} is already used by {KEYBIND_LABELS[conflict_action]}."
                )
                self._speak("Choose a different key.")
                return
            keybinds = dict(self.settings["keybinds"])
            keybinds[action] = key_name
            self.settings["keybinds"] = keybinds
            self._save_settings()
            self.audio.play_item_selected()
            self._speak(f"{KEYBIND_LABELS[action]} set to {self._display_key_name(key_name)}")
            self._speak(self.keybind_menu.selected)
            return

        if event.key == pygame.K_ESCAPE:
            self.state = "SETTINGS"
            self._speak("Settings")
            self._speak(self.settings_menu.selected)
            self._speak(self._settings_hint())
            return
        if event.key == pygame.K_UP and self.keybind_menu.move(-1):
            self.audio.play_item_selected()
            self._speak(self.keybind_menu.selected)
            return
        if event.key == pygame.K_DOWN and self.keybind_menu.move(1):
            self.audio.play_item_selected()
            self._speak(self.keybind_menu.selected)
            return
        if event.key not in (pygame.K_RETURN, pygame.K_SPACE):
            return
        idx = self.keybind_menu.selected_index
        if idx >= len(KEYBIND_ACTIONS):
            self.state = "SETTINGS"
            self._speak("Settings")
            self._speak(self.settings_menu.selected)
            self._speak(self._settings_hint())
            return
        action = KEYBIND_ACTIONS[idx]
        self.waiting_for_keybind_action = action
        self._speak(
            f"Press new key for {KEYBIND_LABELS[action]}. Escape to cancel."
        )

    def _handle_tutorial_input(self, event: pygame.event.Event) -> None:
        if not self.tutorial:
            return
        self.tutorial.handle_event(event)

    def _dispatch_state_event(self, event: pygame.event.Event) -> None:
        if self.state == "MENU":
            self._handle_menu_input(event)
        elif self.state == "SONGS":
            self._handle_songs_input(event)
        elif self.state == "GAME":
            self._handle_game_input(event)
        elif self.state == "HELP":
            self._handle_help_input(event)
        elif self.state == "SETTINGS":
            self._handle_settings_input(event)
        elif self.state == "KEYBINDS":
            self._handle_keybinds_input(event)
        elif self.state == "TUTORIAL":
            self._handle_tutorial_input(event)
        elif self.state == "HELP_SOUNDS":
            self._handle_help_sounds_input(event)
        elif self.state == "CALIBRATION":
            self._handle_calibration_input(event)
        elif self.state == "ACHIEVEMENTS":
            self._handle_achievements_input(event)
        elif self.state in {"CREDITS", "ACHIEVEMENTS"}:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.state = "MENU"
                self.audio.start_menu_music()
                self._speak("Main menu")

    def _draw_help(self) -> None:
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            pygame.Rect(50, 48, WINDOW_SIZE[0] - 100, WINDOW_SIZE[1] - 96),
            2,
        )
        title = self.font_item.render("HELP", True, (255, 255, 255))
        self.screen.blit(title, (90, 84))

        self.help_rects = [pygame.Rect(0, 0, 0, 0) for _ in self.help_menu.options]
        for idx, option in enumerate(self.help_menu.options):
            selected = idx == self.help_menu.selected_index
            color = (255, 255, 255) if selected else (205, 205, 205)
            prefix = ">> " if selected else "   "
            surf = self.font_small.render(f"{prefix}{option}", True, color)
            rect = surf.get_rect(topleft=(90, 166 + idx * 56))
            self.screen.blit(surf, rect)
            self.help_rects[idx] = rect

        keybinds = self.settings["keybinds"]
        detail_lines = [
            (
                "Keys: "
                f"{self._display_key_name(keybinds['lane_a'])} "
                f"{self._display_key_name(keybinds['lane_s'])} "
                f"{self._display_key_name(keybinds['lane_d'])} "
                f"{self._display_key_name(keybinds['lane_f'])} + "
                f"{self._display_key_name(keybinds['lane_space'])}, "
                f"{self._display_key_name(keybinds['speed_down'])}/"
                f"{self._display_key_name(keybinds['speed_up'])} speed, "
                f"{self._display_key_name(keybinds['pause'])} pause, up/down song volume, [ ] cue volume."
            ),
            "Reduced Inputs mode merges pitched lanes to S/D plus drum lane.",
            "Rithm mode uses any key against a drum-only rhythm chart.",
            "Misses lower health; if health hits zero, the song fails.",
            "Pick Sounds Practice for cue rehearsal.",
            "Controller support: D-pad navigate, A select, B back, shoulder buttons speed.",
            "ESC returns to Main Menu.",
        ]
        for i, line in enumerate(detail_lines):
            surf = self.font_small.render(line, True, (220, 220, 220))
            self.screen.blit(surf, (90, 450 + i * 46))

    def _draw_help_sounds(self) -> None:
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(50, 48, WINDOW_SIZE[0] - 100, WINDOW_SIZE[1] - 96), 2)
        keybinds = self.settings["keybinds"]
        if self.settings["reduced_inputs"]:
            lane_line = (
                f"{self._display_key_name(keybinds['lane_s'])}   "
                f"{self._display_key_name(keybinds['lane_d'])}   "
                f"{self._display_key_name(keybinds['lane_space'])}"
            )
        else:
            lane_line = (
                f"{self._display_key_name(keybinds['lane_a'])}   "
                f"{self._display_key_name(keybinds['lane_s'])}   "
                f"{self._display_key_name(keybinds['lane_d'])}   "
                f"{self._display_key_name(keybinds['lane_f'])}   "
                f"{self._display_key_name(keybinds['lane_space'])}"
            )
        lines = [
            "SOUNDS PRACTICE",
            "Press keys to hear their gameplay cue sounds:",
            lane_line,
            "Use this to memorize key-to-sound mapping",
            "ESC: Back to Help",
        ]
        for i, line in enumerate(lines):
            font = self.font_item if i == 0 else self.font_small
            surf = font.render(line, True, (255, 255, 255))
            self.screen.blit(surf, (90, 120 + i * 88))

    def _draw_credits(self) -> None:
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(50, 48, WINDOW_SIZE[0] - 100, WINDOW_SIZE[1] - 96), 2)
        lines = [
            "CREDITS",
            "Concept and direction: User",
            "Implementation: Codex",
            "Accessibility: NVDA bridge + accessible output fallback",
            "ESC to return",
        ]
        for i, line in enumerate(lines):
            font = self.font_item if i == 0 else self.font_small
            surf = font.render(line, True, (255, 255, 255))
            self.screen.blit(surf, (90, 84 + i * 70))

    def _draw_calibration(self) -> None:
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(50, 48, WINDOW_SIZE[0] - 100, WINDOW_SIZE[1] - 96), 2)
        latency_ms = int(round(float(self.settings.get("input_latency_s", 0.0)) * 1000.0))
        lines = [
            "LATENCY CALIBRATION",
            "Press Enter to start. Wait for the drum sound.",
            "Press Space as quickly as possible after the drum.",
            self.calibration_status,
            f"Stored input latency: {latency_ms} milliseconds",
            "ESC: Back to Main Menu",
        ]
        for i, line in enumerate(lines):
            font = self.font_item if i == 0 else self.font_small
            surf = font.render(line, True, (255, 255, 255))
            self.screen.blit(surf, (90, 108 + i * 74))

    def _draw_achievements(self) -> None:
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(50, 48, WINDOW_SIZE[0] - 100, WINDOW_SIZE[1] - 96), 2)
        unlocked = self.profile.achievements()
        songs_played = len(self.profile.data.get("song_stats", {}))
        status = lambda aid: "Unlocked" if aid in unlocked else "Locked"
        lines = [f"{label}: {status(aid)}" for aid, label in ACHIEVEMENT_LABELS.items()]
        header = self.font_item.render(
            f"ACHIEVEMENTS {len(unlocked)} / {len(ACHIEVEMENT_LABELS)}",
            True,
            (255, 255, 255),
        )
        self.screen.blit(header, (90, 84))
        step = 33
        for i, line in enumerate(lines):
            column = i // 15
            row = i % 15
            surf = self.font_small.render(line, True, (255, 255, 255))
            self.screen.blit(surf, (70 + column * 510, 145 + row * step))
        footer = self.font_small.render(
            f"Level {self.profile.level()}  XP {self.profile.xp()} / {self.profile.xp_to_next_level()}  Songs {songs_played}  ESC to return",
            True,
            (220, 220, 220),
        )
        self.screen.blit(footer, (70, WINDOW_SIZE[1] - 62))

    def _draw_settings(self) -> None:
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            pygame.Rect(50, 48, WINDOW_SIZE[0] - 100, WINDOW_SIZE[1] - 96),
            2,
        )
        title = self.font_item.render("SETTINGS", True, (255, 255, 255))
        self.screen.blit(title, (90, 84))

        row_top = 154
        row_step = 42
        visible_count = 9
        total = len(self.settings_menu.options)
        start = max(
            0,
            min(
                self.settings_menu.selected_index - (visible_count // 2),
                max(0, total - visible_count),
            ),
        )
        end = min(total, start + visible_count)

        for row, i in enumerate(range(start, end)):
            line = self.settings_menu.options[i]
            selected = i == self.settings_menu.selected_index
            color = (255, 255, 255) if selected else (200, 200, 200)
            prefix = ">> " if selected else "   "
            surf = self.font_small.render(f"{prefix}{line}", True, color)
            self.screen.blit(surf, (90, row_top + row * row_step))

        if start > 0:
            surf = self.font_small.render("^ more", True, (180, 180, 180))
            self.screen.blit(surf, (WINDOW_SIZE[0] - 210, row_top))
        if end < total:
            surf = self.font_small.render("v more", True, (180, 180, 180))
            self.screen.blit(surf, (WINDOW_SIZE[0] - 210, row_top + (visible_count - 1) * row_step))

        hint = self._settings_hint()
        hint_surface = self.font_small.render(hint, True, (220, 220, 220))
        self.screen.blit(hint_surface, (90, WINDOW_SIZE[1] - 120))
        footer = self.font_small.render(
            "Enter/Left/Right toggles, ESC goes back",
            True,
            (220, 220, 220),
        )
        self.screen.blit(footer, (90, WINDOW_SIZE[1] - 78))

    def _draw_keybinds(self) -> None:
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            pygame.Rect(50, 48, WINDOW_SIZE[0] - 100, WINDOW_SIZE[1] - 96),
            2,
        )
        title = self.font_item.render("KEYBINDS", True, (255, 255, 255))
        self.screen.blit(title, (90, 84))

        for i, line in enumerate(self.keybind_menu.options):
            selected = i == self.keybind_menu.selected_index
            color = (255, 255, 255) if selected else (200, 200, 200)
            prefix = ">> " if selected else "   "
            surf = self.font_small.render(f"{prefix}{line}", True, color)
            self.screen.blit(surf, (90, 160 + i * 52))

        if self.waiting_for_keybind_action:
            hint = (
                f"Listening for {KEYBIND_LABELS[self.waiting_for_keybind_action]}. "
                "Press any key or ESC to cancel."
            )
        else:
            hint = "Enter to rebind selected action. ESC returns to Settings."
        hint_surface = self.font_small.render(hint, True, (220, 220, 220))
        self.screen.blit(hint_surface, (90, WINDOW_SIZE[1] - 120))

    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                self._dispatch_state_event(event)
                for mapped_event in self._controller_key_events(event):
                    self._dispatch_state_event(mapped_event)

            self.audio.update()
            if self.state == "CALIBRATION":
                self._update_calibration()

            if self.state == "MENU":
                self.main_rects = draw_centered_menu(
                    self.screen,
                    "Rithmatic",
                    self.main_menu,
                    self.font_title,
                    self.font_item,
                )
            elif self.state == "SONGS":
                self.song_rects = draw_centered_menu(
                    self.screen,
                    "Songs (R refresh, C cleanup files, ESC back)",
                    self.song_menu,
                    self.font_title,
                    self.font_small,
                )
            elif self.state == "HELP":
                self._draw_help()
            elif self.state == "HELP_SOUNDS":
                self._draw_help_sounds()
            elif self.state == "SETTINGS":
                self._draw_settings()
            elif self.state == "KEYBINDS":
                self._draw_keybinds()
            elif self.state == "CALIBRATION":
                self._draw_calibration()
            elif self.state == "CREDITS":
                self._draw_credits()
            elif self.state == "ACHIEVEMENTS":
                self._draw_achievements()
            elif self.state == "GAME":
                if self.session is None:
                    self.state = "SONGS"
                else:
                    done = self.session.update()
                    self.session.draw(self.font_small, self.font_small)
                    if done:
                        analysis_onset_density = float(self.session.analysis.onset_density)
                        result = self.session.finish()
                        reward = self.profile.apply_song_result(
                            song_name=self.current_song_name or "unknown",
                            score=result.score,
                            hits=result.hits,
                            misses=result.misses,
                            bpm=result.bpm,
                            perfect_hits=result.perfect_hits,
                            good_hits=result.good_hits,
                            ok_hits=result.ok_hits,
                            max_combo=result.max_combo,
                            genre=self.session.analysis.genre,
                            onset_density=self.session.analysis.onset_density,
                            shanra_hits=self.session.analysis.shanra_hits,
                            failed=result.failed,
                        )
                        accuracy_pct = int(reward.accuracy * 100)
                        self.last_summary = (
                            f"Score {result.score}, Hits {result.hits}, Misses {result.misses}, "
                            f"Stars {reward.stars}, Failed {'yes' if result.failed else 'no'}, Rating {reward.rating_score:.1f}, "
                            f"XP +{reward.xp_gained}, Accuracy {accuracy_pct}%"
                        )
                        self.session = None
                        self.state = "SONGS"
                        self._refresh_song_list()
                        if result.failed:
                            self._speak(
                                f"Song failed. You earned {reward.stars} stars. "
                                f"Rating {reward.rating_score:.1f}. {reward.xp_gained} XP gained."
                            )
                        else:
                            self._speak(
                                f"Song complete. You earned {reward.stars} stars. "
                                f"Rating {reward.rating_score:.1f}. {reward.xp_gained} XP gained. "
                                f"Level {reward.level_after}."
                            )
                        if reward.unlocked_achievements:
                            self._speak(
                                f"New achievements unlocked: {', '.join(reward.unlocked_achievements)}."
                            )
                        if self.tutorial_song_guidance:
                            difficulty_label = self._difficulty_label(
                                result.bpm,
                                analysis_onset_density,
                            )
                            self._speak(
                                "Tutorial guidance summary. "
                                f"This song reads as {difficulty_label} difficulty. "
                                f"BPM means tempo speed. Yours was about {int(round(result.bpm))}. "
                                f"Onset density means how many note events happen per second. "
                                f"Yours was {analysis_onset_density:.1f}. "
                                "Higher stars come from cleaner timing, fewer misses, and stronger combos."
                            )
                            self.tutorial_song_guidance = False
            elif self.state == "TUTORIAL":
                if self.tutorial is None:
                    self.state = "MENU"
                    self.audio.start_menu_music()
                else:
                    done = self.tutorial.update()
                    self.tutorial.draw(self.font_item, self.font_small)
                    if done:
                        redirect_to_songs = self.tutorial.redirect_to_songs
                        self.tutorial = None
                        if redirect_to_songs:
                            self.state = "SONGS"
                            self.tutorial_song_guidance = True
                            self._refresh_song_list(announce=True)
                            self._speak(
                                "Tutorial complete. Songs menu. Select a song and press Enter to start your practice run."
                            )
                        else:
                            self.state = "MENU"
                            self.audio.start_menu_music()
                            self._speak("Main menu")

            pygame.display.flip()
            self.clock.tick(60)


def main() -> None:
    app = App()
    app.run()


if __name__ == "__main__":
    main()
