from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pygame

from gameplay import DEFAULT_KEYBINDS, GameplayCueBank


@dataclass
class TutorialSection:
    section_id: str
    title: str
    paragraphs: list[str]


class TutorialSession:
    def __init__(
        self,
        screen: pygame.Surface,
        assets_root: str | Path,
        announce: Callable[[str], None],
        has_songs: Callable[[], bool],
        on_progress: Callable[[str, bool], None],
        announce_lane: Callable[[str], None] | None = None,
        cue_type: str = "both",
        cue_volume: float = 0.1,
        cue_panning: bool = False,
        speak_letters: bool = True,
        reduced_inputs: bool = False,
        difficulty_level: int = 5,
        keybinds: dict[str, str] | None = None,
    ):
        self.screen = screen
        self.assets_root = Path(assets_root)
        self.announce = announce
        self.announce_lane = announce_lane or announce
        self.has_songs = has_songs
        self.on_progress = on_progress

        self.cues = GameplayCueBank(
            self.assets_root,
            cue_type=cue_type,
            cue_volume=cue_volume,
            cue_panning=cue_panning,
        )
        self.cue_type = cue_type
        self.cue_panning = bool(cue_panning)
        self.speak_letters = bool(speak_letters)
        self.reduced_inputs = bool(reduced_inputs)
        self.difficulty_level = max(1, min(10, int(difficulty_level)))
        self.keybinds = self._normalize_keybinds(keybinds)

        self.finished = False
        self.redirect_to_songs = False
        self.phase = "sections"
        self.section_index = 0

        self.sections = self._build_sections()
        self.settings_rows = self._build_settings_rows()
        self.settings_row_index = 0

        self.item_selected_sound = self._load_ui_sound("item selected.mp3")
        self.song_selected_sound = self._load_ui_sound("song selected.mp3")

        self.on_progress("intro", False)
        self._announce_current_section(initial=True)

    def _normalize_keybinds(self, keybinds: dict[str, str] | None) -> dict[str, str]:
        merged = dict(DEFAULT_KEYBINDS)
        if not isinstance(keybinds, dict):
            return merged
        for action, key in keybinds.items():
            if action not in merged:
                continue
            key_name = str(key).strip().lower()
            if key_name:
                merged[action] = key_name
        return merged

    def _bind_key_code(self, action: str) -> int:
        name = str(self.keybinds.get(action, DEFAULT_KEYBINDS[action])).strip().lower()
        try:
            return pygame.key.key_code(name)
        except ValueError:
            return pygame.key.key_code(DEFAULT_KEYBINDS[action])

    def _display_key_name(self, key_name: str) -> str:
        key_name = str(key_name).strip().lower()
        if key_name == "space":
            return "Space"
        if key_name == "return":
            return "Enter"
        return key_name.upper() if key_name else "?"

    def _bind_label(self, action: str) -> str:
        return self._display_key_name(self.keybinds.get(action, DEFAULT_KEYBINDS[action]))

    def _build_sections(self) -> list[TutorialSection]:
        lane_text = (
            f"{self._bind_label('lane_s')} and {self._bind_label('lane_d')} for pitched lanes"
            if self.reduced_inputs
            else (
                f"{self._bind_label('lane_a')} {self._bind_label('lane_s')} "
                f"{self._bind_label('lane_d')} {self._bind_label('lane_f')} for pitched lanes"
            )
        )
        cue_label = {
            "both": "Both",
            "audio": "Audio",
            "visual": "Visual",
        }.get(self.cue_type, "Both")
        return [
            TutorialSection(
                section_id="what_is_this",
                title="What This Game Is",
                paragraphs=[
                    "This is a keyboard rhythm game for your own song files.",
                    "Notes line up to a hit line, and you press the matching lane on time.",
                    "You can play it without looking at the screen because cue sounds and spoken lanes are built in.",
                    "Press Enter or Space to move forward.",
                ],
            ),
            TutorialSection(
                section_id="controls",
                title="Controls In Plain English",
                paragraphs=[
                    f"Use {lane_text}, and {self._bind_label('lane_space')} for the drum lane.",
                    f"{self._bind_label('speed_down')} slows speed, {self._bind_label('speed_up')} speeds up, and {self._bind_label('pause')} pauses.",
                    "Arrow keys and Enter move through menus. Mouse also works in menus.",
                    "Press Enter or Space to continue.",
                ],
            ),
            TutorialSection(
                section_id="settings",
                title="Settings Walkthrough",
                paragraphs=[
                    "This part walks through the main toggle settings with live examples.",
                    "Up and Down change which setting you are focused on.",
                    "Left and Right change that setting value and replay the demo cues.",
                    "Enter or Space goes to the next section.",
                ],
            ),
            TutorialSection(
                section_id="cue_layers",
                title="Cue Layers And Lane Sounds",
                paragraphs=[
                    f"Current cue type is {cue_label}.",
                    "A is highest pitched lane, S next, D next, F lowest, and Space is drum lane.",
                    "You will hear each lane cue now, then hit and miss examples.",
                    "Press Enter or Space to keep going.",
                ],
            ),
            TutorialSection(
                section_id="song_handoff",
                title="Now Do A Real Song",
                paragraphs=[
                    "Next step is practical: pick a song from the Songs menu and play one run.",
                    "After the run, the game explains difficulty, BPM, density, stars, and rating in normal words.",
                    "Press Enter or Space to jump to Songs.",
                ],
            ),
        ]

    def _build_settings_rows(self) -> list[dict[str, object]]:
        return [
            {
                "id": "cue_type",
                "label": "Cue Type",
                "values": ["both", "audio", "visual"],
                "value": self.cue_type,
            },
            {
                "id": "cue_panning",
                "label": "Cue Panning",
                "values": [False, True],
                "value": self.cue_panning,
            },
            {
                "id": "speak_letters",
                "label": "Read Letters",
                "values": [True, False],
                "value": self.speak_letters,
            },
            {
                "id": "reduced_inputs",
                "label": "Reduced Inputs",
                "values": [False, True],
                "value": self.reduced_inputs,
            },
            {
                "id": "difficulty_level",
                "label": "Difficulty",
                "values": [2, 5, 8],
                "value": 5 if 3 <= self.difficulty_level <= 7 else (2 if self.difficulty_level < 3 else 8),
            },
        ]

    def _load_ui_sound(self, filename: str) -> pygame.mixer.Sound | None:
        path = self.assets_root / "sounds" / filename
        if not path.exists():
            return None
        try:
            return pygame.mixer.Sound(path.as_posix())
        except pygame.error:
            return None

    def _play_ui_sound(self, sound: pygame.mixer.Sound | None, volume: float = 0.65) -> None:
        if sound is None:
            return
        channel = sound.play()
        if channel is not None:
            v = max(0.0, min(1.0, float(volume)))
            channel.set_volume(v, v)

    def _current_section(self) -> TutorialSection:
        return self.sections[self.section_index]

    def _announce_current_section(self, initial: bool = False) -> None:
        section = self._current_section()
        if initial:
            self.announce("Tutorial started. Press Enter or Space to move through sections.")
        self.announce(section.title)
        for paragraph in section.paragraphs:
            self.announce(paragraph)
        if section.section_id == "settings":
            self.on_progress("settings_walkthrough", False)
            self._announce_settings_row(play_demo=True)
        elif section.section_id == "cue_layers":
            self.on_progress("cue_layers", False)
            self._play_cue_layer_demo()
        elif section.section_id == "song_handoff":
            self.on_progress("song_handoff", False)

    def _settings_value_text(self, row: dict[str, object]) -> str:
        value = row["value"]
        if isinstance(value, bool):
            return "ON" if value else "OFF"
        if row["id"] == "cue_type":
            return {"both": "Both", "audio": "Audio", "visual": "Visual"}.get(str(value), str(value))
        return str(value)

    def _announce_settings_row(self, play_demo: bool = False) -> None:
        row = self.settings_rows[self.settings_row_index]
        self.announce(f"{row['label']}: {self._settings_value_text(row)}")
        if play_demo:
            self._play_setting_demo(row)

    def _change_setting_value(self, delta: int) -> None:
        row = self.settings_rows[self.settings_row_index]
        values = row["values"]
        current = row["value"]
        try:
            idx = values.index(current)
        except ValueError:
            idx = 0
        idx = (idx + delta) % len(values)
        row["value"] = values[idx]

        if row["id"] == "cue_type":
            self.cue_type = str(row["value"])
            self.cues.set_mode(self.cue_type)
        elif row["id"] == "cue_panning":
            self.cue_panning = bool(row["value"])
            self.cues.set_cue_panning(self.cue_panning)
        elif row["id"] == "speak_letters":
            self.speak_letters = bool(row["value"])
        elif row["id"] == "reduced_inputs":
            self.reduced_inputs = bool(row["value"])
        elif row["id"] == "difficulty_level":
            self.difficulty_level = int(row["value"])

        self._play_ui_sound(self.item_selected_sound, volume=0.58)
        self._announce_settings_row(play_demo=True)

    def _play_setting_demo(self, row: dict[str, object]) -> None:
        row_id = str(row["id"])
        if row_id == "cue_type":
            self.announce(
                "Cue type controls layer output. Audio plays cue sounds, Visual shows lane letters, Both does both."
            )
            self.cues.play_lane("A", volume=0.68)
            self.cues.play_lane("SPACE", volume=0.72)
            return

        if row_id == "cue_panning":
            if bool(row["value"]):
                self.announce("Panning on. Left lanes sit left, right lanes sit right.")
            else:
                self.announce("Panning off. All lane cues are centered.")
            self.cues.play_lane("A", volume=0.72)
            self.cues.play_lane("F", volume=0.72)
            return

        if row_id == "speak_letters":
            if bool(row["value"]):
                self.announce("Read letters on. The game speaks lane names before notes.")
                self.announce_lane("A")
                self.announce_lane("Space")
            else:
                self.announce("Read letters off. You rely on cue sounds or visuals only.")
            return

        if row_id == "reduced_inputs":
            if bool(row["value"]):
                self.announce("Reduced Inputs on. A and S merge to S. D and F merge to D. Space stays drum lane.")
            else:
                self.announce("Reduced Inputs off. You keep full A S D F plus Space lanes.")
            self.cues.play_lane("S", volume=0.72)
            self.cues.play_lane("D", volume=0.72)
            self.cues.play_lane("SPACE", volume=0.72)
            return

        if row_id == "difficulty_level":
            level = int(row["value"])
            self.announce(
                f"Difficulty {level}. Higher levels usually increase speed pressure and note density."
            )

    def _play_cue_layer_demo(self) -> None:
        self.cues.set_mode("both")
        self.cues.play_lane("A", volume=0.62)
        self.announce_lane("A highest lane")
        self.cues.play_lane("S", volume=0.62)
        self.announce_lane("S lane")
        self.cues.play_lane("D", volume=0.62)
        self.announce_lane("D lane")
        self.cues.play_lane("F", volume=0.62)
        self.announce_lane("F lowest lane")
        self.cues.play_lane("SPACE", volume=0.68)
        self.announce_lane("Space drum lane")
        self.cues.play_hit("SPACE")
        self.announce("Hit confirmation cue played.")
        self.cues.play_miss()
        self.announce("Miss cue played.")

    def _advance_section(self) -> None:
        self.section_index += 1
        if self.section_index < len(self.sections):
            self._announce_current_section()
            return

        if self.has_songs():
            self.redirect_to_songs = True
            self.finished = True
            self.on_progress("completed", True)
            self._play_ui_sound(self.song_selected_sound, volume=0.72)
            self.announce("Tutorial section complete. Redirecting to Songs now.")
            self.announce("Choose a song and press Enter. Use speed controls during the run.")
            return

        self.phase = "add_song"
        self.on_progress("add_song", False)
        self.announce("No songs found yet.")
        self.announce("Add at least one audio file to songs folder, then press Enter or Space.")

    def _wrapped_lines(self, text: str, font: pygame.font.Font, max_width: int) -> list[str]:
        words = text.split()
        if not words:
            return [""]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type != pygame.KEYDOWN:
            return False

        if event.key == pygame.K_ESCAPE:
            self.finished = True
            self.redirect_to_songs = False
            self.on_progress(self.phase, False)
            self.announce("Tutorial canceled. Returning to menu.")
            return True

        if self.phase == "add_song":
            if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                if self.has_songs():
                    self.redirect_to_songs = True
                    self.finished = True
                    self.on_progress("completed", True)
                    self._play_ui_sound(self.song_selected_sound, volume=0.72)
                    self.announce("Song detected. Opening Songs menu.")
                else:
                    self.announce("Still no songs found. Add one file, then press Enter or Space again.")
            return True

        if self._current_section().section_id == "settings":
            if event.key == pygame.K_UP:
                self.settings_row_index = (self.settings_row_index - 1) % len(self.settings_rows)
                self._play_ui_sound(self.item_selected_sound, volume=0.52)
                self._announce_settings_row(play_demo=False)
                return True
            if event.key == pygame.K_DOWN:
                self.settings_row_index = (self.settings_row_index + 1) % len(self.settings_rows)
                self._play_ui_sound(self.item_selected_sound, volume=0.52)
                self._announce_settings_row(play_demo=False)
                return True
            if event.key == pygame.K_LEFT:
                self._change_setting_value(-1)
                return True
            if event.key == pygame.K_RIGHT:
                self._change_setting_value(1)
                return True

        if event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self._play_ui_sound(self.item_selected_sound, volume=0.56)
            self._advance_section()
            return True

        if event.key == self._bind_key_code("lane_a"):
            self.cues.play_lane("A", volume=0.74)
            self.announce_lane("A")
            return True
        if event.key == self._bind_key_code("lane_s"):
            self.cues.play_lane("S", volume=0.74)
            self.announce_lane("S")
            return True
        if event.key == self._bind_key_code("lane_d"):
            self.cues.play_lane("D", volume=0.74)
            self.announce_lane("D")
            return True
        if event.key == self._bind_key_code("lane_f"):
            self.cues.play_lane("F", volume=0.74)
            self.announce_lane("F")
            return True
        if event.key == self._bind_key_code("lane_space"):
            self.cues.play_lane("SPACE", volume=0.78)
            self.announce_lane("Space")
            return True

        return False

    def update(self) -> bool:
        return self.finished

    def draw(self, title_font: pygame.font.Font, body_font: pygame.font.Font) -> None:
        w, h = self.screen.get_size()
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(30, 24, w - 60, h - 48), 2)

        if self.phase == "add_song":
            page = TutorialSection(
                section_id="add_song",
                title="Add A Song File",
                paragraphs=[
                    "Put an audio file into songs folder.",
                    "Supported examples: MP3, WAV, OGG, FLAC, M4A.",
                    "Press Enter or Space to check again.",
                ],
            )
            self._draw_section(page, title_font, body_font)
            return

        section = self._current_section()
        self._draw_section(section, title_font, body_font)

        if section.section_id == "settings":
            self._draw_settings_preview(body_font)

    def _draw_section(self, section: TutorialSection, title_font: pygame.font.Font, body_font: pygame.font.Font) -> None:
        w, h = self.screen.get_size()
        title = title_font.render(section.title, True, (255, 255, 255))
        self.screen.blit(title, (50, 36))
        pygame.draw.line(self.screen, (220, 220, 220), (50, 96), (w - 50, 96), 2)

        y = 132
        max_width = w - 110
        for paragraph in section.paragraphs:
            for line in self._wrapped_lines(paragraph, body_font, max_width):
                surf = body_font.render(line, True, (235, 235, 235))
                self.screen.blit(surf, (50, y))
                y += 40
            y += 16

        footer = body_font.render("Enter/Space next, ESC exit", True, (210, 210, 210))
        self.screen.blit(footer, (50, h - 52))

    def _draw_settings_preview(self, body_font: pygame.font.Font) -> None:
        w, _ = self.screen.get_size()
        panel = pygame.Rect(50, 360, w - 100, 290)
        pygame.draw.rect(self.screen, (255, 255, 255), panel, 1)

        header = body_font.render("Demo Settings", True, (255, 255, 255))
        self.screen.blit(header, (panel.x + 18, panel.y + 14))

        for idx, row in enumerate(self.settings_rows):
            selected = idx == self.settings_row_index
            color = (255, 255, 255) if selected else (200, 200, 200)
            prefix = ">>" if selected else "  "
            text = f"{prefix} {row['label']}: {self._settings_value_text(row)}"
            surf = body_font.render(text, True, color)
            self.screen.blit(surf, (panel.x + 18, panel.y + 58 + idx * 42))

        hint = body_font.render("Up/Down choose setting, Left/Right change value", True, (220, 220, 220))
        self.screen.blit(hint, (panel.x + 18, panel.bottom - 42))
