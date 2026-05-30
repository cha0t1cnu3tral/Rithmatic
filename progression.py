from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SETTINGS = {
    "cue_type": "both",
    "cue_volume": 0.1,
    "default_song_volume": 0.5,
    "cue_panning": False,
    "speak_letters": True,
    "music_enabled": True,
    "menu_sounds_enabled": True,
    "auto_skip_intro": True,
    "reduced_inputs": False,
    "difficulty_level": 5,
    "default_start_speed_index": 9,
    "keybinds": {
        "lane_a": "a",
        "lane_s": "s",
        "lane_d": "d",
        "lane_f": "f",
        "lane_space": "space",
        "speed_down": "j",
        "speed_up": "l",
        "pause": "h",
    },
}

VALID_CUE_TYPES = {"both", "audio", "visual"}
MIN_CUE_VOLUME = 0.0
MAX_CUE_VOLUME = 2.5
MIN_SONG_VOLUME = 0.0
MAX_SONG_VOLUME = 1.0
MIN_START_SPEED_INDEX = 0
MAX_START_SPEED_INDEX = 15
MIN_DIFFICULTY_LEVEL = 1
MAX_DIFFICULTY_LEVEL = 10
VALID_KEYBIND_ACTIONS = set(DEFAULT_SETTINGS["keybinds"].keys())


GENRE_KEYWORDS = {
    "rock": ["rock", "metal", "punk", "grunge"],
    "pop": ["pop", "dancepop", "synthpop"],
    "hiphop": ["hiphop", "hip-hop", "rap", "trap"],
    "electronic": ["edm", "electro", "house", "techno", "trance", "dubstep"],
    "jazz": ["jazz", "swing", "bebop", "fusion"],
    "classical": ["classical", "orchestra", "piano", "violin"],
    "lofi": ["lofi", "lo-fi", "chill"],
}

@dataclass
class ProgressionReward:
    stars: int
    xp_gained: int
    level_before: int
    level_after: int
    accuracy: float
    genre: str
    bpm_bucket: str
    rating_score: float
    unlocked_achievements: list[str]


class PlayerProfile:
    def __init__(self, profile_path: str | Path):
        self.path = Path(profile_path)
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return self._default_profile()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            merged = self._default_profile()
            merged.update(raw)
            if not isinstance(merged.get("song_stats"), dict):
                merged["song_stats"] = {}
            if not isinstance(merged.get("genres_seen"), list):
                merged["genres_seen"] = []
            if not isinstance(merged.get("bpm_buckets_seen"), list):
                merged["bpm_buckets_seen"] = []
            if not isinstance(merged.get("tutorial"), dict):
                merged["tutorial"] = self._default_tutorial_state()
            if not isinstance(merged.get("achievements"), dict):
                merged["achievements"] = {}
            if not isinstance(merged.get("song_identities"), dict):
                merged["song_identities"] = {}
            merged.pop("career", None)
            settings = merged.get("settings")
            if not isinstance(settings, dict):
                settings = {}
            normalized = dict(DEFAULT_SETTINGS)
            normalized.update(settings)
            legacy_cue_mode = settings.get("cue_mode")
            if normalized.get("cue_type") not in VALID_CUE_TYPES:
                if legacy_cue_mode == "audio_letters":
                    normalized["cue_type"] = "both"
                elif legacy_cue_mode == "audio_only":
                    normalized["cue_type"] = "audio"
                elif legacy_cue_mode == "letters_only":
                    normalized["cue_type"] = "visual"
            if normalized.get("cue_type") not in VALID_CUE_TYPES:
                normalized["cue_type"] = DEFAULT_SETTINGS["cue_type"]
            normalized["speak_letters"] = bool(normalized.get("speak_letters", True))
            normalized["cue_volume"] = self._normalize_cue_volume(
                normalized.get("cue_volume", DEFAULT_SETTINGS["cue_volume"])
            )
            normalized["default_song_volume"] = self._normalize_song_volume(
                normalized.get("default_song_volume", DEFAULT_SETTINGS["default_song_volume"])
            )
            normalized["cue_panning"] = bool(normalized.get("cue_panning", False))
            normalized["music_enabled"] = bool(normalized.get("music_enabled", True))
            normalized["menu_sounds_enabled"] = bool(normalized.get("menu_sounds_enabled", True))
            normalized["auto_skip_intro"] = bool(normalized.get("auto_skip_intro", True))
            normalized["reduced_inputs"] = bool(normalized.get("reduced_inputs", False))
            normalized["difficulty_level"] = max(
                MIN_DIFFICULTY_LEVEL,
                min(MAX_DIFFICULTY_LEVEL, int(normalized.get("difficulty_level", DEFAULT_SETTINGS["difficulty_level"]))),
            )
            normalized["default_start_speed_index"] = max(
                MIN_START_SPEED_INDEX,
                min(
                    MAX_START_SPEED_INDEX,
                    int(normalized.get("default_start_speed_index", DEFAULT_SETTINGS["default_start_speed_index"])),
                ),
            )
            normalized["keybinds"] = self._normalize_keybinds(normalized.get("keybinds"))
            merged["settings"] = normalized
            merged["lifetime_max_combo"] = int(merged.get("lifetime_max_combo", 0))
            merged["lifetime_shanra_hits"] = int(merged.get("lifetime_shanra_hits", 0))
            merged["lifetime_perfect_hits"] = int(merged.get("lifetime_perfect_hits", 0))
            return merged
        except (json.JSONDecodeError, OSError):
            return self._default_profile()

    def _default_profile(self) -> dict:
        return {
            "version": 1,
            "xp": 0,
            "level": 1,
            "song_stats": {},
            "genres_seen": [],
            "bpm_buckets_seen": [],
            "last_reward": "",
            "achievements": {},
            "lifetime_max_combo": 0,
            "lifetime_shanra_hits": 0,
            "lifetime_perfect_hits": 0,
            "song_identities": {},
            "tutorial": self._default_tutorial_state(),
            "settings": dict(DEFAULT_SETTINGS),
        }

    def _default_tutorial_state(self) -> dict:
        return {
            "status": "not_started",
            "phase": "none",
            "updated_at": "",
        }

    def _normalize_keybinds(self, raw_keybinds: dict | None) -> dict[str, str]:
        defaults = dict(DEFAULT_SETTINGS["keybinds"])
        if not isinstance(raw_keybinds, dict):
            return defaults
        normalized = dict(defaults)
        for action, value in raw_keybinds.items():
            if action not in VALID_KEYBIND_ACTIONS:
                continue
            if not isinstance(value, str):
                continue
            key_name = value.strip().lower()
            if key_name:
                normalized[action] = key_name
        return normalized

    def _normalize_cue_volume(self, value: object) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = float(DEFAULT_SETTINGS["cue_volume"])
        return max(MIN_CUE_VOLUME, min(MAX_CUE_VOLUME, parsed))

    def _normalize_song_volume(self, value: object) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = float(DEFAULT_SETTINGS["default_song_volume"])
        return max(MIN_SONG_VOLUME, min(MAX_SONG_VOLUME, parsed))

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def song_stars(self, song_name: str) -> int:
        stats = self.data["song_stats"].get(song_name.lower())
        if not stats:
            return 0
        return int(stats.get("best_stars", 0))

    def song_stats(self, song_name: str) -> dict:
        stats = self.data["song_stats"].get(song_name.lower())
        if not isinstance(stats, dict):
            return {}
        return stats

    def set_song_identity(self, song_name: str, identity: dict) -> None:
        identities = self.data.get("song_identities")
        if not isinstance(identities, dict):
            identities = {}
            self.data["song_identities"] = identities
        safe = {
            "primary_genre": str(identity.get("primary_genre", "unknown")),
            "subgenres": [str(x) for x in identity.get("subgenres", [])][:6],
            "mood_tags": [str(x) for x in identity.get("mood_tags", [])][:6],
            "energy_tier": str(identity.get("energy_tier", "mid")),
        }
        identities[song_name.lower()] = safe
        self.save()

    def song_identity(self, song_name: str) -> dict:
        identities = self.data.get("song_identities", {})
        if not isinstance(identities, dict):
            return {}
        identity = identities.get(song_name.lower(), {})
        if not isinstance(identity, dict):
            return {}
        return identity

    def level(self) -> int:
        return int(self.data.get("level", 1))

    def xp(self) -> int:
        return int(self.data.get("xp", 0))

    def xp_to_next_level(self) -> int:
        level = self.level()
        return self._xp_needed_for_level(level + 1)

    def apply_song_result(
        self,
        song_name: str,
        score: int,
        hits: int,
        misses: int,
        bpm: float,
        perfect_hits: int = 0,
        good_hits: int = 0,
        ok_hits: int = 0,
        max_combo: int = 0,
        genre: str | None = None,
        onset_density: float | None = None,
        shanra_hits: int = 0,
        failed: bool = False,
    ) -> ProgressionReward:
        total = max(1, hits + misses)
        accuracy = hits / total
        if genre is None:
            genre = self._infer_genre(song_name)
        bpm_bucket = self._bpm_bucket(bpm)
        rating_score = self._performance_rating(
            accuracy=accuracy,
            misses=misses,
            perfect_hits=perfect_hits,
            good_hits=good_hits,
            ok_hits=ok_hits,
            max_combo=max_combo,
            total_notes=total,
            bpm=bpm,
            onset_density=onset_density or 0.0,
        )
        stars = self._stars_from_rating(rating_score)
        if failed:
            stars = 0

        xp_gained = int(30 + stars * 24 + (accuracy * 50))
        if failed:
            # Failed runs should not grant XP, including discovery bonuses.
            xp_gained = 0
        else:
            if genre not in self.data["genres_seen"]:
                self.data["genres_seen"].append(genre)
                xp_gained += 35
            if bpm_bucket not in self.data["bpm_buckets_seen"]:
                self.data["bpm_buckets_seen"].append(bpm_bucket)
                xp_gained += 20

        level_before = self.level()
        self.data["xp"] = self.xp() + xp_gained
        self._recalculate_level()
        level_after = self.level()

        key = song_name.lower()
        stats = self.data["song_stats"].get(key, {"plays": 0, "best_score": 0, "best_stars": 0})
        stats["plays"] = int(stats.get("plays", 0)) + 1
        stats["best_score"] = max(int(stats.get("best_score", 0)), int(score))
        stats["best_stars"] = max(int(stats.get("best_stars", 0)), stars)
        stats["best_rating_score"] = max(float(stats.get("best_rating_score", 0.0)), float(rating_score))
        stats["best_combo"] = max(int(stats.get("best_combo", 0)), int(max_combo))
        stats["last_accuracy"] = round(accuracy, 4)
        stats["last_stars"] = int(stars)
        stats["last_rating_score"] = round(rating_score, 2)
        stats["last_failed"] = bool(failed)
        stats["last_bpm"] = round(float(bpm), 2)
        stats["last_onset_density"] = round(float(onset_density or 0.0), 3)
        stats["genre"] = str(genre)
        self.data["song_stats"][key] = stats
        self.data["lifetime_max_combo"] = max(int(self.data.get("lifetime_max_combo", 0)), int(max_combo))
        self.data["lifetime_shanra_hits"] = int(self.data.get("lifetime_shanra_hits", 0)) + int(shanra_hits)
        self.data["lifetime_perfect_hits"] = int(self.data.get("lifetime_perfect_hits", 0)) + int(perfect_hits)

        unlocked_achievements = self._update_achievements(
            total_notes=total,
            misses=misses,
            score=score,
            accuracy=accuracy,
            max_combo=max_combo,
            shanra_hits=shanra_hits,
        )

        level_up_text = f" Level up to {level_after}." if level_after > level_before else ""
        achievements_text = ""
        if unlocked_achievements:
            achievements_text = f" Unlocked: {', '.join(unlocked_achievements)}."
        self.data["last_reward"] = (
            f"{song_name}: {stars} stars ({rating_score:.1f}), +{xp_gained} XP, "
            f"genre {genre}, bpm {bpm_bucket}.{level_up_text}{achievements_text}"
        )
        self.save()
        return ProgressionReward(
            stars=stars,
            xp_gained=xp_gained,
            level_before=level_before,
            level_after=level_after,
            accuracy=accuracy,
            genre=genre,
            bpm_bucket=bpm_bucket,
            rating_score=rating_score,
            unlocked_achievements=unlocked_achievements,
        )

    def set_tutorial_progress(self, phase: str, completed: bool) -> None:
        tutorial = self.data.get("tutorial")
        if not isinstance(tutorial, dict):
            tutorial = self._default_tutorial_state()
            self.data["tutorial"] = tutorial
        tutorial["status"] = "completed" if completed else "in_progress"
        tutorial["phase"] = phase
        tutorial["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()

    def achievements(self) -> dict[str, str]:
        achievements = self.data.get("achievements", {})
        if not isinstance(achievements, dict):
            return {}
        return {str(k): str(v) for k, v in achievements.items()}

    def settings(self) -> dict:
        settings = self.data.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}
        normalized = dict(DEFAULT_SETTINGS)
        normalized.update(settings)
        legacy_cue_mode = settings.get("cue_mode")
        if normalized.get("cue_type") not in VALID_CUE_TYPES:
            if legacy_cue_mode == "audio_letters":
                normalized["cue_type"] = "both"
            elif legacy_cue_mode == "audio_only":
                normalized["cue_type"] = "audio"
            elif legacy_cue_mode == "letters_only":
                normalized["cue_type"] = "visual"
        if normalized["cue_type"] not in VALID_CUE_TYPES:
            normalized["cue_type"] = DEFAULT_SETTINGS["cue_type"]
        normalized["speak_letters"] = bool(normalized.get("speak_letters", True))
        normalized["cue_volume"] = self._normalize_cue_volume(
            normalized.get("cue_volume", DEFAULT_SETTINGS["cue_volume"])
        )
        normalized["default_song_volume"] = self._normalize_song_volume(
            normalized.get("default_song_volume", DEFAULT_SETTINGS["default_song_volume"])
        )
        normalized["cue_panning"] = bool(normalized.get("cue_panning", False))
        normalized["music_enabled"] = bool(normalized.get("music_enabled", True))
        normalized["menu_sounds_enabled"] = bool(normalized.get("menu_sounds_enabled", True))
        normalized["auto_skip_intro"] = bool(normalized.get("auto_skip_intro", True))
        normalized["reduced_inputs"] = bool(normalized.get("reduced_inputs", False))
        normalized["difficulty_level"] = max(
            MIN_DIFFICULTY_LEVEL,
            min(MAX_DIFFICULTY_LEVEL, int(normalized.get("difficulty_level", DEFAULT_SETTINGS["difficulty_level"]))),
        )
        normalized["default_start_speed_index"] = max(
            MIN_START_SPEED_INDEX,
            min(
                MAX_START_SPEED_INDEX,
                int(normalized.get("default_start_speed_index", DEFAULT_SETTINGS["default_start_speed_index"])),
            ),
        )
        normalized["keybinds"] = self._normalize_keybinds(normalized.get("keybinds"))
        return normalized

    def update_settings(
        self,
        cue_type: str | None = None,
        cue_volume: float | None = None,
        default_song_volume: float | None = None,
        cue_panning: bool | None = None,
        speak_letters: bool | None = None,
        music_enabled: bool | None = None,
        menu_sounds_enabled: bool | None = None,
        auto_skip_intro: bool | None = None,
        reduced_inputs: bool | None = None,
        difficulty_level: int | None = None,
        default_start_speed_index: int | None = None,
        keybinds: dict | None = None,
    ) -> dict:
        settings = self.settings()
        if cue_type is not None and cue_type in VALID_CUE_TYPES:
            settings["cue_type"] = cue_type
        if cue_volume is not None:
            settings["cue_volume"] = self._normalize_cue_volume(cue_volume)
        if default_song_volume is not None:
            settings["default_song_volume"] = self._normalize_song_volume(default_song_volume)
        if cue_panning is not None:
            settings["cue_panning"] = bool(cue_panning)
        if speak_letters is not None:
            settings["speak_letters"] = bool(speak_letters)
        if music_enabled is not None:
            settings["music_enabled"] = bool(music_enabled)
        if menu_sounds_enabled is not None:
            settings["menu_sounds_enabled"] = bool(menu_sounds_enabled)
        if auto_skip_intro is not None:
            settings["auto_skip_intro"] = bool(auto_skip_intro)
        if reduced_inputs is not None:
            settings["reduced_inputs"] = bool(reduced_inputs)
        if difficulty_level is not None:
            settings["difficulty_level"] = max(
                MIN_DIFFICULTY_LEVEL, min(MAX_DIFFICULTY_LEVEL, int(difficulty_level))
            )
        if default_start_speed_index is not None:
            settings["default_start_speed_index"] = max(
                MIN_START_SPEED_INDEX, min(MAX_START_SPEED_INDEX, int(default_start_speed_index))
            )
        if keybinds is not None:
            settings["keybinds"] = self._normalize_keybinds(keybinds)
        settings.pop("cue_mode", None)
        self.data["settings"] = settings
        self.save()
        return settings

    def _recalculate_level(self) -> None:
        xp = self.xp()
        level = 1
        while xp >= self._xp_needed_for_level(level + 1):
            level += 1
        self.data["level"] = level

    def _xp_needed_for_level(self, level: int) -> int:
        if level <= 1:
            return 0
        # Escalating but steady progression curve.
        return 100 * (level - 1) + 30 * (level - 1) * (level - 2)

    def _performance_rating(
        self,
        accuracy: float,
        misses: int,
        perfect_hits: int,
        good_hits: int,
        ok_hits: int,
        max_combo: int,
        total_notes: int,
        bpm: float,
        onset_density: float,
    ) -> float:
        total_hits = max(1, perfect_hits + good_hits + ok_hits)
        timing_quality = (
            (perfect_hits * 1.0) + (good_hits * 0.72) + (ok_hits * 0.45)
        ) / total_hits
        miss_ratio = misses / max(1, total_notes)
        combo_ratio = max_combo / max(1, total_notes)

        # Difficulty boost is intentionally mild so "hard songs" help, not dominate.
        bpm_factor = min(1.25, max(0.75, bpm / 130.0))
        density_factor = min(1.25, max(0.8, 0.9 + onset_density / 7.5))
        difficulty_factor = (bpm_factor * density_factor) ** 0.5

        base = (
            (accuracy * 100.0 * 0.55)
            + (timing_quality * 100.0 * 0.30)
            + (combo_ratio * 100.0 * 0.15)
        )
        penalty = miss_ratio * 28.0
        rating = max(0.0, min(100.0, (base - penalty) * difficulty_factor))
        return rating

    def _stars_from_rating(self, rating_score: float) -> int:
        if rating_score < 35.0:
            return 0
        if rating_score >= 94.0:
            return 5
        if rating_score >= 82.0:
            return 4
        if rating_score >= 68.0:
            return 3
        if rating_score >= 52.0:
            return 2
        return 1

    def _bpm_bucket(self, bpm: float) -> str:
        if bpm < 90:
            return "slow"
        if bpm < 130:
            return "mid"
        if bpm < 170:
            return "fast"
        return "extreme"

    def _infer_genre(self, song_name: str) -> str:
        identity = self.song_identity(song_name)
        primary = str(identity.get("primary_genre", "")).strip().lower()
        if primary:
            return primary
        lowered = song_name.lower().replace("_", " ").replace("-", " ")
        for genre, words in GENRE_KEYWORDS.items():
            if any(word in lowered for word in words):
                return genre
        return "unknown"

    def _unlock_achievement(self, achievement_id: str) -> bool:
        achievements = self.data.get("achievements")
        if not isinstance(achievements, dict):
            achievements = {}
            self.data["achievements"] = achievements
        if achievement_id in achievements:
            return False
        achievements[achievement_id] = datetime.now(timezone.utc).isoformat()
        return True

    def _update_achievements(
        self,
        total_notes: int,
        misses: int,
        score: int,
        accuracy: float,
        max_combo: int,
        shanra_hits: int,
    ) -> list[str]:
        unlocked: list[str] = []
        plays = sum(int(v.get("plays", 0)) for v in self.data.get("song_stats", {}).values())
        lifetime_shanra = int(self.data.get("lifetime_shanra_hits", 0))
        lifetime_perfect = int(self.data.get("lifetime_perfect_hits", 0))
        genres_seen = len(self.data.get("genres_seen", []))

        def check(condition: bool, aid: str, label: str) -> None:
            if condition and self._unlock_achievement(aid):
                unlocked.append(label)

        check(plays >= 1, "first_song", "First Song")
        check(total_notes >= 20 and misses == 0, "full_combo", "Full Combo")
        check(score >= 5000, "rhythm_machine", "Rhythm Machine")
        check(max_combo >= 25, "combo_25", "Combo 25")
        check(max_combo >= 50, "combo_50", "Combo 50")
        check(accuracy >= 0.9 and total_notes >= 30, "precision_90", "Precision 90%")
        check(int(shanra_hits) >= 8, "shanra_spotter", "Shanra Spotter")
        check(lifetime_shanra >= 40, "shanra_hunter", "Shanra Hunter")
        check(lifetime_perfect >= 200, "timing_master", "Timing Master")
        check(genres_seen >= 4, "genre_explorer", "Genre Explorer")
        check(self.level() >= 5, "level_5", "Level 5")
        check(self.level() >= 10, "level_10", "Level 10")
        return unlocked
