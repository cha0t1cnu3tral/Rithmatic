from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gameplay import GameSession
from main import App
from progression import ACHIEVEMENT_LABELS, PlayerProfile


class SongDiscoveryTests(unittest.TestCase):
    def test_song_paths_excludes_non_audio_and_generated_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            songs_dir = Path(tmp)
            for name in (
                ".gitkeep",
                "README.txt",
                "clip.wav",
                "track.mp3",
                "video.mp4",
                "track.rate1200.auto.wav",
            ):
                (songs_dir / name).write_bytes(b"test")

            app = App.__new__(App)
            app.songs_dir = songs_dir

            self.assertEqual(
                [path.name for path in app._song_paths()],
                ["clip.wav", "track.mp3"],
            )


class PlaybackRateTests(unittest.TestCase):
    def test_failed_rate_conversion_falls_back_to_normal_timeline_rate(self) -> None:
        session = GameSession.__new__(GameSession)
        base_path = Path("song.wav")

        with patch(
            "gameplay.convert_song_to_rate_wav",
            side_effect=RuntimeError("conversion unavailable"),
        ):
            playback_path, timeline_rate = session._resolve_rate_playback_path(base_path, 1.5)

        self.assertEqual(playback_path, base_path)
        self.assertEqual(timeline_rate, 1.0)

    def test_reaction_time_controls_approach_cue_lead(self) -> None:
        session = GameSession.__new__(GameSession)
        session.analysis = SimpleNamespace(bpm=120.0)
        session.difficulty_level = 5
        session.reaction_time_s = 1.7

        session._configure_tempo_windows()

        self.assertEqual(session.approach_lead_s, 1.7)


class PlayerProfileTests(unittest.TestCase):
    def test_reaction_time_setting_is_clamped_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "profile.json"
            profile = PlayerProfile(profile_path)

            profile.update_settings(reaction_time_s=9.0)

            self.assertEqual(PlayerProfile(profile_path).settings()["reaction_time_s"], 2.0)

    def test_expanded_achievement_catalog_unlocks_new_milestones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = PlayerProfile(Path(tmp) / "profile.json")

            reward = profile.apply_song_result(
                song_name="test song",
                score=12000,
                hits=100,
                misses=0,
                bpm=180.0,
                perfect_hits=100,
                max_combo=100,
            )

            self.assertGreater(len(ACHIEVEMENT_LABELS), 12)
            self.assertIn("Score Crusher", reward.unlocked_achievements)
            self.assertIn("Combo 100", reward.unlocked_achievements)
            self.assertIn("Clean Run", reward.unlocked_achievements)


if __name__ == "__main__":
    unittest.main()
