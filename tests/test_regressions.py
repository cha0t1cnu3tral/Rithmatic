from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pygame

from audio_analysis import NoteEvent
from gameplay import APPROACH_CUE_LEAD_S, GameSession
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


class CalibrationFlowTests(unittest.TestCase):
    def test_calibration_trial_plays_drum_and_records_space_delay(self) -> None:
        app = App.__new__(App)
        played_lanes: list[str] = []
        recorded_delays: list[float] = []
        app.help_sound_cues = SimpleNamespace(
            play_lane=lambda lane, volume: played_lanes.append(lane)
        )
        app.profile = SimpleNamespace(
            record_latency_calibration=lambda delay: recorded_delays.append(delay) or delay,
            settings=lambda: {"input_latency_s": recorded_delays[-1]},
        )
        app.settings = {}
        app.calibration_cue_started_ms = None
        app.calibration_cue_due_ms = 0
        app.calibration_status = ""
        app._speak = lambda text: None

        with patch("main.random.randint", return_value=700), patch(
            "main.pygame.time.get_ticks", return_value=1000
        ):
            app._begin_calibration_trial()
        with patch("main.pygame.time.get_ticks", return_value=1700):
            app._update_calibration()
        with patch("main.pygame.time.get_ticks", return_value=1900):
            app._handle_calibration_input(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))

        self.assertEqual(played_lanes, ["SPACE"])
        self.assertEqual(recorded_delays, [0.2])
        self.assertEqual(app.settings["input_latency_s"], 0.2)


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

    def test_approach_cue_lead_is_near_instant(self) -> None:
        session = GameSession.__new__(GameSession)
        session.analysis = SimpleNamespace(bpm=120.0)
        session.difficulty_level = 5

        session._configure_tempo_windows()

        self.assertEqual(session.approach_lead_s, APPROACH_CUE_LEAD_S)
        self.assertLessEqual(session.hit_window, 0.1)

    def test_calibrated_latency_is_subtracted_when_judging_hits(self) -> None:
        session = GameSession.__new__(GameSession)
        note = NoteEvent(time_s=1.0, lane="A")
        session.analysis = SimpleNamespace(notes=[note])
        session.input_latency_s = 0.2
        session.hit_window = 0.08
        session.space_hit_window = 0.075
        session.perfect_window = 0.03
        session.good_window = 0.06
        session.score = 0
        session.hits = 0
        session.misses = 0
        session.combo = 0
        session.max_combo = 0
        session.perfect_hits = 0
        session.good_hits = 0
        session.ok_hits = 0
        session.health = 100.0
        session.health_max = 150.0
        session.fail_hit_recovery = 3.2
        session.cues = SimpleNamespace(play_hit=lambda lane: None, play_combo=lambda: None)
        session._song_time = lambda: 1.2

        session._judge_lane_hit("A")

        self.assertTrue(note.hit)
        self.assertEqual(session.hits, 1)


class PlayerProfileTests(unittest.TestCase):
    def test_latency_calibration_is_clamped_logged_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "profile.json"
            profile = PlayerProfile(profile_path)

            profile.record_latency_calibration(9.0)
            reloaded = PlayerProfile(profile_path)

            self.assertEqual(reloaded.settings()["input_latency_s"], 0.75)
            self.assertEqual(reloaded.data["calibration_history"][-1]["input_latency_ms"], 750)

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
