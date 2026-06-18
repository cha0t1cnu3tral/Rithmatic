from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pygame

from audio_analysis import NoteEvent
from gameplay import HIT_CUE_LEAD_S, GameSession, GameplayCueBank
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


class PlaybackAnalysisAlignmentTests(unittest.TestCase):
    def test_small_detected_start_offset_trims_playback_to_match_analysis(self) -> None:
        app = App.__new__(App)
        app.settings = {"auto_skip_intro": True}
        app.converted_songs_dir = Path("converted")
        spoken: list[str] = []
        app._speak = spoken.append
        analysis = SimpleNamespace(
            start_offset_s=0.32,
            duration_s=2.0,
            notes=[NoteEvent(time_s=0.5, lane="A")],
        )

        with patch("main.trim_song_to_start", return_value=Path("song.start000320.auto.wav")) as trim:
            playback_path = app._align_playback_to_analysis(Path("song.wav"), analysis)

        self.assertEqual(playback_path, Path("song.start000320.auto.wav"))
        trim.assert_called_once_with(Path("song.wav"), 0.32, cache_dir=Path("converted"))
        self.assertEqual(analysis.notes[0].time_s, 0.5)
        self.assertEqual(spoken, [])

    def test_disabled_intro_skip_restores_chart_to_original_playback_timeline(self) -> None:
        app = App.__new__(App)
        app.settings = {"auto_skip_intro": False}
        app.converted_songs_dir = Path("converted")
        analysis = SimpleNamespace(
            start_offset_s=0.32,
            duration_s=2.0,
            notes=[NoteEvent(time_s=0.5, lane="A"), NoteEvent(time_s=1.0, lane="SPACE")],
        )

        playback_path = app._align_playback_to_analysis(Path("song.wav"), analysis)

        self.assertEqual(playback_path, Path("song.wav"))
        self.assertEqual([round(note.time_s, 2) for note in analysis.notes], [0.82, 1.32])
        self.assertEqual(analysis.duration_s, 2.32)
        self.assertEqual(analysis.start_offset_s, 0.0)


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

    def test_audio_first_timing_windows_are_playable_from_cues(self) -> None:
        session = GameSession.__new__(GameSession)
        session.analysis = SimpleNamespace(bpm=120.0)
        session.difficulty_level = 5

        session._configure_tempo_windows()

        self.assertGreaterEqual(session.hit_window, 0.15)
        self.assertGreaterEqual(session.space_hit_window, 0.16)
        self.assertLess(HIT_CUE_LEAD_S, session.hit_window)

    def test_gameplay_cues_do_not_generate_fallback_tones(self) -> None:
        self.assertFalse(hasattr(GameplayCueBank, "_make_tone"))

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

    def test_calibrated_latency_delays_auto_miss_until_hit_can_arrive(self) -> None:
        session = GameSession.__new__(GameSession)
        note = NoteEvent(time_s=1.0, lane="A")
        session.analysis = SimpleNamespace(notes=[note], duration_s=10.0)
        session.input_latency_s = 0.4
        session.hit_window = 0.1
        session.space_hit_window = 0.1
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
        session.fail_miss_penalty = 1.9
        session.paused = False
        session.finished = False
        session.failed = False
        session.cues = SimpleNamespace(
            play_lane=lambda lane, volume=1.0: None,
            play_hit=lambda lane: None,
            play_miss=lambda: None,
            play_combo=lambda: None,
        )
        current_time = [note.time_s - HIT_CUE_LEAD_S + session.input_latency_s - 0.01]
        session._song_time = lambda: current_time[0]

        session.update()

        self.assertFalse(note.judged)
        self.assertEqual(session.misses, 0)

        current_time[0] = note.time_s - HIT_CUE_LEAD_S + session.input_latency_s
        session._judge_lane_hit("A")

        self.assertTrue(note.hit)
        self.assertEqual(session.hits, 1)

    def test_rithm_mode_builds_playable_drum_only_timing_chart(self) -> None:
        session = GameSession.__new__(GameSession)
        session.analysis = SimpleNamespace(
            bpm=120.0,
            notes=[
                NoteEvent(time_s=0.00, lane="A"),
                NoteEvent(time_s=0.03, lane="S"),
                NoteEvent(time_s=0.25, lane="D"),
                NoteEvent(time_s=0.50, lane="SPACE", is_drum=True),
            ],
        )
        session.difficulty_level = 5

        rhythm_notes = session._rithm_notes_from_analysis()

        self.assertEqual([note.lane for note in rhythm_notes], ["SPACE", "SPACE"])
        self.assertTrue(all(note.is_drum for note in rhythm_notes))
        self.assertEqual([round(note.time_s, 2) for note in rhythm_notes], [0.0, 0.5])

    def test_rithm_mode_any_key_uses_calibrated_drum_hit(self) -> None:
        session = GameSession.__new__(GameSession)
        note = NoteEvent(time_s=1.0, lane="SPACE", is_drum=True)
        session.analysis = SimpleNamespace(notes=[note])
        session.game_mode = "rithm"
        session.keybinds = {
            "speed_down": "j",
            "speed_up": "l",
            "pause": "h",
        }
        session.paused = False
        session.input_cooldown_s = 0.03
        session.last_key_press_s = {}
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
        session._key_code = lambda action: pygame.key.key_code(session.keybinds[action])

        handled = session.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q))

        self.assertTrue(handled)
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

    def test_game_mode_setting_is_validated_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "profile.json"
            profile = PlayerProfile(profile_path)

            profile.update_settings(game_mode="rithm")
            reloaded = PlayerProfile(profile_path)

            self.assertEqual(reloaded.settings()["game_mode"], "rithm")

            reloaded.data["settings"]["game_mode"] = "invalid"
            reloaded.save()
            normalized = PlayerProfile(profile_path)

            self.assertEqual(normalized.settings()["game_mode"], "chart")

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

    def test_rating_keeps_nonzero_credit_for_rough_rithm_runs_with_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = PlayerProfile(Path(tmp) / "profile.json")

            reward = profile.apply_song_result(
                song_name="rough rithm song",
                score=250,
                hits=6,
                misses=94,
                bpm=120.0,
                ok_hits=6,
                max_combo=2,
                onset_density=4.0,
                failed=True,
            )
            all_miss_reward = profile.apply_song_result(
                song_name="all miss rithm song",
                score=0,
                hits=0,
                misses=100,
                bpm=120.0,
                onset_density=4.0,
                failed=True,
            )

            self.assertGreater(reward.rating_score, 0.0)
            self.assertEqual(all_miss_reward.rating_score, 0.0)


if __name__ == "__main__":
    unittest.main()
