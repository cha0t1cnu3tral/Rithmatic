from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gameplay import GameSession
from main import App


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


if __name__ == "__main__":
    unittest.main()
