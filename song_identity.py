from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SongIdentity:
    primary_genre: str
    subgenres: list[str]
    mood_tags: list[str]
    energy_tier: str


_GENRE_KEYWORDS: dict[str, list[str]] = {
    "rock": ["rock", "alt", "alternative", "metal", "punk", "grunge", "hardcore", "emo"],
    "pop": ["pop", "dance pop", "synthpop", "electropop", "teen pop", "chart"],
    "hip-hop": ["hip hop", "hip-hop", "rap", "trap", "drill", "boom bap", "grime"],
    "r&b": ["r&b", "rnb", "soul", "neo soul", "quiet storm"],
    "electronic": ["edm", "electro", "house", "techno", "trance", "dubstep", "dnb", "drum and bass"],
    "classical": ["classical", "orchestra", "orchestral", "baroque", "romantic", "sonata", "symphony"],
    "jazz": ["jazz", "swing", "bebop", "fusion", "big band", "latin jazz"],
    "country": ["country", "bluegrass", "americana", "honky tonk"],
    "folk": ["folk", "acoustic", "singer songwriter", "singer-songwriter", "indie folk"],
    "latin": ["latin", "reggaeton", "bachata", "salsa", "cumbia", "merengue", "corridos"],
    "reggae": ["reggae", "dancehall", "dub"],
    "afro": ["afrobeat", "afrobeats", "amapiano", "highlife"],
    "k-pop": ["k-pop", "kpop"],
    "j-pop": ["j-pop", "jpop", "city pop"],
    "ambient": ["ambient", "drone", "meditation", "new age"],
    "lofi": ["lofi", "lo-fi", "chillhop", "study beats"],
    "cinematic": ["soundtrack", "score", "trailer", "cinematic"],
    "blues": ["blues", "delta blues", "chicago blues"],
    "gospel": ["gospel", "worship", "praise"],
    "world": ["world", "traditional", "ethnic", "global"],
}

_SUBGENRE_KEYWORDS: dict[str, list[str]] = {
    "alt rock": ["alt rock", "alternative"],
    "metalcore": ["metalcore"],
    "pop punk": ["pop punk"],
    "synthwave": ["synthwave"],
    "future bass": ["future bass"],
    "deep house": ["deep house"],
    "tech house": ["tech house"],
    "progressive house": ["progressive house"],
    "trap": ["trap"],
    "drill": ["drill"],
    "boom bap": ["boom bap"],
    "neo soul": ["neo soul"],
    "orchestral": ["orchestral", "orchestra"],
    "piano": ["piano"],
    "indie folk": ["indie folk"],
    "reggaeton": ["reggaeton"],
    "afrobeats": ["afrobeats", "afrobeat"],
    "amapiano": ["amapiano"],
    "drum and bass": ["drum and bass", "dnb"],
    "dubstep": ["dubstep"],
    "trailer score": ["trailer"],
    "lo-fi hip-hop": ["lo-fi", "lofi", "chillhop"],
}


def _clean_text(song_name: str) -> str:
    lowered = song_name.lower().replace("_", " ").replace("-", " ")
    stem = Path(lowered).stem
    return " ".join(stem.split())


def _detect_primary_from_text(text: str) -> str | None:
    for genre, keywords in _GENRE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return genre
    return None


def _detect_subgenres(text: str) -> list[str]:
    found: list[str] = []
    for subgenre, keywords in _SUBGENRE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            found.append(subgenre)
    return found[:4]


def identify_song_identity(
    song_name: str,
    analysis_genre: str,
    bpm: float,
    onset_density: float,
    lane_profile: str,
) -> SongIdentity:
    text = _clean_text(song_name)
    normalized_analysis = (analysis_genre or "").strip().lower()

    primary = _detect_primary_from_text(text)
    if primary is None:
        if normalized_analysis in {"edm", "electronic"}:
            primary = "electronic"
        elif normalized_analysis in {"hip-hop", "hiphop"}:
            primary = "hip-hop"
        elif normalized_analysis:
            primary = normalized_analysis
        else:
            primary = "unknown"

    subgenres = _detect_subgenres(text)
    if not subgenres:
        if primary == "electronic" and bpm >= 150:
            subgenres.append("drum and bass")
        elif primary == "rock" and onset_density >= 3.0:
            subgenres.append("alt rock")
        elif primary == "hip-hop" and bpm < 115:
            subgenres.append("boom bap")

    mood_tags: list[str] = []
    if lane_profile in {"bright", "high"}:
        mood_tags.append("bright")
    if lane_profile in {"low"}:
        mood_tags.append("heavy")
    if bpm < 90:
        mood_tags.append("laid-back")
    elif bpm > 145:
        mood_tags.append("driving")
    if onset_density >= 3.2:
        mood_tags.append("dense")

    if onset_density < 1.4:
        energy_tier = "low"
    elif onset_density < 2.6:
        energy_tier = "mid"
    else:
        energy_tier = "high"

    return SongIdentity(
        primary_genre=primary,
        subgenres=subgenres,
        mood_tags=mood_tags,
        energy_tier=energy_tier,
    )
