# AGENTS.md

## Project
Rhythm game for local song files with strong screen reader accessibility and simple keyboard-first controls.

## Core Experience
- User drops audio files into `songs/`.
- User launches game, enters main menu, goes to Songs, picks a song, then plays.
- Game auto-analyzes rhythm (BPM, beats, onsets), maps timestamps to playable note events, and runs a scrolling note gameplay loop.

## Required Controls
- `Space`: Drum lane input.
- `A S D F`: Pitched note lanes (`A` highest, `F` lowest).
- `J`: Slow speed.
- `L`: Fast speed.
- `H`: Pause/unpause.
- Menu navigation: Arrow keys + Enter, plus mouse support.

## Main Menu Requirements
- Options in order:
  1. Songs
  2. Help
  3. Credits
  4. Achievements
  5. Quit
- Visual style: black background, high-contrast white text.
- Screen reader behavior:
  - Announces focused menu item on arrow movement and hover.
  - Announces confirmations and screen transitions.

## Audio Requirements
- Background menu music from `assets/music/mmm1.mp3`..`mmm4.mp3`.
- Background music must:
  - Play only in main menu context.
  - Stop/fade out cleanly when entering Songs/gameplay.
  - Fade in cleanly when returning to menu.
- UI SFX in `assets/sounds/`:
  - `item selected.mp3` for focus/click navigation.
  - `song selected.mp3` when confirming a song to start.

## Accessibility Requirements
- Reuse and integrate `screenreader.py` NVDA bridge helper.
- Every navigable UI focus target must have spoken feedback.
- Gameplay state changes are announced (start, paused, resumed, song finished).

## Backend Flow (Target)
1. User puts song in `songs/`.
2. User picks song from Songs screen.
3. `librosa.load` -> waveform and sample rate.
4. Detect BPM.
5. Detect beats.
6. Detect onsets.
7. Convert timestamps to note objects (lane + spawn time + hit window).
8. Game loop spawns and scrolls notes by current speed mode.
9. Player inputs are judged and scored.

## Initial Architecture
- `main.py`
  - App lifecycle + state machine (`MENU`, `SONGS`, `HELP`, `CREDITS`, `ACHIEVEMENTS`, `GAME`).
- `audio_analysis.py`
  - Song analysis and note map generation from song file.
- `audio_runtime.py`
  - Music/SFX management, fade handling, and asset loading.
- `ui.py`
  - Menu rendering, focus logic, mouse + keyboard input helpers.
- `gameplay.py`
  - Note highway, hit detection, speed controls, pause behavior, score.
- `screenreader.py`
  - Existing bridge module (already present).

## Milestone Plan
1. Build runnable app shell with window, state machine, and accessible main menu.
2. Add songs screen and song discovery from `songs/` with mouse + keyboard selection.
3. Add audio runtime manager (menu music rotation + fade, UI SFX).
4. Add analysis pipeline (`librosa`) and transform output to playable note events.
5. Add gameplay with key mappings, pause, speed switching, and scoring.
6. Add help/credits/achievements content and spoken announcements.
7. Add validation + polish pass for accessibility and transitions.

## Execution Notes
- Start with Milestones 1-3 in this pass so UI/audio/accessibility foundation is solid.
- Then implement analysis/gameplay in Milestones 4-5.
- Keep files small and testable.
- Preserve simple controls and low-friction UX over complex visuals.

## Next Step (User Confirmed)
- Prioritize accessible gameplay feedback through audio cues, not visuals.
- User will collect and add the required gameplay cue files to `assets/sounds/`.
- After cue files are added, wire per-lane/per-state audio cues into gameplay:
  - Lane approach cues (timing cues for incoming notes).
  - Hit confirmation cues.
  - Miss/error cues.
  - Optional combo/streak cues.
- Keep all gameplay cues screen-reader-friendly and usable without sight.
