# Rithmatic

Rithmatic is a keyboard-first rhythm game for local audio files, with screen reader support and audio gameplay cues.

## Play

1. Download and extract the Windows release.
2. Put audio files in the included `songs` folder.
3. Launch `Rithmatic.exe`.

## Controls

- `Space`: drum lane
- `A S D F`: pitched lanes
- `J` / `L`: slower / faster
- `H`: pause or resume
- Arrow keys and `Enter`: menu navigation

## Run From Source

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python main.py
```

Local song files and generated player profiles are intentionally excluded from version control.
