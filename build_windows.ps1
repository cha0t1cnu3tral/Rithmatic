Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-CheckedCommand {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Description,
    [Parameter(Mandatory = $true)]
    [scriptblock]$Command
  )
  Write-Host $Description
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed ($LASTEXITCODE): $Description"
  }
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$appName = "Rithmatic"
$releaseRoot = Join-Path $projectRoot "release"
$releaseDir = Join-Path $releaseRoot "Rithmatic-Windows"
$zipPath = Join-Path $projectRoot "Rithmatic-Windows.zip"

Invoke-CheckedCommand "Installing build dependencies with uv..." {
  uv pip install -r requirements-build.txt
}

Write-Host "Cleaning previous build outputs..."
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path $releaseRoot) { Remove-Item -Recurse -Force $releaseRoot }
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }

Invoke-CheckedCommand "Building one-file executable with PyInstaller..." {
  uv run python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name "$appName" `
    --collect-all librosa `
    --collect-all scipy `
    --collect-all soundfile `
    --collect-all audioread `
    --collect-all imageio_ffmpeg `
    --collect-all pygame `
    --collect-all accessible_output2 `
    main.py
}

Write-Host "Staging release folder..."
New-Item -ItemType Directory -Path $releaseDir | Out-Null
Copy-Item -Path (Join-Path "dist" "$appName.exe") -Destination (Join-Path $releaseDir "$appName.exe")
Copy-Item -Path "assets" -Destination (Join-Path $releaseDir "assets") -Recurse
New-Item -ItemType Directory -Path (Join-Path $releaseDir "songs") | Out-Null

@"
Drop your song files into this folder before launching Rithmatic.

Supported formats include mp3, wav, ogg, flac, m4a, aac, opus, webm, mp4, and mkv.
"@ | Set-Content -Path (Join-Path $releaseDir "songs\README.txt")

@"
Rithmatic Windows Release

Contents:
- Rithmatic.exe
- assets\
- songs\

Setup:
1. Keep Rithmatic.exe, assets, and songs together in the same folder.
2. Put your song files inside the songs folder.
3. Launch Rithmatic.exe.
"@ | Set-Content -Path (Join-Path $releaseDir "README.txt")

Write-Host "Creating release zip..."
Compress-Archive -Path (Join-Path $releaseDir "*") -DestinationPath $zipPath

Write-Host "Build complete."
Write-Host "Executable: $releaseDir\$appName.exe"
Write-Host "Release folder: $releaseDir"
Write-Host "Release zip: $zipPath"
