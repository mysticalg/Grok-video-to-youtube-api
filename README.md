# Grok Video Desktop Studio (Windows GUI) — **Beta**

**Version:** `1.0.0`

A desktop-first Python/PyQt app for generating Grok videos, iterating quickly in-browser, and uploading finished results to YouTube.

## What it does

- Generate videos from:
  - Manual prompts (embedded browser workflow)
  - Grok API prompt generation
  - OpenAI API prompt generation
- Queue multiple variants in one run.
- Keep a generated-video list in-session.
- Preview generated or local videos inside the app.
- Continue from the latest frame or a local seed image.
- Stitch all listed videos into one final output, with optional crossfade blending between clips.
- Optional stitched-output enhancements:
  - Frame interpolation (24 → 48 fps or 60 fps) for smoother motion.
  - AI-style 2x upscaling (Lanczos, capped at 4K output).
- Stitching now shows a progress window with elapsed/ETA and active stitch settings.
- Configure video options (including crossfade duration) from settings.
- Set a custom download folder in settings.
- Set a default manual prompt template in settings.
- Upload a selected video to YouTube.

## Download Windows binary (recommended)

If you just want to run the app on Windows, download the prebuilt zip here:

- **Releases:** https://github.com/mysticalg/Grok-video-to-youtube-api/releases
- **Latest CI artifacts (main branch builds):** https://github.com/dhookster/Grok-video-to-youtube-api/actions/workflows/windows-build-release.yml

Look for `GrokVideoDesktopStudio-windows-x64.zip`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Environment variables

- `GROK_API_KEY`
- `GROK_CHAT_MODEL` (default: `grok-3-mini`)
- `GROK_VIDEO_MODEL` (default: `grok-video-latest`)
- `XAI_API_BASE` (default: `https://api.x.ai/v1`)
- `OPENAI_API_KEY`
- `OPENAI_CHAT_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_API_BASE` (default: `https://api.openai.com/v1`)
- `GROK_PLAYWRIGHT_BROWSER` (default: `chromium`; options: `chromium`, `firefox`, `webkit`)

## Browser performance tuning (embedded Chromium)

If in-app playback feels choppy, the app enables a persistent disk cache and Chromium GPU/media flags by default.

Optional overrides:

- `GROK_BROWSER_DISK_CACHE_BYTES` (default: `536870912`, 512 MB)
- `GROK_BROWSER_MEDIA_CACHE_BYTES` (default: `268435456`, 256 MB)
- `QTWEBENGINE_CHROMIUM_FLAGS`

Example:

```bash
export GROK_BROWSER_DISK_CACHE_BYTES=1073741824
export GROK_BROWSER_MEDIA_CACHE_BYTES=536870912
export QTWEBENGINE_CHROMIUM_FLAGS="--max-gum-fps=30"
python app.py
```

## Notes

- Manual prompt mode runs against the embedded `grok.com/imagine` browser session (no xAI API generation call).
- Downloaded videos are saved under `downloads/`.
- Last-frame extraction, video stitching, interpolation, and upscaling require `ffmpeg` in `PATH`.
- YouTube upload requires `client_secret.json` (token saved to `youtube_token.json`).

## Support this project ☕

If this saves you hours, grab me a ☕.

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=000000)](https://buymeacoffee.com/dhooksterm)

Also supported via:

- PayPal: https://www.paypal.com/paypalme/dhookster
- Crypto (SOL): `6HiqW3jeF3ymxjK5Fcm6dHi46gDuFmeCeSNdW99CfJjp`

### Demo callout

Use this line in demos/tutorials:

> If this saves you hours, grab me a ☕.

## Build Windows binaries on GitHub

This repository includes a GitHub Actions workflow at `.github/workflows/windows-build-release.yml`.

- On push to `main`, it builds a Windows executable with PyInstaller and uploads a workflow artifact.
- On tags that start with `v` (for example `v1.0.1`), it also uploads the same `.zip` to a GitHub Release.

To enable this in your fork/remote:

1. Ensure your repository has this workflow committed.
2. Push to GitHub.
3. Create and push a version tag:

```bash
git tag v1.0.1
git push origin v1.0.1
```

Then download `GrokVideoDesktopStudio-windows-x64.zip` from Actions artifacts or the tagged Release.
