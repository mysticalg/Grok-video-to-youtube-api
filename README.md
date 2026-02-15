# Grok Video Studio (Windows/macOS desktop + Android companion) — **Beta**

**Version:** `1.0.0`

A desktop-first Python/PySide6 app for generating Grok videos, iterating quickly in-browser, stitching clips, and uploading finished results to social platforms.

## What it does

- Generate videos from:
  - Manual prompts (embedded browser workflow)
  - Grok API prompt generation
  - OpenAI API prompt generation
- Queue multiple variants in one run.
- Choose video resolution (480p/720p), duration (6s/10s), and aspect ratio (2:3, 3:2, 1:1, 9:16, 16:9) before generating prompts/jobs; these are also applied when setting Grok Imagine options in Populate Video/Image flows.
- Keep a generated-video list in-session.
- Preview generated or local videos inside the app.
- Continue from the latest frame or a local seed image.
- Stitch all listed videos into one final output:
  - Hard-cut or crossfade transitions
  - Optional frame interpolation (48/60 fps)
  - Optional 2x upscaling (capped at 4K)
  - Optional custom WAV/MP3 music mix with per-track volume and fade controls
- See stitch progress with elapsed/ETA and active setting summary.
- Configure app behavior from settings dialogs (Model/API settings, video options, default manual prompt, custom download folder).
- Upload the selected local video to:
  - YouTube
  - Facebook Page
  - Instagram Reels (requires a publicly reachable video URL)

## Download Windows binary (recommended)

If you just want to run the app on Windows, download the prebuilt zip here:

- **Releases:** https://github.com/mysticalg/Grok-video-to-youtube-api/releases
- **Latest CI builds (all workflow runs):** https://github.com/mysticalg/Grok-video-to-youtube-api/actions

Look for:
- `GrokVideoDesktopStudio-windows-x64.zip`
- `GrokVideoDesktopStudio-macos-universal2.zip`
- `GrokVideoDesktopStudio-android-play-aab`


## Platform support

- **Windows desktop app:** packaged with PyInstaller.
- **macOS desktop app:** packaged with PyInstaller (`.app` zipped artifact).
- **Android companion app:** native WebView wrapper that opens `https://grok.com/imagine`, built as a signed **AAB** for Google Play upload.

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

### Generation

- `GROK_API_KEY`
- `GROK_CHAT_MODEL` (default: `grok-3-mini`)
- `GROK_VIDEO_MODEL` (default: `grok-video-latest`)
- `XAI_API_BASE` (default: `https://api.x.ai/v1`)
- `OPENAI_API_KEY`
- `OPENAI_CHAT_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_API_BASE` (default: `https://api.openai.com/v1`)

### Embedded browser/runtime

- `GROK_PLAYWRIGHT_BROWSER` (default: `chromium`; options: `chromium`, `firefox`, `webkit`)
- `GROK_BROWSER_CACHE_DIR` (optional custom QtWebEngine cache directory)
- `GROK_BROWSER_DISK_CACHE_BYTES` (default: `536870912`, 512 MB)
- `GROK_BROWSER_MEDIA_CACHE_BYTES` (default: `268435456`, 256 MB)
- `QTWEBENGINE_CHROMIUM_FLAGS` (optional additional Chromium flags)

## Browser performance tuning (embedded Chromium)

If in-app playback feels choppy, the app enables a persistent disk cache and Chromium GPU/media flags by default.

Example:

```bash
export GROK_BROWSER_DISK_CACHE_BYTES=1073741824
export GROK_BROWSER_MEDIA_CACHE_BYTES=536870912
export QTWEBENGINE_CHROMIUM_FLAGS="--max-gum-fps=30"
python app.py
```

## Upload requirements

- **YouTube:** requires `client_secret.json` (token stored in `youtube_token.json` after OAuth).
- **Facebook:** requires Graph API credentials configured via environment/settings used by the uploader workflow.
- **Instagram Reels:** requires Meta Graph API credentials and a publicly accessible HTTP(S) video URL.

## Notes

- Manual prompt mode runs against the embedded `grok.com/imagine` browser session (no xAI API generation call).
- Downloaded videos are saved under `downloads/` unless you choose a custom folder in settings.
- Last-frame extraction, stitching, interpolation, upscaling, and custom audio mixing require `ffmpeg` in `PATH`.

## Legal

- [Terms of Service](TERMS_OF_SERVICE.md)
- [Privacy Policy](PRIVACY_POLICY.md)

## Support this project ☕

If this saves you hours, grab me a ☕.

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=000000)](https://buymeacoffee.com/dhooksterm)

Also supported via:

- PayPal: https://www.paypal.com/paypalme/dhookster
- Crypto (SOL): `6HiqW3jeF3ymxjK5Fcm6dHi46gDuFmeCeSNdW99CfJjp`

## Build distributables on GitHub

This repository now includes:

- `.github/workflows/windows-build-release.yml` (Windows desktop zip)
- `.github/workflows/macos-build-release.yml` (macOS desktop app zip)
- `.github/workflows/android-build-release.yml` (Android signed `.aab` for Play Store upload)

Behavior:

- On push to `main`, workflows build platform artifacts and upload them to Actions artifacts.
- On tags that start with `v` (for example `v1.0.1`), workflows also attach the artifacts to a GitHub Release.

### Android Play Store signing secrets

Set these repository secrets for the Android workflow:

- `ANDROID_SIGNING_KEY_BASE64` (base64-encoded upload keystore file)
- `ANDROID_KEYSTORE_PASSWORD`
- `ANDROID_KEY_ALIAS`
- `ANDROID_KEY_PASSWORD`

Once set, the workflow outputs a signed `app-release.aab` that can be uploaded directly to Google Play Console.

To create and push a release tag:

```bash
git tag v1.0.1
git push origin v1.0.1
```
