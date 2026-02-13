# Grok Video Desktop Studio (Windows GUI) — **Beta**

**Version:** `0.1.0`

This project is now a **desktop Python GUI app** instead of a Flask web server.

## What changed

- No web server routes/pages.
- Removed legacy Flask template files (`templates/index.html`, `templates/dashboard.html`).
- Uses a split-pane desktop layout.
- The **browser is permanently embedded in the right-hand pane** using Qt WebEngine.
- The left pane contains Grok settings, concept input, generate controls, and logs.

## Features

1. Enter Grok API key and video model (API key only required for API-driven generation).
2. Choose prompt generation mode:
   - Manual prompt (no prompt API call)
   - Grok API prompt generation
   - OpenAI API prompt generation
3. Generate one or more video variants from a concept or manual prompt.
4. Keep a generated-video list in the GUI session.
5. Play selected videos in an in-app video preview player.
6. Open a local video file and preview it in the same in-app player.
7. Extract the last frame from a generated video, save it to `downloads/`, and copy it to clipboard for pasting into Grok's "Type to imagine" tab.
8. Continue from the latest generated video by copying its last frame, auto-pasting it into Grok, and starting a new manual generation.
9. Manual mode repeat count now queues full generation runs and decrements to zero as each video is downloaded.
10. Quickly return to Grok with the **Show Browser (grok.com)** button.
11. Stitch all currently listed videos into a single output file in creation order.
12. Upload a selected video to YouTube by entering title and description in an upload dialog.

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

If video playback feels choppy in the embedded browser, the app now enables a persistent disk cache and Chromium GPU/media flags by default.

For Playwright-based web automation (`grok_web_automation.py`), you can also choose the browser engine:

```bash
export GROK_PLAYWRIGHT_BROWSER=firefox
# or: webkit / chromium
```

You can override cache sizing via environment variables:

- `GROK_BROWSER_DISK_CACHE_BYTES` (default: `536870912`, i.e. 512 MB)
- `GROK_BROWSER_MEDIA_CACHE_BYTES` (default: `268435456`, i.e. 256 MB)

You can also append custom Chromium flags with:

- `QTWEBENGINE_CHROMIUM_FLAGS`

Example:

```bash
export GROK_BROWSER_DISK_CACHE_BYTES=1073741824
export GROK_BROWSER_MEDIA_CACHE_BYTES=536870912
export QTWEBENGINE_CHROMIUM_FLAGS="--max-gum-fps=30"
python app.py
```

## Notes

- In **Manual prompt** mode, clicking Generate uses the embedded right-pane browser session at `grok.com/imagine` (no xAI API call), submits your prompt, waits for output, and downloads the generated video into `downloads/`.
- Downloaded videos are saved under `downloads/`.
- The right-hand pane is always present and opens `https://grok.com` so you can quickly use "Type to imagine".
- The local video preview now uses Qt Multimedia (`QMediaPlayer` + `QVideoWidget`) for more reliable playback.
- Last-frame extraction and video stitching require `ffmpeg` in `PATH`.
- YouTube upload requires a valid OAuth client secret file at `client_secret.json` (written token stored in `youtube_token.json`).

## Support this project

If you find this useful, please consider donating:

- PayPal: https://www.paypal.com/paypalme/dhookster
- Crypto (SOL): `6HiqW3jeF3ymxjK5Fcm6dHi46gDuFmeCeSNdW99CfJjp`
