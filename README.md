# Grok Video to YouTube API (GUI)

A Flask GUI app that lets users:

1. Register/login (stored locally with hashed passwords)
2. Enter Grok + YouTube API credentials in session
3. Ask Grok Chat to create a polished video prompt
4. Generate a 10-second video via Grok Imagine API in 720p, with fallback to 420p
5. Download the generated video
6. Upload video to YouTube with Grok-generated title/description/hashtags

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open: `http://localhost:5000`

## Notes

- The app stores user auth data in `users.db`.
- API keys and client secrets are stored in Flask session data for the active login session.
- YouTube OAuth token is persisted per user in `youtube_token_<username>.json`.
- Video files are downloaded under `downloads/`.

## Important API assumptions

This app targets these xAI endpoints:

- `POST /v1/chat/completions`
- `POST /v1/imagine/video/generations`
- `GET /v1/imagine/video/generations/{id}`

If your tenant uses different endpoint names/fields, update `start_video_job()` and `poll_video_job()` in `app.py`.
