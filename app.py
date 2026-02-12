import json
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from youtube_uploader import upload_video_to_youtube

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "users.db"
DOWNLOAD_DIR = BASE_DIR / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")


@dataclass
class GrokConfig:
    api_key: str
    chat_model: str
    image_model: str


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.commit()


init_db()


def get_user(username: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT username, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()


def create_user(username: str, password: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password)),
        )
        conn.commit()


def logged_in() -> bool:
    return bool(session.get("username"))


def require_login():
    if not logged_in():
        flash("Please login first.", "error")
        return redirect(url_for("index"))
    return None


def load_grok_config() -> Optional[GrokConfig]:
    api_key = session.get("grok_api_key") or os.getenv("GROK_API_KEY")
    if not api_key:
        return None
    return GrokConfig(
        api_key=api_key,
        chat_model=session.get("chat_model", "grok-2-latest"),
        image_model=session.get("image_model", "grok-2-image-latest"),
    )


def call_grok_chat(api_key: str, model: str, system: str, user: str) -> str:
    response = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.9,
        },
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()


def start_video_job(api_key: str, model: str, prompt: str, resolution: str, duration: int = 10) -> str:
    response = requests.post(
        "https://api.x.ai/v1/imagine/video/generations",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "prompt": prompt,
            "duration_seconds": duration,
            "resolution": resolution,
            "fps": 24,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("id")


def poll_video_job(api_key: str, job_id: str, timeout_s: int = 420) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        response = requests.get(
            f"https://api.x.ai/v1/imagine/video/generations/{job_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status == "succeeded":
            return payload
        if status == "failed":
            raise RuntimeError(payload.get("error", "Video generation failed"))
        time.sleep(5)
    raise TimeoutError("Timed out waiting for video generation.")


def download_video(video_url: str, username: str) -> Path:
    timestamp = int(time.time())
    file_path = DOWNLOAD_DIR / f"{username}_{timestamp}.mp4"
    with requests.get(video_url, stream=True, timeout=240) as response:
        response.raise_for_status()
        with open(file_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
    return file_path


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        action = request.form.get("action")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password are required.", "error")
            return redirect(url_for("index"))

        if action == "register":
            try:
                create_user(username, password)
                flash("Registration complete. Please log in.", "success")
            except sqlite3.IntegrityError:
                flash("Username already exists.", "error")
            return redirect(url_for("index"))

        user = get_user(username)
        if user and check_password_hash(user["password_hash"], password):
            session["username"] = username
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username/password.", "error")
    return render_template("index.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("index"))


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    auth_redirect = require_login()
    if auth_redirect:
        return auth_redirect

    if request.method == "POST":
        session["grok_api_key"] = request.form.get("grok_api_key", "").strip()
        session["chat_model"] = request.form.get("chat_model", "grok-2-latest").strip()
        session["image_model"] = request.form.get("image_model", "grok-2-image-latest").strip()
        session["youtube_client_secret_json"] = request.form.get("youtube_client_secret_json", "").strip()
        flash("API settings saved to your session.", "success")
        return redirect(url_for("dashboard"))

    return render_template(
        "dashboard.html",
        username=session.get("username"),
        grok_api_key=session.get("grok_api_key", ""),
        chat_model=session.get("chat_model", "grok-2-latest"),
        image_model=session.get("image_model", "grok-2-image-latest"),
        youtube_client_secret_json=session.get("youtube_client_secret_json", ""),
    )


@app.route("/generate", methods=["POST"])
def generate():
    auth_redirect = require_login()
    if auth_redirect:
        return auth_redirect

    config = load_grok_config()
    if not config:
        flash("Set Grok API key first in API settings.", "error")
        return redirect(url_for("dashboard"))

    concept = request.form.get("concept", "").strip()
    if not concept:
        flash("Please add a concept.", "error")
        return redirect(url_for("dashboard"))

    try:
        prompt = call_grok_chat(
            config.api_key,
            config.chat_model,
            "You write highly visual prompts for short cinematic AI videos.",
            f"Create one polished video prompt for a 10 second scene in 720p from this concept: {concept}",
        )

        metadata_raw = call_grok_chat(
            config.api_key,
            config.chat_model,
            (
                "Return ONLY compact JSON with keys: title, description, hashtags. "
                "hashtags must be an array of 5 short tags beginning with #."
            ),
            f"Create YouTube metadata for this video concept and prompt. Concept: {concept}\nPrompt:{prompt}",
        )

        metadata = json.loads(metadata_raw)
        session["generated_prompt"] = prompt
        session["generated_title"] = metadata.get("title", "Grok Video")
        session["generated_description"] = metadata.get("description", "")
        session["generated_hashtags"] = " ".join(metadata.get("hashtags", []))

        video_job_id = None
        chosen_resolution = None
        for resolution in ["1280x720", "640x420"]:
            try:
                video_job_id = start_video_job(config.api_key, config.image_model, prompt, resolution, duration=10)
                chosen_resolution = resolution
                break
            except requests.HTTPError:
                continue

        if not video_job_id:
            raise RuntimeError("Could not start a video generation job at 720p or 420p.")

        result = poll_video_job(config.api_key, video_job_id)
        video_url = result.get("output", {}).get("video_url") or result.get("video_url")
        if not video_url:
            raise RuntimeError("Generation succeeded but no video URL was returned.")

        file_path = download_video(video_url, session["username"])
        session["video_file_path"] = str(file_path)
        session["video_resolution"] = chosen_resolution
        flash(f"Video generated in {chosen_resolution} and downloaded successfully.", "success")
    except Exception as exc:
        flash(f"Generation failed: {exc}", "error")

    return redirect(url_for("dashboard"))


@app.route("/download")
def download():
    auth_redirect = require_login()
    if auth_redirect:
        return auth_redirect

    video_file_path = session.get("video_file_path")
    if not video_file_path or not Path(video_file_path).exists():
        flash("No generated video found in session.", "error")
        return redirect(url_for("dashboard"))
    return send_file(video_file_path, as_attachment=True)


@app.route("/youtube/upload", methods=["POST"])
def youtube_upload():
    auth_redirect = require_login()
    if auth_redirect:
        return auth_redirect

    video_file_path = session.get("video_file_path")
    if not video_file_path or not Path(video_file_path).exists():
        flash("Generate a video before uploading to YouTube.", "error")
        return redirect(url_for("dashboard"))

    client_secret_json = session.get("youtube_client_secret_json", "").strip()
    if not client_secret_json:
        flash("Paste your YouTube OAuth client secret JSON in API settings.", "error")
        return redirect(url_for("dashboard"))

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            tmp.write(client_secret_json)
            tmp_path = tmp.name

        title = request.form.get("title") or session.get("generated_title") or "Grok Generated Video"
        description = request.form.get("description") or session.get("generated_description") or ""
        hashtags = request.form.get("hashtags") or session.get("generated_hashtags") or ""

        if hashtags:
            description = f"{description}\n\n{hashtags}".strip()

        token_file = str(BASE_DIR / f"youtube_token_{session['username']}.json")
        video_id = upload_video_to_youtube(
            client_secret_file=tmp_path,
            token_file=token_file,
            video_path=video_file_path,
            title=title,
            description=description,
            tags=[tag.lstrip("#") for tag in hashtags.split() if tag.startswith("#")],
        )
        session["youtube_video_id"] = video_id
        flash(f"Uploaded to YouTube successfully. Video ID: {video_id}", "success")
    except Exception as exc:
        flash(f"YouTube upload failed: {exc}", "error")
    finally:
        if "tmp_path" in locals() and Path(tmp_path).exists():
            Path(tmp_path).unlink()

    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
