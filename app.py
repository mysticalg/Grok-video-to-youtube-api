import os
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Callable

import requests
from PySide6.QtCore import QMimeData, QThread, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication, QIcon, QImage, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSlider,
    QSpinBox,
    QSplitter,
    QScrollArea,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

from social_uploaders import upload_video_to_facebook_page, upload_video_to_instagram_reels
from youtube_uploader import upload_video_to_youtube

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR = DOWNLOAD_DIR / ".thumbnails"
THUMBNAILS_DIR.mkdir(exist_ok=True)
CACHE_DIR = BASE_DIR / ".qtwebengine"
QTWEBENGINE_USE_DISK_CACHE = True
MIN_VALID_VIDEO_BYTES = 1 * 1024 * 1024
API_BASE_URL = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
DEFAULT_PREFERENCES_FILE = BASE_DIR / "preferences.json"
GITHUB_REPO_URL = "https://github.com/mysticalg/Grok-video-to-youtube-api"
GITHUB_RELEASES_URL = "https://github.com/mysticalg/Grok-video-to-youtube-api/releases"
GITHUB_ACTIONS_RUNS_URL = f"{GITHUB_REPO_URL}/actions"
BUY_ME_A_COFFEE_URL = "https://buymeacoffee.com/dhooksterm"
PAYPAL_DONATION_URL = "https://www.paypal.com/paypalme/dhookster"
SOL_DONATION_ADDRESS = "6HiqW3jeF3ymxjK5Fcm6dHi46gDuFmeCeSNdW99CfJjp"
DEFAULT_MANUAL_PROMPT_TEXT = (
    "abstract surreal artistic photorealistic strange random dream like scifi fast moving camera, "
    "fast moving fractals morphing and intersecting, highly detailed"
)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default




def _path_supports_rw(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_file = path / f".rw_probe_{os.getpid()}"
        moved_file = path / f".rw_probe_{os.getpid()}_moved"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.replace(moved_file)
        moved_file.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _resolve_qtwebengine_cache_dir() -> tuple[Path, bool]:
    candidates: list[Path] = []

    env_cache_dir = os.getenv("GROK_BROWSER_CACHE_DIR", "").strip()
    if env_cache_dir:
        candidates.append(Path(env_cache_dir).expanduser())

    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if local_app_data:
        candidates.append(Path(local_app_data) / "GrokVideoDesktopStudio" / "qtwebengine")

    xdg_cache_home = os.getenv("XDG_CACHE_HOME", "").strip()
    if xdg_cache_home:
        candidates.append(Path(xdg_cache_home) / "GrokVideoDesktopStudio" / "qtwebengine")

    candidates.append(Path.home() / "Library" / "Caches" / "GrokVideoDesktopStudio" / "qtwebengine")
    candidates.append(Path.home() / ".cache" / "GrokVideoDesktopStudio" / "qtwebengine")

    candidates.append(BASE_DIR / ".qtwebengine")
    candidates.append(Path(tempfile.gettempdir()) / "GrokVideoDesktopStudio" / "qtwebengine")

    for candidate in candidates:
        if _path_supports_rw(candidate):
            return candidate, True

    fallback = Path(tempfile.gettempdir()) / f"GrokVideoDesktopStudio_qtwebengine_{os.getpid()}"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback, False


def _configure_qtwebengine_runtime() -> None:
    global CACHE_DIR, QTWEBENGINE_USE_DISK_CACHE
    CACHE_DIR, QTWEBENGINE_USE_DISK_CACHE = _resolve_qtwebengine_cache_dir()

    default_flags = [
        "--enable-gpu-rasterization",
        "--enable-zero-copy",
        "--ignore-gpu-blocklist",
        "--disable-renderer-backgrounding",
        "--autoplay-policy=no-user-gesture-required",
        f"--media-cache-size={_env_int('GROK_BROWSER_MEDIA_CACHE_BYTES', 268435456)}",
        f"--disk-cache-size={_env_int('GROK_BROWSER_DISK_CACHE_BYTES', 536870912)}",
    ]
    if not QTWEBENGINE_USE_DISK_CACHE:
        default_flags.extend(["--disable-gpu-shader-disk-cache", "--disable-features=MediaHistoryLogging"])

    existing_flags = os.getenv("QTWEBENGINE_CHROMIUM_FLAGS", "").strip()
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = " ".join(default_flags + ([existing_flags] if existing_flags else []))


@dataclass
class GrokConfig:
    api_key: str
    chat_model: str
    image_model: str


@dataclass
class PromptConfig:
    source: str
    concept: str
    manual_prompt: str
    openai_api_key: str
    openai_chat_model: str
    video_resolution: str
    video_resolution_label: str
    video_aspect_ratio: str
    video_duration_seconds: int


@dataclass
class AISocialMetadata:
    title: str
    description: str
    hashtags: list[str]
    category: str


class GenerateWorker(QThread):
    finished_video = Signal(dict)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, config: GrokConfig, prompt_config: PromptConfig, count: int, download_dir: Path):
        super().__init__()
        self.config = config
        self.prompt_config = prompt_config
        self.count = count
        self.download_dir = download_dir
        self.stop_requested = False

    def request_stop(self) -> None:
        self.stop_requested = True
        self.requestInterruption()

    def _ensure_not_stopped(self) -> None:
        if self.stop_requested or self.isInterruptionRequested():
            raise RuntimeError("Generation stopped by user")

    def run(self) -> None:
        try:
            for idx in range(1, self.count + 1):
                self._ensure_not_stopped()
                self.status.emit(f"Generating variant {idx}/{self.count}...")
                video = self.generate_one_video(idx)
                self.finished_video.emit(video)
            self.status.emit("Generation complete.")
        except Exception as exc:
            if str(exc) == "Generation stopped by user":
                self.status.emit("Generation stopped.")
                return
            self.failed.emit(str(exc))

    def _api_error_message(self, response: requests.Response) -> str:
        try:
            return response.json().get("error", {}).get("message", response.text[:500])
        except Exception:
            return response.text[:500] or response.reason

    def call_grok_chat(self, system: str, user: str) -> str:
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        response = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": self.config.chat_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.9,
            },
            timeout=90,
        )
        if not response.ok:
            raise RuntimeError(f"Chat request failed: {response.status_code} {self._api_error_message(response)}")
        return response.json()["choices"][0]["message"]["content"].strip()

    def call_openai_chat(self, system: str, user: str) -> str:
        headers = {"Authorization": f"Bearer {self.prompt_config.openai_api_key}", "Content-Type": "application/json"}
        response = requests.post(
            f"{OPENAI_API_BASE}/chat/completions",
            headers=headers,
            json={
                "model": self.prompt_config.openai_chat_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.9,
            },
            timeout=90,
        )
        if not response.ok:
            raise RuntimeError(f"OpenAI chat request failed: {response.status_code} {self._api_error_message(response)}")
        return response.json()["choices"][0]["message"]["content"].strip()

    def build_prompt(self, variant: int) -> str:
        self._ensure_not_stopped()
        source = self.prompt_config.source
        if source == "manual":
            return self.prompt_config.manual_prompt

        system = "You write highly visual prompts for short cinematic AI videos."
        user = (
            "Create one polished video prompt for a "
            f"{self.prompt_config.video_duration_seconds} second scene in "
            f"{self.prompt_config.video_resolution_label} with a {self.prompt_config.video_aspect_ratio} aspect ratio "
            f"from this concept: {self.prompt_config.concept}. This is variant #{variant}."
        )

        return self.call_openai_chat(system, user) if source == "openai" else self.call_grok_chat(system, user)

    def start_video_job(self, prompt: str, resolution: str, duration_seconds: int) -> str:
        self._ensure_not_stopped()
        response = requests.post(
            f"{API_BASE_URL}/imagine/video/generations",
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.config.image_model,
                "prompt": prompt,
                "duration_seconds": duration_seconds,
                "resolution": resolution,
                "fps": 24,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("id")

    def poll_video_job(self, job_id: str, timeout_s: int = 420) -> dict:
        start = time.time()
        while time.time() - start < timeout_s:
            self._ensure_not_stopped()
            response = requests.get(
                f"{API_BASE_URL}/imagine/video/generations/{job_id}",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
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
        raise TimeoutError("Timed out waiting for video generation")

    def download_video(self, video_url: str, suffix: str) -> Path:
        self._ensure_not_stopped()
        self.download_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.download_dir / f"video_{int(time.time() * 1000)}_{suffix}.mp4"
        with requests.get(video_url, stream=True, timeout=240) as response:
            response.raise_for_status()
            with open(file_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    self._ensure_not_stopped()
                    if chunk:
                        handle.write(chunk)
        return file_path

    def generate_one_video(self, variant: int) -> dict:
        prompt = self.build_prompt(variant)

        try:
            video_job_id = self.start_video_job(
                prompt,
                self.prompt_config.video_resolution,
                int(self.prompt_config.video_duration_seconds),
            )
        except requests.HTTPError as exc:
            raise RuntimeError(
                "Could not start a video generation job with the selected resolution "
                f"({self.prompt_config.video_resolution_label})."
            ) from exc

        result = self.poll_video_job(video_job_id)
        video_url = result.get("output", {}).get("video_url") or result.get("video_url")
        if not video_url:
            raise RuntimeError("No video URL returned")

        file_path = self.download_video(video_url, f"v{variant}")
        return {
            "title": f"Generated Video {variant}",
            "prompt": prompt,
            "resolution": f"{self.prompt_config.video_resolution_label} ({self.prompt_config.video_aspect_ratio})",
            "video_file_path": str(file_path),
            "source_url": video_url,
        }

class StitchWorker(QThread):
    progress = Signal(int, str)
    status = Signal(str)
    finished_stitch = Signal(dict)
    failed = Signal(str, str)

    def __init__(
        self,
        window: "MainWindow",
        video_paths: list[Path],
        output_file: Path,
        stitched_base_file: Path,
        crossfade_enabled: bool,
        crossfade_duration: float,
        interpolate_enabled: bool,
        interpolation_fps: int,
        upscale_enabled: bool,
        upscale_target: str,
        use_gpu_encoding: bool,
        custom_music_file: Path | None,
        mute_original_audio: bool,
        original_audio_volume: float,
        music_volume: float,
        audio_fade_duration: float,
    ):
        super().__init__()
        self.window = window
        self.video_paths = video_paths
        self.output_file = output_file
        self.stitched_base_file = stitched_base_file
        self.crossfade_enabled = crossfade_enabled
        self.crossfade_duration = crossfade_duration
        self.interpolate_enabled = interpolate_enabled
        self.interpolation_fps = interpolation_fps
        self.upscale_enabled = upscale_enabled
        self.upscale_target = upscale_target
        self.use_gpu_encoding = use_gpu_encoding
        self.custom_music_file = custom_music_file
        self.mute_original_audio = mute_original_audio
        self.original_audio_volume = original_audio_volume
        self.music_volume = music_volume
        self.audio_fade_duration = audio_fade_duration

    def run(self) -> None:
        enhancement_enabled = self.interpolate_enabled or self.upscale_enabled
        try:
            stitch_target = self.stitched_base_file if enhancement_enabled else self.output_file

            if self.crossfade_enabled:
                self.status.emit(f"Stitching videos with {self.crossfade_duration:.1f}s crossfade transitions enabled.")
                self.window._stitch_videos_with_crossfade(
                    self.video_paths,
                    stitch_target,
                    crossfade_duration=self.crossfade_duration,
                    progress_callback=lambda p: self.progress.emit(max(5, int(5 + (p * 0.70))), "Stitching clips with crossfade..."),
                    use_gpu_encoding=self.use_gpu_encoding,
                )
            else:
                self.status.emit("Stitching videos with hard cuts (no crossfade).")
                self.window._stitch_videos_concat(
                    self.video_paths,
                    stitch_target,
                    progress_callback=lambda p: self.progress.emit(max(5, int(5 + (p * 0.70))), "Stitching clips with hard cuts..."),
                    use_gpu_encoding=self.use_gpu_encoding,
                )

            if enhancement_enabled:
                interpolation_status = f"{self.interpolation_fps} fps" if self.interpolate_enabled else "off"
                self.status.emit(
                    "Applying stitched video enhancements: "
                    f"frame interpolation={interpolation_status}, "
                    f"upscaling={self.upscale_target if self.upscale_enabled else 'off'}."
                )
                self.window._enhance_stitched_video(
                    input_file=stitch_target,
                    output_file=self.output_file,
                    interpolate=self.interpolate_enabled,
                    interpolation_fps=self.interpolation_fps,
                    upscale=self.upscale_enabled,
                    upscale_target=self.upscale_target,
                    progress_callback=lambda p: self.progress.emit(max(75, int(75 + (p * 0.25))), "Applying interpolation/upscaling..."),
                    use_gpu_encoding=self.use_gpu_encoding,
                )

            if self.custom_music_file is not None and self.custom_music_file.exists():
                self.status.emit(
                    "Adding custom music track with "
                    f"video audio {'muted' if self.mute_original_audio else f'at {self.original_audio_volume:.0f}%'} "
                    f"and fade in/out {self.audio_fade_duration:.1f}s."
                )
                self.window._apply_custom_music_track(
                    input_file=self.output_file,
                    output_file=self.output_file,
                    music_file=self.custom_music_file,
                    mute_original_audio=self.mute_original_audio,
                    original_audio_volume=self.original_audio_volume,
                    music_volume=self.music_volume,
                    fade_duration=self.audio_fade_duration,
                    progress_callback=lambda p: self.progress.emit(max(92, int(92 + (p * 0.08))), "Mixing custom music..."),
                    use_gpu_encoding=self.use_gpu_encoding,
                )

            self.progress.emit(100, "Finalizing stitched video...")
            self.finished_stitch.emit(
                {
                    "title": "Stitched Video",
                    "prompt": "stitched",
                    "resolution": "mixed",
                    "video_file_path": str(self.output_file),
                    "source_url": "local-stitch",
                }
            )
        except FileNotFoundError:
            self.failed.emit("ffmpeg Missing", "ffmpeg is required for stitching but was not found in PATH.")
        except subprocess.CalledProcessError as exc:
            self.failed.emit("Stitch Failed", exc.stderr[-800:] or "ffmpeg failed.")
        except RuntimeError as exc:
            self.failed.emit("Stitch Failed", str(exc))
        finally:
            if self.stitched_base_file.exists():
                self.stitched_base_file.unlink()



class FilteredWebEnginePage(QWebEnginePage):
    """Suppress noisy third-party console warnings from grok.com in the embedded browser."""

    _IGNORED_CONSOLE_PATTERNS = (
        "cdn-cgi/speculation",
        "react-i18next:: useTranslation",
        "Permissions-Policy header: Unrecognized feature: 'pointer-lock'",
        "violates the following Content Security Policy directive",
        "Play failed: [object DOMException]",
    )

    def __init__(self, on_console_message, profile: QWebEngineProfile | None = None, parent=None):
        if profile is not None:
            super().__init__(profile, parent)
        else:
            super().__init__(parent)
        self._on_console_message = on_console_message

    def javaScriptConsoleMessage(self, level, message, line_number, source_id):  # type: ignore[override]
        if any(pattern in message for pattern in self._IGNORED_CONSOLE_PATTERNS):
            return

        if self._on_console_message:
            self._on_console_message(f"Browser JS: {message} (source={source_id}:{line_number})")

        super().javaScriptConsoleMessage(level, message, line_number, source_id)


class MainWindow(QMainWindow):
    MANUAL_IMAGE_PICK_RETRY_LIMIT = 3
    MANUAL_IMAGE_SUBMIT_RETRY_LIMIT = 3

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grok Video Desktop Studio")
        self.resize(1500, 900)
        self.download_dir = DOWNLOAD_DIR
        self.videos: list[dict] = []
        self.worker: GenerateWorker | None = None
        self.stitch_worker: StitchWorker | None = None
        self._ffmpeg_nvenc_checked = False
        self._ffmpeg_nvenc_available = False
        self.preview_fullscreen_overlay_btn: QPushButton | None = None
        self.stop_all_requested = False
        self.manual_generation_queue: list[dict] = []
        self.manual_image_generation_queue: list[dict] = []
        self.pending_manual_variant_for_download: int | None = None
        self.pending_manual_download_type: str | None = None
        self.pending_manual_image_prompt: str | None = None
        self.manual_image_pick_clicked = False
        self.manual_image_video_submit_sent = False
        self.manual_image_pick_retry_count = 0
        self.manual_image_submit_retry_count = 0
        self.manual_image_submit_token = 0
        self.manual_download_deadline: float | None = None
        self.manual_download_click_sent = False
        self.manual_video_start_click_sent = False
        self.manual_video_make_click_fallback_used = False
        self.manual_video_allow_make_click = True
        self.manual_download_in_progress = False
        self.manual_download_started_at: float | None = None
        self.manual_download_poll_timer = QTimer(self)
        self.manual_download_poll_timer.setSingleShot(True)
        self.manual_download_poll_timer.timeout.connect(self._poll_for_manual_video)
        self.continue_from_frame_active = False
        self.continue_from_frame_target_count = 0
        self.continue_from_frame_completed = 0
        self.continue_from_frame_prompt = ""
        self.continue_from_frame_current_source_video = ""
        self.continue_from_frame_seed_image_path: Path | None = None
        self.continue_from_frame_waiting_for_reload = False
        self.continue_from_frame_reload_timeout_timer = QTimer(self)
        self.continue_from_frame_reload_timeout_timer.setSingleShot(True)
        self.continue_from_frame_reload_timeout_timer.timeout.connect(self._on_continue_reload_timeout)
        self.video_playback_hack_timer = QTimer(self)
        self.video_playback_hack_timer.setInterval(1800)
        self.video_playback_hack_timer.setSingleShot(True)
        self.video_playback_hack_timer.timeout.connect(self._ensure_browser_video_playback)
        self._playback_hack_success_logged = False
        self.last_extracted_frame_path: Path | None = None
        self.preview_muted = False
        self.preview_volume = 100
        self.custom_music_file: Path | None = None
        self.ai_social_metadata = AISocialMetadata(
            title="AI Generated Video",
            description="",
            hashtags=["grok", "ai", "generated-video"],
            category="22",
        )
        self._build_ui()
        self._load_startup_preferences()
        self._apply_default_theme()

    def _apply_default_theme(self) -> None:
        self.setStyleSheet("")

    def _build_ui(self) -> None:
        splitter = QSplitter()

        left = QWidget()
        left_layout = QVBoxLayout(left)

        self._build_model_api_settings_dialog()

        prompt_group = QGroupBox("✨ Prompt Inputs")
        prompt_group_layout = QVBoxLayout(prompt_group)

        prompt_group_layout.addWidget(QLabel("Concept"))
        self.concept = QPlainTextEdit()
        self.concept.setPlaceholderText("Describe the video idea...")
        self.concept.setMaximumHeight(90)
        prompt_group_layout.addWidget(self.concept)

        prompt_group_layout.addWidget(QLabel("Manual Prompt (used only when source is Manual)"))
        self.manual_prompt = QPlainTextEdit()
        self.manual_prompt.setPlaceholderText("Paste or write an exact prompt to skip prompt APIs...")
        self.manual_prompt.setPlainText(self.manual_prompt_default_input.toPlainText().strip() or DEFAULT_MANUAL_PROMPT_TEXT)
        self.manual_prompt.setMaximumHeight(110)
        prompt_group_layout.addWidget(self.manual_prompt)

        self.generate_prompt_btn = QPushButton("✨ Generate Prompt + Social Metadata from Concept")
        self.generate_prompt_btn.setToolTip(
            "Uses selected AI provider to convert Concept into a 10-second Grok Imagine prompt and social metadata."
        )
        self.generate_prompt_btn.clicked.connect(self.generate_prompt_from_concept)
        prompt_group_layout.addWidget(self.generate_prompt_btn)

        row = QHBoxLayout()
        row.addWidget(QLabel("Count"))
        self.count = QSpinBox()
        self.count.setRange(1, 10)
        self.count.setValue(1)
        row.addWidget(self.count)

        row.addWidget(QLabel("Resolution"))
        self.video_resolution = QComboBox()
        self.video_resolution.addItem("480p (854x480)", "854x480")
        self.video_resolution.addItem("720p (1280x720)", "1280x720")
        self.video_resolution.setCurrentIndex(1)
        row.addWidget(self.video_resolution)

        row.addWidget(QLabel("Duration"))
        self.video_duration = QComboBox()
        self.video_duration.addItem("6s", 6)
        self.video_duration.addItem("10s", 10)
        self.video_duration.setCurrentIndex(1)
        row.addWidget(self.video_duration)

        row.addWidget(QLabel("Aspect"))
        self.video_aspect_ratio = QComboBox()
        self.video_aspect_ratio.addItem("2:3", "2:3")
        self.video_aspect_ratio.addItem("3:2", "3:2")
        self.video_aspect_ratio.addItem("1:1", "1:1")
        self.video_aspect_ratio.addItem("9:16", "9:16")
        self.video_aspect_ratio.addItem("16:9", "16:9")
        self.video_aspect_ratio.setCurrentIndex(4)
        row.addWidget(self.video_aspect_ratio)
        prompt_group_layout.addLayout(row)

        left_layout.addWidget(prompt_group)

        actions_group = QGroupBox("🚀 Actions")
        actions_layout = QGridLayout(actions_group)

        self.generate_btn = QPushButton("🎬 Generate Video")
        self.generate_btn.setToolTip("Generate videos from the selected prompt source.")
        self.generate_btn.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: 700;"
            "border: 1px solid #1b5e20; border-radius: 6px; padding: 8px;"
        )
        self.generate_btn.clicked.connect(self.start_generation)
        actions_layout.addWidget(self.generate_btn, 0, 0)

        self.generate_image_btn = QPushButton("🖼️ Populate Image Prompt in Browser")
        self.generate_image_btn.setToolTip("Build and paste an image prompt into the Grok browser tab.")
        self.generate_image_btn.setStyleSheet(
            "background-color: #43a047; color: white; font-weight: 700;"
            "border: 1px solid #2e7d32; border-radius: 6px; padding: 8px;"
        )
        self.generate_image_btn.clicked.connect(self.start_image_generation)
        actions_layout.addWidget(self.generate_image_btn, 0, 1)

        self.stop_all_btn = QPushButton("🛑 Stop All Jobs")
        self.stop_all_btn.setToolTip("Stop active generation jobs after current requests complete.")
        self.stop_all_btn.setStyleSheet(
            "background-color: #8b0000; color: white; font-weight: 700;"
            "border: 1px solid #5c0000; border-radius: 6px; padding: 8px;"
        )
        self.stop_all_btn.clicked.connect(self.stop_all_jobs)
        actions_layout.addWidget(self.stop_all_btn, 1, 0)

        self.continue_frame_btn = QPushButton("🟨 Continue from Last Frame (paste + generate)")
        self.continue_frame_btn.setToolTip("Use the last generated video's final frame and continue from it.")
        self.continue_frame_btn.setStyleSheet(
            "background-color: #fdd835; color: #222; font-weight: 700;"
            "border: 1px solid #f9a825; border-radius: 6px; padding: 8px;"
        )
        self.continue_frame_btn.clicked.connect(self.continue_from_last_frame)
        actions_layout.addWidget(self.continue_frame_btn, 2, 0)

        self.continue_image_btn = QPushButton("🖼️ Continue from Local Image (paste + generate)")
        self.continue_image_btn.setToolTip("Choose a local image and continue generation from that frame.")
        self.continue_image_btn.setStyleSheet(
            "background-color: #fff176; color: #222; font-weight: 700;"
            "border: 1px solid #fbc02d; border-radius: 6px; padding: 8px;"
        )
        self.continue_image_btn.clicked.connect(self.continue_from_local_image)
        actions_layout.addWidget(self.continue_image_btn, 2, 1)

        self.show_browser_btn = QPushButton("🌐 Show Browser (grok.com/imagine)")
        self.show_browser_btn.setToolTip("Bring the embedded Grok browser to the front.")
        self.show_browser_btn.setStyleSheet(
            "background-color: #ffffff; color: #222; font-weight: 700;"
            "border: 1px solid #cfcfcf; border-radius: 6px; padding: 8px;"
        )
        self.show_browser_btn.clicked.connect(self.show_browser_page)
        actions_layout.addWidget(self.show_browser_btn, 3, 0)

        self.stitch_btn = QPushButton("🧵 Stitch All Videos")
        self.stitch_btn.setToolTip("Combine all downloaded videos into one stitched output file.")
        self.stitch_btn.setStyleSheet(
            "background-color: #81d4fa; color: #0d47a1; font-weight: 700;"
            "border: 1px solid #4fc3f7; border-radius: 6px; padding: 8px;"
        )
        self.stitch_btn.clicked.connect(self.stitch_all_videos)
        actions_layout.addWidget(self.stitch_btn, 3, 1)

        self.stitch_crossfade_checkbox = QCheckBox("Enable 0.5s crossfade between clips")
        self.stitch_crossfade_checkbox.setToolTip("Blend each clip transition using a 0.5 second crossfade.")
        actions_layout.addWidget(self.stitch_crossfade_checkbox, 4, 0, 1, 1)

        self.stitch_interpolation_checkbox = QCheckBox("Enable frame interpolation")
        self.stitch_interpolation_checkbox.setToolTip(
            "After stitching, use ffmpeg minterpolate to smooth motion by generating in-between frames."
        )
        actions_layout.addWidget(self.stitch_interpolation_checkbox, 5, 0, 1, 1)

        self.stitch_interpolation_fps = QComboBox()
        self.stitch_interpolation_fps.addItem("48 fps", 48)
        self.stitch_interpolation_fps.addItem("60 fps", 60)
        self.stitch_interpolation_fps.setCurrentIndex(0)
        self.stitch_interpolation_fps.setToolTip("Target frame rate used when frame interpolation is enabled.")
        actions_layout.addWidget(self.stitch_interpolation_fps, 5, 1, 1, 1)

        self.stitch_upscale_checkbox = QCheckBox("Enable AI-style upscaling")
        self.stitch_upscale_checkbox.setToolTip(
            "After stitching, upscale output to a selected target resolution using high-quality Lanczos scaling."
        )
        actions_layout.addWidget(self.stitch_upscale_checkbox, 6, 0, 1, 1)

        self.stitch_upscale_target = QComboBox()
        self.stitch_upscale_target.addItem("2x (max 4K)", "2x")
        self.stitch_upscale_target.addItem("1080p (1920x1080)", "1080p")
        self.stitch_upscale_target.addItem("1440p (2560x1440)", "1440p")
        self.stitch_upscale_target.addItem("4K (3840x2160)", "4k")
        self.stitch_upscale_target.setCurrentIndex(0)
        self.stitch_upscale_target.setToolTip("Choose output upscale target resolution.")
        actions_layout.addWidget(self.stitch_upscale_target, 6, 1, 1, 1)

        self.stitch_gpu_checkbox = QCheckBox("Use GPU encoding for stitching (NVENC)")
        self.stitch_gpu_checkbox.setToolTip("Use NVIDIA NVENC encoder when available to reduce CPU load.")
        self.stitch_gpu_checkbox.setChecked(True)
        self.stitch_gpu_checkbox.toggled.connect(lambda _: self._sync_video_options_label())
        actions_layout.addWidget(self.stitch_gpu_checkbox, 7, 0, 1, 2)

        self.video_options_dropdown = QComboBox()
        self.video_options_dropdown.addItem("0.2s", 0.2)
        self.video_options_dropdown.addItem("0.3s", 0.3)
        self.video_options_dropdown.addItem("0.5s", 0.5)
        self.video_options_dropdown.addItem("0.8s", 0.8)
        self.video_options_dropdown.addItem("1.0s", 1.0)
        self.video_options_dropdown.addItem("Advanced...", None)
        self.video_options_dropdown.setCurrentIndex(2)
        self.video_options_dropdown.setMaximumWidth(140)
        self.video_options_dropdown.setToolTip("Crossfade duration for stitching.")
        self.video_options_dropdown.currentIndexChanged.connect(self._on_video_options_selected)
        actions_layout.addWidget(self.video_options_dropdown, 4, 1, 1, 1, alignment=Qt.AlignRight)

        self.music_file_label = QLabel("Music: none selected")
        self.music_file_label.setStyleSheet("color: #9fb3c8;")
        self.music_file_label.setWordWrap(True)
        actions_layout.addWidget(self.music_file_label, 8, 0, 1, 2)

        music_actions_layout = QHBoxLayout()
        self.choose_music_btn = QPushButton("🎵 Choose Music (wav/mp3)")
        self.choose_music_btn.setToolTip("Select a local WAV or MP3 file to mix under the stitched video.")
        self.choose_music_btn.clicked.connect(self._choose_custom_music_file)
        music_actions_layout.addWidget(self.choose_music_btn)

        self.clear_music_btn = QPushButton("Clear Music")
        self.clear_music_btn.setToolTip("Remove any selected custom background music file.")
        self.clear_music_btn.clicked.connect(self._clear_custom_music_file)
        music_actions_layout.addWidget(self.clear_music_btn)
        actions_layout.addLayout(music_actions_layout, 9, 0, 1, 2)

        self.stitch_mute_original_checkbox = QCheckBox("Mute original video audio when music is used")
        self.stitch_mute_original_checkbox.setToolTip("If enabled, only the selected music is audible in the stitched output.")
        actions_layout.addWidget(self.stitch_mute_original_checkbox, 10, 0, 1, 2)

        self.stitch_original_audio_volume = QSpinBox()
        self.stitch_original_audio_volume.setRange(0, 200)
        self.stitch_original_audio_volume.setValue(100)
        self.stitch_original_audio_volume.setPrefix("Original audio: ")
        self.stitch_original_audio_volume.setSuffix("%")
        self.stitch_original_audio_volume.setToolTip("Original video audio level used during custom music mixing.")
        actions_layout.addWidget(self.stitch_original_audio_volume, 11, 0)

        self.stitch_music_volume = QSpinBox()
        self.stitch_music_volume.setRange(0, 200)
        self.stitch_music_volume.setValue(100)
        self.stitch_music_volume.setPrefix("Music audio: ")
        self.stitch_music_volume.setSuffix("%")
        self.stitch_music_volume.setToolTip("Custom music level used during stitched output mixing.")
        actions_layout.addWidget(self.stitch_music_volume, 11, 1)

        self.stitch_audio_fade_duration = QDoubleSpinBox()
        self.stitch_audio_fade_duration.setRange(0.0, 10.0)
        self.stitch_audio_fade_duration.setSingleStep(0.1)
        self.stitch_audio_fade_duration.setDecimals(1)
        self.stitch_audio_fade_duration.setValue(0.5)
        self.stitch_audio_fade_duration.setSuffix(" s")
        self.stitch_audio_fade_duration.setToolTip("Fade-in and fade-out duration applied to stitched output audio mix.")
        actions_layout.addWidget(self.stitch_audio_fade_duration, 12, 0)

        self.stitch_audio_fade_label = QLabel("Audio fade in/out")
        self.stitch_audio_fade_label.setStyleSheet("color: #9fb3c8;")
        actions_layout.addWidget(self.stitch_audio_fade_label, 12, 1)

        upload_group = QGroupBox("Upload")
        upload_layout = QHBoxLayout(upload_group)

        self.upload_youtube_btn = QPushButton("YouTube")
        self.upload_youtube_btn.setToolTip("Upload the currently selected local video to your YouTube channel.")
        self.upload_youtube_btn.setStyleSheet(
            "background-color: #cc0000; color: white; font-weight: 700;"
            "border: 1px solid #990000; border-radius: 6px; padding: 5px 10px;"
        )
        self.upload_youtube_btn.clicked.connect(self.upload_selected_to_youtube)
        upload_layout.addWidget(self.upload_youtube_btn)

        self.upload_facebook_btn = QPushButton("Facebook")
        self.upload_facebook_btn.setToolTip("Upload the selected local video to your Facebook Page as an unpublished video.")
        self.upload_facebook_btn.setStyleSheet(
            "background-color: #1877F2; color: white; font-weight: 700;"
            "border: 1px solid #115bcc; border-radius: 6px; padding: 5px 10px;"
        )
        self.upload_facebook_btn.clicked.connect(self.upload_selected_to_facebook)
        upload_layout.addWidget(self.upload_facebook_btn)

        self.upload_instagram_btn = QPushButton("Instagram")
        self.upload_instagram_btn.setToolTip("Publish selected video to Instagram Reels using Meta Graph API (requires a public source URL).")
        self.upload_instagram_btn.setStyleSheet(
            "background-color: #8a3ab9; color: white; font-weight: 700;"
            "border: 1px solid #6d2f94; border-radius: 6px; padding: 5px 10px;"
        )
        self.upload_instagram_btn.clicked.connect(self.upload_selected_to_instagram)
        upload_layout.addWidget(self.upload_instagram_btn)

        actions_layout.addWidget(upload_group, 13, 0, 1, 2)

        self.buy_coffee_btn = QPushButton("☕ Buy Me a Coffee")
        self.buy_coffee_btn.setToolTip("If this saves you hours, grab me a ☕")
        self.buy_coffee_btn.setStyleSheet(
            "font-size: 15px; font-weight: 700; padding: 10px;"
            "background-color: #ffdd00; color: #222; border-radius: 8px;"
        )
        self.buy_coffee_btn.clicked.connect(self.open_buy_me_a_coffee)
        actions_layout.addWidget(self.buy_coffee_btn, 14, 0, 1, 2)

        left_layout.addWidget(actions_group)

        left_layout.addWidget(QLabel("Generated Videos"))
        self.video_picker = QComboBox()
        self.video_picker.setIconSize(QPixmap(144, 82).size())
        self.video_picker.setMinimumHeight(42)
        self.video_picker.currentIndexChanged.connect(self.show_selected_video)
        left_layout.addWidget(self.video_picker)

        video_list_controls = QHBoxLayout()
        self.open_video_btn = QPushButton("📂 Open Video")
        self.open_video_btn.setToolTip("Open a local video file and add it to Generated Videos.")
        self.open_video_btn.clicked.connect(self.open_local_video)
        video_list_controls.addWidget(self.open_video_btn)

        self.video_move_up_btn = QPushButton("⬆ Move Up")
        self.video_move_up_btn.setToolTip("Move selected video earlier in the Generated Videos order.")
        self.video_move_up_btn.clicked.connect(lambda: self.move_selected_video(-1))
        video_list_controls.addWidget(self.video_move_up_btn)

        self.video_move_down_btn = QPushButton("⬇ Move Down")
        self.video_move_down_btn.setToolTip("Move selected video later in the Generated Videos order.")
        self.video_move_down_btn.clicked.connect(lambda: self.move_selected_video(1))
        video_list_controls.addWidget(self.video_move_down_btn)

        self.video_remove_btn = QPushButton("🗑 Remove")
        self.video_remove_btn.setToolTip("Remove selected video from Generated Videos list.")
        self.video_remove_btn.clicked.connect(self.remove_selected_video)
        video_list_controls.addWidget(self.video_remove_btn)

        left_layout.addLayout(video_list_controls)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.preview = QVideoWidget()
        self.player.setVideoOutput(self.preview)

        self.browser = QWebEngineView()
        if QTWEBENGINE_USE_DISK_CACHE:
            self.browser_profile = QWebEngineProfile("grok-video-desktop-profile", self)
        else:
            # Off-the-record profile avoids startup cache/quota errors on locked folders (common on synced drives).
            self.browser_profile = QWebEngineProfile(self)

        self.browser.setPage(FilteredWebEnginePage(self._append_log, self.browser_profile, self.browser))
        browser_profile = self.browser_profile
        if QTWEBENGINE_USE_DISK_CACHE:
            (CACHE_DIR / "profile").mkdir(parents=True, exist_ok=True)
            (CACHE_DIR / "cache").mkdir(parents=True, exist_ok=True)
            browser_profile.setPersistentStoragePath(str(CACHE_DIR / "profile"))
            browser_profile.setCachePath(str(CACHE_DIR / "cache"))
            browser_profile.setPersistentCookiesPolicy(QWebEngineProfile.ForcePersistentCookies)
            browser_profile.setHttpCacheType(QWebEngineProfile.DiskHttpCache)
            browser_profile.setHttpCacheMaximumSize(_env_int("GROK_BROWSER_DISK_CACHE_BYTES", 536870912))
        else:
            browser_profile.setPersistentCookiesPolicy(QWebEngineProfile.NoPersistentCookies)
            browser_profile.setHttpCacheType(QWebEngineProfile.MemoryHttpCache)

        browser_settings = self.browser.settings()
        browser_settings.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
        browser_settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        browser_settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        browser_settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)

        self.browser.setUrl(QUrl("https://grok.com/imagine"))
        self.browser.loadFinished.connect(self._on_browser_load_finished)
        self.browser_profile.downloadRequested.connect(self._on_browser_download_requested)

        log_group = QGroupBox("📡 Activity Log")
        log_layout = QVBoxLayout(log_group)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(260)
        log_layout.addWidget(self.log)

        self.stitch_progress_label = QLabel("Stitch progress: idle")
        self.stitch_progress_label.setStyleSheet("color: #9fb3c8;")
        log_layout.addWidget(self.stitch_progress_label)

        self.stitch_progress_bar = QProgressBar()
        self.stitch_progress_bar.setRange(0, 100)
        self.stitch_progress_bar.setValue(0)
        self.stitch_progress_bar.setVisible(False)
        log_layout.addWidget(self.stitch_progress_bar)

        if QTWEBENGINE_USE_DISK_CACHE:
            self._append_log(f"Browser cache path: {CACHE_DIR}")
        else:
            self._append_log(
                "Browser cache: running in memory/off-the-record mode because no writable cache folder was available."
            )

        preview_group = QGroupBox("🎞️ Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.addWidget(self.preview)

        preview_controls = QHBoxLayout()
        self.preview_play_btn = QPushButton("▶️ Play")
        self.preview_play_btn.setToolTip("Play the selected video in the preview pane.")
        self.preview_play_btn.clicked.connect(self.play_preview)
        preview_controls.addWidget(self.preview_play_btn)
        self.preview_stop_btn = QPushButton("⏹️ Stop")
        self.preview_stop_btn.setToolTip("Stop playback in the preview pane.")
        self.preview_stop_btn.setStyleSheet(
            "background-color: #8b0000; color: white; font-weight: 700;"
            "border: 1px solid #5c0000; border-radius: 6px; padding: 6px 10px;"
        )
        self.preview_stop_btn.clicked.connect(self.stop_preview)
        preview_controls.addWidget(self.preview_stop_btn)

        self.preview_mute_checkbox = QCheckBox("Mute")
        self.preview_mute_checkbox.toggled.connect(self._set_preview_muted)
        preview_controls.addWidget(self.preview_mute_checkbox)

        self.preview_volume_label = QLabel("Volume")
        preview_controls.addWidget(self.preview_volume_label)

        self.preview_volume_slider = QSpinBox()
        self.preview_volume_slider.setRange(0, 100)
        self.preview_volume_slider.setValue(self.preview_volume)
        self.preview_volume_slider.setSuffix("%")
        self.preview_volume_slider.valueChanged.connect(self._set_preview_volume)
        preview_controls.addWidget(self.preview_volume_slider)

        self.preview_fullscreen_btn = QPushButton("⛶ Fullscreen")
        self.preview_fullscreen_btn.setToolTip("Toggle fullscreen preview.")
        self.preview_fullscreen_btn.clicked.connect(self.toggle_preview_fullscreen)
        self.preview.fullScreenChanged.connect(self._on_preview_fullscreen_changed)
        preview_controls.addWidget(self.preview_fullscreen_btn)
        preview_layout.addLayout(preview_controls)

        timeline_layout = QHBoxLayout()
        self.preview_position_label = QLabel("00:00 / 00:00")
        timeline_layout.addWidget(self.preview_position_label)

        self.preview_seek_slider = QSlider(Qt.Horizontal)
        self.preview_seek_slider.setRange(0, 0)
        self.preview_seek_slider.sliderMoved.connect(self.seek_preview)
        timeline_layout.addWidget(self.preview_seek_slider)
        preview_layout.addLayout(timeline_layout)

        self.audio_output.setMuted(self.preview_muted)
        self.audio_output.setVolume(self.preview_volume / 100)
        self.player.positionChanged.connect(self._on_preview_position_changed)
        self.player.durationChanged.connect(self._on_preview_duration_changed)

        bottom_splitter = QSplitter()
        bottom_splitter.addWidget(preview_group)
        bottom_splitter.addWidget(log_group)
        bottom_splitter.setSizes([500, 800])

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(self.browser)
        right_splitter.addWidget(bottom_splitter)
        right_splitter.setSizes([620, 280])

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_splitter)
        splitter.setSizes([760, 1140])

        # Keep browser visible as a fixed right-hand pane
        splitter.setChildrenCollapsible(False)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

        self._build_menu_bar()
        self._toggle_prompt_source_fields()
        self._sync_video_options_label()

    def _build_model_api_settings_dialog(self) -> None:
        self.model_api_settings_dialog = QDialog(self)
        self.model_api_settings_dialog.setWindowTitle("Model/API Settings")
        dialog_layout = QVBoxLayout(self.model_api_settings_dialog)
        form_layout = QFormLayout()

        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setText(os.getenv("GROK_API_KEY", ""))
        form_layout.addRow("Grok API Key", self.api_key)

        self.chat_model = QLineEdit(os.getenv("GROK_CHAT_MODEL", "grok-3-mini"))
        form_layout.addRow("Chat Model", self.chat_model)

        self.image_model = QLineEdit(os.getenv("GROK_VIDEO_MODEL", "grok-video-latest"))
        form_layout.addRow("Video Model", self.image_model)

        self.prompt_source = QComboBox()
        self.prompt_source.addItem("Manual prompt (no API)", "manual")
        self.prompt_source.addItem("Grok API", "grok")
        self.prompt_source.addItem("OpenAI API", "openai")
        self.prompt_source.currentIndexChanged.connect(self._toggle_prompt_source_fields)
        form_layout.addRow("Prompt Source", self.prompt_source)

        self.openai_api_key = QLineEdit()
        self.openai_api_key.setEchoMode(QLineEdit.Password)
        self.openai_api_key.setText(os.getenv("OPENAI_API_KEY", ""))
        form_layout.addRow("OpenAI API Key", self.openai_api_key)

        self.openai_chat_model = QLineEdit(os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
        form_layout.addRow("OpenAI Chat Model", self.openai_chat_model)

        self.ai_auth_method = QComboBox()
        self.ai_auth_method.addItem("API key", "api_key")
        self.ai_auth_method.addItem("Browser sign-in (preferred)", "browser")
        form_layout.addRow("AI Authorization", self.ai_auth_method)

        self.browser_auth_btn = QPushButton("Open Provider Login in Browser")
        self.browser_auth_btn.clicked.connect(self.open_ai_provider_login)
        form_layout.addRow("Browser Authorization", self.browser_auth_btn)

        self.youtube_api_key = QLineEdit()
        self.youtube_api_key.setEchoMode(QLineEdit.Password)
        self.youtube_api_key.setText(os.getenv("YOUTUBE_API_KEY", ""))
        form_layout.addRow("YouTube API Key", self.youtube_api_key)

        self.facebook_page_id = QLineEdit(os.getenv("FACEBOOK_PAGE_ID", ""))
        form_layout.addRow("Facebook Page ID", self.facebook_page_id)

        self.facebook_access_token = QLineEdit()
        self.facebook_access_token.setEchoMode(QLineEdit.Password)
        self.facebook_access_token.setText(os.getenv("FACEBOOK_ACCESS_TOKEN", ""))
        form_layout.addRow("Facebook Access Token", self.facebook_access_token)

        self.instagram_business_id = QLineEdit(os.getenv("INSTAGRAM_BUSINESS_ID", ""))
        form_layout.addRow("Instagram Business ID", self.instagram_business_id)

        self.instagram_access_token = QLineEdit()
        self.instagram_access_token.setEchoMode(QLineEdit.Password)
        self.instagram_access_token.setText(os.getenv("INSTAGRAM_ACCESS_TOKEN", ""))
        form_layout.addRow("Instagram Access Token", self.instagram_access_token)

        self.download_path_input = QLineEdit(str(self.download_dir))
        self.download_path_input.setReadOnly(True)
        choose_download_path_btn = QPushButton("Browse...")
        choose_download_path_btn.clicked.connect(self._choose_download_path)
        download_path_row = QHBoxLayout()
        download_path_row.addWidget(self.download_path_input)
        download_path_row.addWidget(choose_download_path_btn)
        form_layout.addRow("Download Folder", download_path_row)

        self.crossfade_duration = QDoubleSpinBox()
        self.crossfade_duration.setRange(0.1, 3.0)
        self.crossfade_duration.setSingleStep(0.1)
        self.crossfade_duration.setDecimals(1)
        self.crossfade_duration.setValue(0.5)
        self.crossfade_duration.setSuffix(" s")
        self.crossfade_duration.valueChanged.connect(self._sync_video_options_label)
        form_layout.addRow("Crossfade Duration", self.crossfade_duration)

        self.manual_prompt_default_input = QPlainTextEdit()
        self.manual_prompt_default_input.setMaximumHeight(90)
        self.manual_prompt_default_input.setPlaceholderText("Default text used to prefill Manual Prompt.")
        self.manual_prompt_default_input.setPlainText(DEFAULT_MANUAL_PROMPT_TEXT)
        form_layout.addRow("Default Manual Prompt", self.manual_prompt_default_input)

        dialog_layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        save_btn = button_box.button(QDialogButtonBox.StandardButton.Save)
        if save_btn is not None:
            save_btn.setText("Save Settings")
            save_btn.clicked.connect(self.save_model_api_settings)
        close_btn = button_box.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.model_api_settings_dialog.close)
        dialog_layout.addWidget(button_box)

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        save_action = QAction("Save Preferences...", self)
        save_action.triggered.connect(self.save_preferences)
        file_menu.addAction(save_action)

        load_action = QAction("Load Preferences...", self)
        load_action.triggered.connect(self.load_preferences)
        file_menu.addAction(load_action)

        open_video_action = QAction("Open Video...", self)
        open_video_action.triggered.connect(self.open_local_video)
        file_menu.addAction(open_video_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        settings_menu = menu_bar.addMenu("Model/API Settings")
        open_settings_action = QAction("Open Model/API Settings", self)
        open_settings_action.triggered.connect(self.show_model_api_settings)
        settings_menu.addAction(open_settings_action)

        help_menu = menu_bar.addMenu("Help")
        info_action = QAction("Info", self)
        info_action.triggered.connect(self.show_app_info)
        help_menu.addAction(info_action)

        github_action = QAction("GitHub", self)
        github_action.triggered.connect(self.open_github_page)
        help_menu.addAction(github_action)

        releases_action = QAction("Download Windows Binary", self)
        releases_action.triggered.connect(self.open_github_releases_page)
        help_menu.addAction(releases_action)

        actions_action = QAction("Build Artifacts", self)
        actions_action.triggered.connect(self.open_github_actions_runs_page)
        help_menu.addAction(actions_action)

    def show_app_info(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("App Info")
        dialog.setMinimumWidth(680)

        info_browser = QTextBrowser(dialog)
        info_browser.setOpenExternalLinks(True)
        info_browser.setReadOnly(True)
        info_browser.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        info_browser.setHtml(
            "<h3>Grok Video Desktop Studio</h3>"
            "<p><b>Version:</b> 1.0.0<br>"
            "<b>Authors:</b> Grok Video Desktop Studio contributors<br>"
            "<b>Desktop workflow:</b> PyQt + embedded Grok browser + YouTube uploader</p>"
            "<p><b>Downloads</b><br>"
            f"- Releases: <a href='{GITHUB_RELEASES_URL}'>{GITHUB_RELEASES_URL}</a><br>"
            f"- CI workflow artifacts: <a href='{GITHUB_ACTIONS_RUNS_URL}'>{GITHUB_ACTIONS_RUNS_URL}</a></p>"
            "<p>If this saves you hours, grab me a ☕.</p>"
            "<p><b>Support links</b><br>"
            f"- Buy Me a Coffee: <a href='{BUY_ME_A_COFFEE_URL}'>{BUY_ME_A_COFFEE_URL}</a><br>"
            f"- PayPal: <a href='{PAYPAL_DONATION_URL}'>{PAYPAL_DONATION_URL}</a><br>"
            f"- Crypto (SOL): <code>{SOL_DONATION_ADDRESS}</code></p>"
        )

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        button_box.button(QDialogButtonBox.StandardButton.Close).clicked.connect(dialog.close)

        layout = QVBoxLayout(dialog)
        layout.addWidget(info_browser)
        layout.addWidget(button_box)

        dialog.exec()

    def open_github_page(self) -> None:
        QDesktopServices.openUrl(QUrl(GITHUB_REPO_URL))

    def open_github_releases_page(self) -> None:
        QDesktopServices.openUrl(QUrl(GITHUB_RELEASES_URL))

    def open_github_actions_runs_page(self) -> None:
        QDesktopServices.openUrl(QUrl(GITHUB_ACTIONS_RUNS_URL))

    def open_buy_me_a_coffee(self) -> None:
        QDesktopServices.openUrl(QUrl(BUY_ME_A_COFFEE_URL))

    def show_model_api_settings(self) -> None:
        self.model_api_settings_dialog.show()
        self.model_api_settings_dialog.raise_()
        self.model_api_settings_dialog.activateWindow()

    def _choose_custom_music_file(self) -> None:
        music_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select background music",
            str(self.download_dir),
            "Audio Files (*.mp3 *.wav)",
        )
        if not music_path:
            return

        selected_path = Path(music_path)
        if not selected_path.exists():
            QMessageBox.warning(self, "Music Missing", "Selected music file was not found on disk.")
            return

        self.custom_music_file = selected_path
        self.music_file_label.setText(f"Music: {selected_path.name}")
        self._append_log(f"Selected custom stitch music: {selected_path}")

    def _clear_custom_music_file(self) -> None:
        self.custom_music_file = None
        self.music_file_label.setText("Music: none selected")
        self._append_log("Cleared custom stitch music selection.")

    def _choose_download_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose Download Folder", str(self.download_dir))
        if not path:
            return
        self.download_dir = Path(path)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.download_path_input.setText(str(self.download_dir))
        self._append_log(f"Download folder set to: {self.download_dir}")

    def _on_video_options_selected(self, index: int) -> None:
        option_value = self.video_options_dropdown.itemData(index)
        if option_value is None:
            self.show_model_api_settings()
            self._sync_video_options_label()
            return

        try:
            self.crossfade_duration.setValue(float(option_value))
            self._sync_video_options_label()
        except (TypeError, ValueError):
            pass

    def _sync_video_options_label(self) -> None:
        duration = self.crossfade_duration.value()
        target_index = self.video_options_dropdown.findData(duration)
        self.video_options_dropdown.blockSignals(True)
        if target_index >= 0:
            self.video_options_dropdown.setCurrentIndex(target_index)
        else:
            self.video_options_dropdown.setCurrentIndex(2)
        self.video_options_dropdown.blockSignals(False)

    def _collect_preferences(self) -> dict:
        return {
            "api_key": self.api_key.text(),
            "chat_model": self.chat_model.text(),
            "image_model": self.image_model.text(),
            "prompt_source": self.prompt_source.currentData(),
            "openai_api_key": self.openai_api_key.text(),
            "openai_chat_model": self.openai_chat_model.text(),
            "ai_auth_method": self.ai_auth_method.currentData(),
            "youtube_api_key": self.youtube_api_key.text(),
            "facebook_page_id": self.facebook_page_id.text(),
            "facebook_access_token": self.facebook_access_token.text(),
            "instagram_business_id": self.instagram_business_id.text(),
            "instagram_access_token": self.instagram_access_token.text(),
            "concept": self.concept.toPlainText(),
            "manual_prompt": self.manual_prompt.toPlainText(),
            "manual_prompt_default": self.manual_prompt_default_input.toPlainText(),
            "count": self.count.value(),
            "video_resolution": str(self.video_resolution.currentData()),
            "video_duration_seconds": int(self.video_duration.currentData()),
            "video_aspect_ratio": str(self.video_aspect_ratio.currentData()),
            "stitch_crossfade_enabled": self.stitch_crossfade_checkbox.isChecked(),
            "stitch_interpolation_enabled": self.stitch_interpolation_checkbox.isChecked(),
            "stitch_interpolation_fps": int(self.stitch_interpolation_fps.currentData()),
            "stitch_upscale_enabled": self.stitch_upscale_checkbox.isChecked(),
            "stitch_upscale_target": str(self.stitch_upscale_target.currentData()),
            "stitch_gpu_enabled": self.stitch_gpu_checkbox.isChecked(),
            "stitch_mute_original_audio": self.stitch_mute_original_checkbox.isChecked(),
            "stitch_original_audio_volume": self.stitch_original_audio_volume.value(),
            "stitch_music_volume": self.stitch_music_volume.value(),
            "stitch_audio_fade_duration": self.stitch_audio_fade_duration.value(),
            "stitch_custom_music_file": str(self.custom_music_file) if self.custom_music_file else "",
            "crossfade_duration": self.crossfade_duration.value(),
            "download_dir": str(self.download_dir),
            "preview_muted": self.preview_mute_checkbox.isChecked(),
            "preview_volume": self.preview_volume_slider.value(),
            "ai_social_metadata": {
                "title": self.ai_social_metadata.title,
                "description": self.ai_social_metadata.description,
                "hashtags": self.ai_social_metadata.hashtags,
                "category": self.ai_social_metadata.category,
            },
        }

    def _apply_preferences(self, preferences: dict) -> None:
        if not isinstance(preferences, dict):
            raise ValueError("Preferences file must contain a JSON object.")

        if "api_key" in preferences:
            self.api_key.setText(str(preferences["api_key"]))
        if "chat_model" in preferences:
            self.chat_model.setText(str(preferences["chat_model"]))
        if "image_model" in preferences:
            self.image_model.setText(str(preferences["image_model"]))
        if "prompt_source" in preferences:
            source_index = self.prompt_source.findData(str(preferences["prompt_source"]))
            if source_index >= 0:
                self.prompt_source.setCurrentIndex(source_index)
        if "openai_api_key" in preferences:
            self.openai_api_key.setText(str(preferences["openai_api_key"]))
        if "openai_chat_model" in preferences:
            self.openai_chat_model.setText(str(preferences["openai_chat_model"]))
        if "ai_auth_method" in preferences:
            auth_index = self.ai_auth_method.findData(str(preferences["ai_auth_method"]))
            if auth_index >= 0:
                self.ai_auth_method.setCurrentIndex(auth_index)
        if "youtube_api_key" in preferences:
            self.youtube_api_key.setText(str(preferences["youtube_api_key"]))
        if "facebook_page_id" in preferences:
            self.facebook_page_id.setText(str(preferences["facebook_page_id"]))
        if "facebook_access_token" in preferences:
            self.facebook_access_token.setText(str(preferences["facebook_access_token"]))
        if "instagram_business_id" in preferences:
            self.instagram_business_id.setText(str(preferences["instagram_business_id"]))
        if "instagram_access_token" in preferences:
            self.instagram_access_token.setText(str(preferences["instagram_access_token"]))
        if "concept" in preferences:
            self.concept.setPlainText(str(preferences["concept"]))
        if "manual_prompt" in preferences:
            self.manual_prompt.setPlainText(str(preferences["manual_prompt"]))
        if "manual_prompt_default" in preferences:
            default_prompt = str(preferences["manual_prompt_default"])
            self.manual_prompt_default_input.setPlainText(default_prompt)
            if "manual_prompt" not in preferences:
                self.manual_prompt.setPlainText(default_prompt)
        if "ai_social_metadata" in preferences and isinstance(preferences["ai_social_metadata"], dict):
            metadata = preferences["ai_social_metadata"]
            hashtags = metadata.get("hashtags", self.ai_social_metadata.hashtags)
            self.ai_social_metadata = AISocialMetadata(
                title=str(metadata.get("title", self.ai_social_metadata.title)),
                description=str(metadata.get("description", self.ai_social_metadata.description)),
                hashtags=[str(tag).strip().lstrip("#") for tag in hashtags if str(tag).strip()],
                category=str(metadata.get("category", self.ai_social_metadata.category)),
            )
        if "count" in preferences:
            try:
                self.count.setValue(int(preferences["count"]))
            except (TypeError, ValueError):
                pass
        if "video_resolution" in preferences:
            resolution_index = self.video_resolution.findData(str(preferences["video_resolution"]))
            if resolution_index >= 0:
                self.video_resolution.setCurrentIndex(resolution_index)
        if "video_duration_seconds" in preferences:
            try:
                duration_value = int(preferences["video_duration_seconds"])
                duration_index = self.video_duration.findData(duration_value)
                if duration_index >= 0:
                    self.video_duration.setCurrentIndex(duration_index)
            except (TypeError, ValueError):
                pass
        if "video_aspect_ratio" in preferences:
            aspect_index = self.video_aspect_ratio.findData(str(preferences["video_aspect_ratio"]))
            if aspect_index >= 0:
                self.video_aspect_ratio.setCurrentIndex(aspect_index)
        if "stitch_crossfade_enabled" in preferences:
            self.stitch_crossfade_checkbox.setChecked(bool(preferences["stitch_crossfade_enabled"]))
        if "stitch_interpolation_enabled" in preferences:
            self.stitch_interpolation_checkbox.setChecked(bool(preferences["stitch_interpolation_enabled"]))
        if "stitch_interpolation_fps" in preferences:
            fps = str(preferences["stitch_interpolation_fps"]).strip()
            fps_index = self.stitch_interpolation_fps.findData(int(fps)) if fps.isdigit() else -1
            if fps_index >= 0:
                self.stitch_interpolation_fps.setCurrentIndex(fps_index)
        if "stitch_upscale_enabled" in preferences:
            self.stitch_upscale_checkbox.setChecked(bool(preferences["stitch_upscale_enabled"]))
        if "stitch_upscale_target" in preferences:
            target_index = self.stitch_upscale_target.findData(str(preferences["stitch_upscale_target"]))
            if target_index >= 0:
                self.stitch_upscale_target.setCurrentIndex(target_index)
        if "stitch_gpu_enabled" in preferences:
            self.stitch_gpu_checkbox.setChecked(bool(preferences["stitch_gpu_enabled"]))
        if "stitch_mute_original_audio" in preferences:
            self.stitch_mute_original_checkbox.setChecked(bool(preferences["stitch_mute_original_audio"]))
        if "stitch_original_audio_volume" in preferences:
            try:
                self.stitch_original_audio_volume.setValue(int(preferences["stitch_original_audio_volume"]))
            except (TypeError, ValueError):
                pass
        if "stitch_music_volume" in preferences:
            try:
                self.stitch_music_volume.setValue(int(preferences["stitch_music_volume"]))
            except (TypeError, ValueError):
                pass
        if "stitch_audio_fade_duration" in preferences:
            try:
                self.stitch_audio_fade_duration.setValue(float(preferences["stitch_audio_fade_duration"]))
            except (TypeError, ValueError):
                pass
        if "stitch_custom_music_file" in preferences:
            music_candidate = Path(str(preferences["stitch_custom_music_file"]))
            if str(preferences["stitch_custom_music_file"]).strip() and music_candidate.exists():
                self.custom_music_file = music_candidate
                self.music_file_label.setText(f"Music: {music_candidate.name}")
            else:
                self.custom_music_file = None
                self.music_file_label.setText("Music: none selected")
        if "crossfade_duration" in preferences:
            try:
                self.crossfade_duration.setValue(float(preferences["crossfade_duration"]))
            except (TypeError, ValueError):
                pass
        if "download_dir" in preferences:
            download_dir = Path(str(preferences["download_dir"]))
            try:
                download_dir.mkdir(parents=True, exist_ok=True)
                self.download_dir = download_dir
                self.download_path_input.setText(str(self.download_dir))
            except Exception:
                pass
        if "preview_muted" in preferences:
            self.preview_mute_checkbox.setChecked(bool(preferences["preview_muted"]))
        if "preview_volume" in preferences:
            try:
                self.preview_volume_slider.setValue(int(preferences["preview_volume"]))
            except (TypeError, ValueError):
                pass

        self._toggle_prompt_source_fields()
        self._sync_video_options_label()

    def _save_preferences_to_path(self, file_path: Path, *, show_feedback: bool = False) -> bool:
        try:
            preferences = self._collect_preferences()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump(preferences, handle, indent=2)
        except Exception as exc:
            if show_feedback:
                QMessageBox.critical(self, "Save Preferences Failed", str(exc))
            self._append_log(f"ERROR: Could not save preferences: {exc}")
            return False

        self._append_log(f"Saved preferences to: {file_path}")
        return True

    def _load_preferences_from_path(self, file_path: Path, *, show_feedback: bool = False) -> bool:
        if not file_path.exists():
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                preferences = json.load(handle)
            self._apply_preferences(preferences)
        except Exception as exc:
            if show_feedback:
                QMessageBox.critical(self, "Load Preferences Failed", str(exc))
            self._append_log(f"ERROR: Could not load preferences: {exc}")
            return False

        self._append_log(f"Loaded preferences from: {file_path}")
        return True

    def save_model_api_settings(self) -> None:
        if self._save_preferences_to_path(DEFAULT_PREFERENCES_FILE, show_feedback=True):
            QMessageBox.information(self, "Settings Saved", f"Settings saved to:\n{DEFAULT_PREFERENCES_FILE}")

    def _load_startup_preferences(self) -> None:
        self._load_preferences_from_path(DEFAULT_PREFERENCES_FILE, show_feedback=False)

    def save_preferences(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Preferences",
            str(DEFAULT_PREFERENCES_FILE),
            "JSON Files (*.json)",
        )
        if not file_path:
            return

        self._save_preferences_to_path(Path(file_path), show_feedback=True)

    def load_preferences(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Preferences",
            str(DEFAULT_PREFERENCES_FILE),
            "JSON Files (*.json)",
        )
        if not file_path:
            return

        self._load_preferences_from_path(Path(file_path), show_feedback=True)

    def _append_log(self, text: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.appendPlainText(f"[{timestamp}] {text}")

    def _on_browser_load_finished(self, ok: bool) -> None:
        if ok:
            self._playback_hack_success_logged = False
            self.video_playback_hack_timer.start()
            self._ensure_browser_video_playback()
            if self.continue_from_frame_waiting_for_reload and self.continue_from_frame_active:
                self.continue_from_frame_waiting_for_reload = False
                self.continue_from_frame_reload_timeout_timer.stop()
                self._append_log(
                    "Continue-from-last-frame: detected page reload after image upload. Proceeding with prompt entry."
                )
                QTimer.singleShot(700, lambda: self._start_manual_browser_generation(self.continue_from_frame_prompt, 1))

    def _retry_continue_after_small_download(self, variant: int) -> None:
        source_video = self.continue_from_frame_current_source_video
        self._append_log(
            f"Variant {variant} download is smaller than 1MB; restarting from previous source video: {source_video}"
        )

        if not source_video or not Path(source_video).exists():
            self._append_log(
                "ERROR: Previous source video is unavailable for retry; continue-from-last-frame workflow is stopping."
            )
            self.continue_from_frame_active = False
            self.continue_from_frame_target_count = 0
            self.continue_from_frame_completed = 0
            self.continue_from_frame_prompt = ""
            self.continue_from_frame_current_source_video = ""
            return

        frame_path = self._extract_last_frame(source_video)
        if frame_path is None:
            self._append_log(
                "ERROR: Could not extract a last frame from the previous source video; "
                "continue-from-last-frame workflow is stopping."
            )
            self.continue_from_frame_active = False
            self.continue_from_frame_target_count = 0
            self.continue_from_frame_completed = 0
            self.continue_from_frame_prompt = ""
            self.continue_from_frame_current_source_video = ""
            return

        self.last_extracted_frame_path = frame_path
        self._upload_frame_into_grok(frame_path, on_uploaded=self._wait_for_continue_upload_reload)

    def _ensure_browser_video_playback(self) -> None:
        if not hasattr(self, "browser") or self.browser is None:
            return

        script = r"""
            (() => {
                try {
                    const videos = [...document.querySelectorAll("video")];
                    if (!videos.length) return { ok: true, found: 0, attempted: 0, playing: 0 };

                    const pokeUserGesture = () => {
                        try {
                            const ev = new MouseEvent("click", { bubbles: true, cancelable: true, composed: true });
                            document.body.dispatchEvent(ev);
                        } catch (_) {}
                    };
                    pokeUserGesture();

                    const common = { bubbles: true, cancelable: true, composed: true };
                    const synthClick = (el) => {
                        if (!el) return;
                        try {
                            el.dispatchEvent(new PointerEvent("pointerdown", common));
                            el.dispatchEvent(new MouseEvent("mousedown", common));
                            el.dispatchEvent(new PointerEvent("pointerup", common));
                            el.dispatchEvent(new MouseEvent("mouseup", common));
                            el.dispatchEvent(new MouseEvent("click", common));
                        } catch (_) {}
                    };

                    let attempted = 0;
                    let playing = 0;
                    for (const video of videos) {
                        try {
                            video.muted = false;
                            video.defaultMuted = false;
                            video.volume = 1;
                            video.autoplay = true;
                            video.playsInline = true;
                            video.loop = false;
                            video.removeAttribute("muted");
                            video.setAttribute("autoplay", "");
                            video.setAttribute("playsinline", "");
                            video.setAttribute("webkit-playsinline", "");
                            video.controls = true;
                            video.preload = "auto";

                            const st = getComputedStyle(video);
                            if (st.display === "none") video.style.display = "block";
                            if (st.visibility === "hidden") video.style.visibility = "visible";
                            if (Number(st.opacity || "1") < 0.1) video.style.opacity = "1";

                            if (video.paused || video.readyState < 2) {
                                attempted += 1;
                                const p = video.play();
                                if (p && typeof p.catch === "function") {
                                    p.catch(() => {
                                        const playButton = video.closest("[role='button'], button")
                                            || video.parentElement?.querySelector("button, [role='button']");
                                        synthClick(playButton);
                                        const p2 = video.play();
                                        if (p2 && typeof p2.catch === "function") p2.catch(() => {});
                                    });
                                }

                                if (video.muted) {
                                    try {
                                        video.muted = false;
                                        video.defaultMuted = false;
                                        video.volume = 1;
                                        video.removeAttribute("muted");
                                    } catch (_) {}
                                }

                                if (video.readyState < 2) {
                                    video.addEventListener("canplay", () => {
                                        const p3 = video.play();
                                        if (p3 && typeof p3.catch === "function") p3.catch(() => {});
                                    }, { once: true });
                                }
                            }

                            if (!video.paused && !video.ended && video.currentTime >= 0) {
                                playing += 1;
                            }
                        } catch (_) {}
                    }

                    return { ok: true, found: videos.length, attempted, playing };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """

        def after(result):
            if not isinstance(result, dict) or not result.get("ok"):
                return
            if result.get("found", 0) > 0 and result.get("playing", 0) > 0 and not self._playback_hack_success_logged:
                self._append_log(
                    f"Playback hack active: {result.get('playing')}/{result.get('found')} embedded video element(s) are playing."
                )
                self._playback_hack_success_logged = True

        self.browser.page().runJavaScript(script, after)

    def start_generation(self) -> None:
        self.stop_all_requested = False
        concept = self.concept.toPlainText().strip()
        source = self.prompt_source.currentData()
        manual_prompt = self.manual_prompt.toPlainText().strip()

        if source != "manual":
            api_key = self.api_key.text().strip()
            if not api_key:
                QMessageBox.warning(self, "Missing API Key", "Please enter a Grok API key.")
                return

        if source != "manual" and not concept:
            QMessageBox.warning(self, "Missing Concept", "Please enter a concept.")
            return
        if source == "manual" and not manual_prompt:
            QMessageBox.warning(self, "Missing Manual Prompt", "Please enter a manual prompt.")
            return
        if source == "openai" and not self.openai_api_key.text().strip():
            QMessageBox.warning(self, "Missing OpenAI API Key", "Please enter an OpenAI API key.")
            return

        if source == "manual":
            self._start_manual_browser_generation(manual_prompt, self.count.value())
            return

        config = GrokConfig(
            api_key=api_key,
            chat_model=self.chat_model.text().strip() or "grok-3-mini",
            image_model=self.image_model.text().strip() or "grok-video-latest",
        )

        selected_resolution = str(self.video_resolution.currentData() or "1280x720")
        selected_resolution_label = self.video_resolution.currentText().split(" ", 1)[0]
        selected_duration_seconds = int(self.video_duration.currentData() or 10)
        selected_aspect_ratio = str(self.video_aspect_ratio.currentData() or "16:9")

        prompt_config = PromptConfig(
            source=source,
            concept=concept,
            manual_prompt=manual_prompt,
            openai_api_key=self.openai_api_key.text().strip(),
            openai_chat_model=self.openai_chat_model.text().strip() or "gpt-4o-mini",
            video_resolution=selected_resolution,
            video_resolution_label=selected_resolution_label,
            video_aspect_ratio=selected_aspect_ratio,
            video_duration_seconds=selected_duration_seconds,
        )

        self.worker = GenerateWorker(config, prompt_config, self.count.value(), self.download_dir)
        self.worker.status.connect(self._append_log)
        self.worker.finished_video.connect(self.on_video_finished)
        self.worker.failed.connect(self.on_generation_error)
        self.worker.start()

    def open_ai_provider_login(self) -> None:
        source = self.prompt_source.currentData()
        if source == "openai":
            self.browser.setUrl(QUrl("https://platform.openai.com/settings/organization/api-keys"))
            self._append_log("Opened OpenAI platform in browser for sign-in/API key management.")
            return

        self.browser.setUrl(QUrl("https://grok.com/"))
        self._append_log("Opened Grok in browser for sign-in.")

    def _call_selected_ai(self, system: str, user: str) -> str:
        source = self.prompt_source.currentData()
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.4,
        }

        if source == "openai":
            openai_key = self.openai_api_key.text().strip()
            if not openai_key:
                raise RuntimeError("OpenAI API key is required.")
            headers["Authorization"] = f"Bearer {openai_key}"
            payload["model"] = self.openai_chat_model.text().strip() or "gpt-4o-mini"
            response = requests.post(f"{OPENAI_API_BASE}/chat/completions", headers=headers, json=payload, timeout=90)
            if not response.ok:
                raise RuntimeError(f"OpenAI request failed: {response.status_code} {response.text[:400]}")
            return response.json()["choices"][0]["message"]["content"].strip()

        grok_key = self.api_key.text().strip()
        if not grok_key:
            if self.ai_auth_method.currentData() == "browser":
                raise RuntimeError("Browser authorization selected, but API-based Grok requests still require a GROK API key.")
            raise RuntimeError("Grok API key is required.")
        headers["Authorization"] = f"Bearer {grok_key}"
        payload["model"] = self.chat_model.text().strip() or "grok-3-mini"
        response = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=90)
        if not response.ok:
            raise RuntimeError(f"Grok request failed: {response.status_code} {response.text[:400]}")
        return response.json()["choices"][0]["message"]["content"].strip()

    def generate_prompt_from_concept(self) -> None:
        concept = self.concept.toPlainText().strip()
        source = self.prompt_source.currentData()
        if source not in {"grok", "openai"}:
            QMessageBox.warning(self, "AI Source Required", "Set Prompt Source to Grok API or OpenAI API.")
            return
        if not concept:
            QMessageBox.warning(self, "Missing Concept", "Please enter a concept first.")
            return

        try:
            instruction = concept + " please turn this into a detailed 10 second prompt for grok imagine"
            system = "You are an expert prompt and social metadata generator for short-form AI videos. Return strict JSON only."
            user = (
                "Generate JSON with keys: manual_prompt, title, description, hashtags, category. "
                "manual_prompt should be detailed and cinematic for a 10-second Grok Imagine clip. "
                "title should be short and catchy. description should be 1-3 sentences. "
                "hashtags should be an array of 5-12 hashtag strings without # prefixes. "
                "category should be the best YouTube category id as a string (default 22 if unsure). "
                f"Concept instruction: {instruction}"
            )
            raw = self._call_selected_ai(system, user)
            parsed = json.loads(raw)
            manual_prompt = str(parsed.get("manual_prompt", "")).strip()
            if not manual_prompt:
                raise RuntimeError("AI response did not include a manual_prompt.")

            hashtags = parsed.get("hashtags", [])
            cleaned_hashtags = [str(tag).strip().lstrip("#") for tag in hashtags if str(tag).strip()]
            self.ai_social_metadata = AISocialMetadata(
                title=str(parsed.get("title", "AI Generated Video")).strip() or "AI Generated Video",
                description=str(parsed.get("description", "")).strip(),
                hashtags=cleaned_hashtags or ["grok", "ai", "generated-video"],
                category=str(parsed.get("category", "22")).strip() or "22",
            )
            self.manual_prompt.setPlainText(manual_prompt)
            self._append_log(
                "AI updated Manual Prompt and social metadata defaults "
                f"(title/category/hashtags: {self.ai_social_metadata.title}/{self.ai_social_metadata.category}/"
                f"{', '.join(self.ai_social_metadata.hashtags)})."
            )
        except json.JSONDecodeError:
            QMessageBox.critical(self, "AI Response Error", "AI response was not valid JSON. Please retry.")
        except Exception as exc:
            QMessageBox.critical(self, "Prompt Generation Failed", str(exc))

    def start_image_generation(self) -> None:
        self.stop_all_requested = False
        source = self.prompt_source.currentData()
        manual_prompt = self.manual_prompt.toPlainText().strip()

        if source != "manual":
            QMessageBox.warning(
                self,
                "Manual Prompt Required",
                "Populate Image Prompt in Browser is only available when Prompt Source is set to Manual prompt (no API).",
            )
            return

        if not manual_prompt:
            QMessageBox.warning(self, "Missing Manual Prompt", "Please enter a manual prompt.")
            return
        
        self._start_manual_browser_image_generation(manual_prompt, self.count.value())

    def _start_manual_browser_generation(self, prompt: str, count: int) -> None:
        self.manual_generation_queue = [{"prompt": prompt, "variant": idx} for idx in range(1, count + 1)]
        self._append_log(
            "Manual mode now reuses the current browser page exactly as-is. "
            "No navigation or reload will happen."
        )
        self._append_log(f"Manual mode queued with repeat count={count}.")
        self._append_log("Attempting to populate the visible Grok prompt box on the current page...")
        self._submit_next_manual_variant()

    def _start_manual_browser_image_generation(self, prompt: str, count: int) -> None:
        self.manual_image_generation_queue = [{"prompt": prompt, "variant": idx} for idx in range(1, count + 1)]
        self._append_log(
            "Manual image mode now reuses the current browser page exactly as-is. "
            "No navigation or reload will happen."
        )
        self._append_log(f"Manual image mode queued with repeat count={count}.")
        self._submit_next_manual_image_variant()

    def _submit_next_manual_image_variant(self) -> None:
        if self.stop_all_requested:
            self._append_log("Stop-all flag active; skipping queued job activity.")
            return

        if not self.manual_image_generation_queue:
            self._append_log("Manual browser image generation complete.")
            return

        item = self.manual_image_generation_queue.pop(0)
        variant = item["variant"]
        prompt = item["prompt"]
        attempts = int(item.get("attempts", 0)) + 1
        item["attempts"] = attempts
        self.pending_manual_variant_for_download = variant
        self.pending_manual_download_type = "image"
        self.pending_manual_image_prompt = prompt
        self.manual_image_pick_clicked = False
        self.manual_image_video_submit_sent = False
        self.manual_image_pick_retry_count = 0
        self.manual_image_submit_retry_count = 0
        self.manual_image_submit_token += 1
        self.manual_download_click_sent = False
        selected_aspect_ratio = str(self.video_aspect_ratio.currentData() or "16:9")

        populate_script = rf"""
            (() => {{
                try {{
                    const prompt = {prompt!r};
                    const selectors = [
                        "textarea[placeholder*='Type to imagine' i]",
                        "input[placeholder*='Type to imagine' i]",
                        "textarea[placeholder*='Type to customize this video' i]",
                        "input[placeholder*='Type to customize this video' i]",
                        "textarea[placeholder*='Type to customize video' i]",
                        "input[placeholder*='Type to customize video' i]",
                        "textarea[placeholder*='Customize video' i]",
                        "input[placeholder*='Customize video' i]",
                        "div.tiptap.ProseMirror[contenteditable='true']",
                        "[contenteditable='true'][aria-label*='Type to imagine' i]",
                        "[contenteditable='true'][data-placeholder*='Type to imagine' i]",
                        "[contenteditable='true'][aria-label*='Type to customize this video' i]",
                        "[contenteditable='true'][data-placeholder*='Type to customize this video' i]",
                        "[contenteditable='true'][aria-label*='Type to customize video' i]",
                        "[contenteditable='true'][data-placeholder*='Type to customize video' i]",
                        "[contenteditable='true'][aria-label*='Make a video' i]",
                        "[contenteditable='true'][data-placeholder*='Customize video' i]"
                    ];
                    const promptInput = selectors.map((sel) => document.querySelector(sel)).find(Boolean);
                    if (!promptInput) return {{ ok: false, error: "Prompt input not found" }};

                    promptInput.focus();
                    if (promptInput.isContentEditable) {{
                        const paragraph = document.createElement("p");
                        paragraph.textContent = prompt;
                        promptInput.replaceChildren(paragraph);
                        promptInput.dispatchEvent(new Event("input", {{ bubbles: true }}));
                        promptInput.dispatchEvent(new Event("change", {{ bubbles: true }}));
                    }} else {{
                        const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(promptInput), "value")?.set;
                        if (setter) setter.call(promptInput, prompt);
                        else promptInput.value = prompt;
                        promptInput.dispatchEvent(new Event("input", {{ bubbles: true }}));
                        promptInput.dispatchEvent(new Event("change", {{ bubbles: true }}));
                    }}
                    const typedValue = promptInput.isContentEditable ? (promptInput.textContent || "") : (promptInput.value || "");
                    if (!typedValue.trim()) return {{ ok: false, error: "Prompt field did not accept text" }};
                    return {{ ok: true, filledLength: typedValue.length }};
                }} catch (err) {{
                    return {{ ok: false, error: String(err && err.stack ? err.stack : err) }};
                }}
            }})()
        """

        set_image_mode_script = r"""
            (() => {
                try {
                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const common = { bubbles: true, cancelable: true, composed: true };
                    const emulateClick = (el) => {
                        if (!el || !isVisible(el) || el.disabled) return false;
                        try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    };
                    const textOf = (el) => (el?.textContent || "").replace(/\s+/g, " ").trim();
                    const hasImageSelectionMarker = () => {
                        const selectedEls = [...document.querySelectorAll("[aria-selected='true'], [aria-pressed='true'], [data-state='checked'], [data-selected='true']")]
                            .filter((el) => isVisible(el));
                        return selectedEls.some((el) => /(^|\s)image(\s|$)/i.test(textOf(el)));
                    };

                    const modelTriggerCandidates = [
                        ...document.querySelectorAll("#model-select-trigger"),
                        ...document.querySelectorAll("button[aria-haspopup='menu'], [role='button'][aria-haspopup='menu']"),
                        ...document.querySelectorAll("button, [role='button']"),
                    ].filter((el, idx, arr) => arr.indexOf(el) === idx && isVisible(el));

                    const modelTrigger = modelTriggerCandidates.find((el) => {
                        const txt = textOf(el);
                        return /model|video|image|options|settings/i.test(txt) || (el.id || "") === "model-select-trigger";
                    }) || null;

                    let optionsOpened = false;
                    if (modelTrigger) {
                        optionsOpened = emulateClick(modelTrigger);
                    }

                    const menuItemSelectors = [
                        "[role='menuitem'][data-radix-collection-item]",
                        "[role='menuitemradio']",
                        "[role='menuitem']",
                        "[role='option']",
                        "[data-radix-collection-item]",
                    ];

                    const menuItems = menuItemSelectors
                        .flatMap((sel) => [...document.querySelectorAll(sel)])
                        .filter((el, idx, arr) => arr.indexOf(el) === idx && isVisible(el));

                    const imageItem = menuItems.find((el) => {
                        const txt = textOf(el);
                        return /(^|\s)image(\s|$)/i.test(txt) || /generate multiple images/i.test(txt);
                    }) || null;

                    const imageClicked = imageItem ? emulateClick(imageItem) : false;

                    const triggerNowSaysImage = !!(modelTrigger && /(^|\s)image(\s|$)/i.test(textOf(modelTrigger)));
                    const imageSelected = imageClicked || hasImageSelectionMarker() || triggerNowSaysImage;

                    return {
                        ok: true,
                        imageSelected,
                        optionsOpened,
                        imageItemFound: !!imageItem,
                        imageClicked,
                        triggerText: modelTrigger ? textOf(modelTrigger) : "",
                    };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """

        set_image_options_script = r"""
            (() => {
                try {
                    const desiredAspect = "{selected_aspect_ratio}";
                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const interactiveSelector = "button, [role='button'], [role='tab'], [role='option'], [role='menuitemradio'], [role='radio'], label, span, div";
                    const textOf = (el) => (el?.textContent || "").replace(/\s+/g, " ").trim();
                    const clickableAncestor = (el) => {
                        if (!el) return null;
                        if (typeof el.closest === "function") {
                            const ancestor = el.closest("button, [role='button'], [role='tab'], [role='option'], [role='menuitemradio'], [role='radio'], label");
                            if (ancestor) return ancestor;
                        }
                        return el;
                    };
                    const visibleTextElements = (root = document) => [...root.querySelectorAll(interactiveSelector)]
                        .filter((el) => isVisible(el) && textOf(el));
                    const selectedTextElements = (root = document) => visibleTextElements(root)
                        .filter((el) => {
                            const target = clickableAncestor(el);
                            if (!target) return false;
                            const ariaPressed = target.getAttribute("aria-pressed") === "true";
                            const ariaSelected = target.getAttribute("aria-selected") === "true";
                            const dataState = (target.getAttribute("data-state") || "").toLowerCase() === "checked";
                            const dataSelected = target.getAttribute("data-selected") === "true";
                            const classSelected = /\b(active|selected|checked|on)\b/i.test(target.className || "");
                            const checkedInput = !!target.querySelector("input[type='radio']:checked, input[type='checkbox']:checked");
                            return ariaPressed || ariaSelected || dataState || dataSelected || checkedInput || classSelected;
                        });

                    const emulateClick = (el) => {
                        if (!el || !isVisible(el) || el.disabled) return false;
                        const common = { bubbles: true, cancelable: true, composed: true };
                        try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    };

                    const matchesAny = (text, patterns) => patterns.some((pattern) => pattern.test(text));
                    const clickByText = (patterns, root = document) => {
                        const candidate = visibleTextElements(root).find((el) => matchesAny(textOf(el), patterns));
                        const target = clickableAncestor(candidate);
                        if (!target) return false;
                        return emulateClick(target);
                    };
                    const hasSelectedByText = (patterns, root = document) => selectedTextElements(root)
                        .some((el) => matchesAny(textOf(el), patterns));

                    const promptInput = document.querySelector("textarea[placeholder*='Type to imagine' i], input[placeholder*='Type to imagine' i], textarea[placeholder*='Type to customize this video' i], input[placeholder*='Type to customize this video' i], textarea[placeholder*='Type to customize video' i], input[placeholder*='Type to customize video' i], textarea[placeholder*='Customize video' i], input[placeholder*='Customize video' i], div.tiptap.ProseMirror[contenteditable='true'], [contenteditable='true'][aria-label*='Type to imagine' i], [contenteditable='true'][data-placeholder*='Type to imagine' i]");
                    const composer = (promptInput && (promptInput.closest("form") || promptInput.closest("main") || promptInput.closest("section"))) || document;

                    const aspectPatterns = {
                        "2:3": [/^2\s*:\s*3$/i],
                        "3:2": [/^3\s*:\s*2$/i],
                        "1:1": [/^1\s*:\s*1$/i],
                        "9:16": [/^9\s*:\s*16$/i],
                        "16:9": [/^16\s*:\s*9$/i],
                    };
                    const imagePatterns = [/(^|\s)image(\s|$)/i, /generate multiple images/i];
                    const desiredAspectPatterns = aspectPatterns[desiredAspect] || aspectPatterns["16:9"];

                    const optionsRequested = [];
                    const optionsApplied = [];

                    const findVisibleButtonByAriaLabel = (ariaLabel, root = document) => {
                        const candidates = [...root.querySelectorAll(`button[aria-label='${ariaLabel}']`)];
                        return candidates.find((el) => isVisible(el) && !el.disabled) || null;
                    };
                    const isOptionButtonSelected = (button) => {
                        if (!button) return false;
                        const ariaPressed = button.getAttribute("aria-pressed") === "true";
                        const ariaSelected = button.getAttribute("aria-selected") === "true";
                        const dataSelected = button.getAttribute("data-selected") === "true";
                        const dataState = (button.getAttribute("data-state") || "").toLowerCase();
                        if (ariaPressed || ariaSelected || dataSelected || dataState === "checked" || dataState === "active") return true;
                        if (/\b(active|selected|checked|on|text-fg-primary)\b/i.test(button.className || "")) return true;
                        const selectedFill = button.querySelector(".bg-primary:not([class*='bg-primary/'])");
                        return !!selectedFill;
                    };
                    const hasSelectedByAriaLabel = (ariaLabel, root = document) => {
                        const button = findVisibleButtonByAriaLabel(ariaLabel, root);
                        return isOptionButtonSelected(button);
                    };
                    const clickVisibleButtonByAriaLabel = (ariaLabel, root = document) => {
                        const button = findVisibleButtonByAriaLabel(ariaLabel, root) || findVisibleButtonByAriaLabel(ariaLabel, document);
                        if (!button) return false;
                        return emulateClick(button);
                    };

                    const applyOption = (name, patterns, ariaLabel = null) => {
                        const alreadySelected = (ariaLabel && (hasSelectedByAriaLabel(ariaLabel, composer) || hasSelectedByAriaLabel(ariaLabel)))
                            || hasSelectedByText(patterns, composer)
                            || hasSelectedByText(patterns);
                        if (alreadySelected) {
                            optionsApplied.push(`${name}(already-selected)`);
                            return;
                        }
                        const clicked = (ariaLabel && (clickVisibleButtonByAriaLabel(ariaLabel, composer) || clickVisibleButtonByAriaLabel(ariaLabel)))
                            || clickByText(patterns, composer)
                            || clickByText(patterns);
                        if (clicked) optionsRequested.push(name);
                        const selected = (ariaLabel && (hasSelectedByAriaLabel(ariaLabel, composer) || hasSelectedByAriaLabel(ariaLabel)))
                            || hasSelectedByText(patterns, composer)
                            || hasSelectedByText(patterns);
                        if (selected) optionsApplied.push(name);
                    };

                    applyOption("image", imagePatterns);
                    applyOption(desiredAspect, desiredAspectPatterns, desiredAspect);

                    const missingOptions = [];
                    if (!(hasSelectedByText(imagePatterns, composer) || hasSelectedByText(imagePatterns))) {
                        missingOptions.push("image");
                    }
                    if (!(hasSelectedByAriaLabel(desiredAspect, composer) || hasSelectedByAriaLabel(desiredAspect)
                        || hasSelectedByText(desiredAspectPatterns, composer)
                        || hasSelectedByText(desiredAspectPatterns))) {
                        missingOptions.push(desiredAspect);
                    }

                    return {
                        ok: true,
                        desiredAspect,
                        optionsRequested,
                        optionsApplied,
                        missingOptions,
                    };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """
        set_image_options_script = set_image_options_script.replace('"{selected_aspect_ratio}"', json.dumps(selected_aspect_ratio))

        submit_script = r"""
            (() => {
                try {
                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const common = { bubbles: true, cancelable: true, composed: true };
                    const emulateClick = (el) => {
                        if (!el || !isVisible(el) || el.disabled) return false;
                        try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    };

                    const promptInput = document.querySelector("textarea[placeholder*='Type to imagine' i], input[placeholder*='Type to imagine' i], textarea[placeholder*='Type to customize this video' i], input[placeholder*='Type to customize this video' i], textarea[placeholder*='Type to customize video' i], input[placeholder*='Type to customize video' i], textarea[placeholder*='Customize video' i], input[placeholder*='Customize video' i], div.tiptap.ProseMirror[contenteditable='true'], [contenteditable='true'][aria-label*='Type to imagine' i], [contenteditable='true'][data-placeholder*='Type to imagine' i]");
                    const submitSelectors = [
                        "button[type='submit'][aria-label='Submit']",
                        "button[aria-label='Submit'][type='submit']",
                        "button[type='submit']",
                        "button[aria-label*='submit' i]",
                        "[role='button'][aria-label*='submit' i]"
                    ];

                    const candidates = [];
                    const collect = (root) => {
                        if (!root || typeof root.querySelectorAll !== "function") return;
                        submitSelectors.forEach((selector) => {
                            const matches = root.querySelectorAll(selector);
                            for (let i = 0; i < matches.length; i += 1) candidates.push(matches[i]);
                        });
                    };

                    const composerRoot = (promptInput && typeof promptInput.closest === "function")
                        ? (promptInput.closest("form") || promptInput.closest("main") || promptInput.closest("section") || promptInput.parentElement)
                        : null;
                    collect(composerRoot);
                    collect(document);

                    const submitButton = [...new Set(candidates)].find((el) => isVisible(el));
                    if (!submitButton) return { ok: false, error: "Submit button not found" };
                    if (submitButton.disabled) {
                        return { ok: false, waiting: true, status: "submit-disabled" };
                    }

                    const clicked = emulateClick(submitButton);
                    return {
                        ok: clicked,
                        waiting: !clicked,
                        status: clicked ? "submit-clicked" : "submit-click-failed",
                        ariaLabel: submitButton.getAttribute("aria-label") || "",
                        disabled: !!submitButton.disabled,
                    };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """

        def _retry_variant(reason: str) -> None:
            self._append_log(f"WARNING: Manual image variant {variant} attempt {attempts} failed: {reason}")
            if attempts >= 4:
                self._append_log(
                    f"ERROR: Could not prepare manual image variant {variant} after {attempts} attempts; skipping variant."
                )
                self._submit_next_manual_image_variant()
                return
            self.manual_image_generation_queue.insert(0, item)
            QTimer.singleShot(1200, self._submit_next_manual_image_variant)

        submit_attempts = 0

        def _run_submit_attempt() -> None:
            nonlocal submit_attempts
            submit_attempts += 1
            self._append_log(
                f"Manual image variant {variant}: attempting submit click ({submit_attempts}/12)."
            )
            self.browser.page().runJavaScript(submit_script, _after_submit)

        def _after_submit(result):
            if isinstance(result, dict) and result.get("ok"):
                #self.show_browser_page()
                self._append_log(
                    f"Submitted manual image variant {variant} (attempt {attempts}); "
                    "waiting for first rendered image, then opening it for download."
                )
                QTimer.singleShot(7000, self._poll_for_manual_image)
                return

            if isinstance(result, dict) and result.get("waiting"):
                if submit_attempts < 12:
                    self._append_log(
                        f"Manual image variant {variant}: submit button still disabled (attempt {submit_attempts}); retrying click..."
                    )
                    QTimer.singleShot(500, _run_submit_attempt)
                    return
                _retry_variant(f"submit button stayed disabled: {result!r}")
                return

            # Some Grok navigations can clear the JS callback value; treat that as submitted.
            if result in (None, ""):
                #self.show_browser_page()
                self._append_log(
                    f"Submitted manual image variant {variant} (attempt {attempts}); "
                    "submit callback returned empty result after page activity; continuing to image polling."
                )
                QTimer.singleShot(7000, self._poll_for_manual_image)
                return

            _retry_variant(f"submit failed: {result!r}")

        def _after_set_mode(result):
            if result in (None, ""):
                self._append_log(
                    f"Manual image variant {variant}: image-mode callback returned empty result; "
                    "continuing with prompt population and assuming current mode is correct."
                )
            elif not isinstance(result, dict) or not result.get("ok"):
                _retry_variant(f"set image mode script failed: {result!r}")
                return
            elif not result.get("imageSelected"):
                _retry_variant(f"image option not selected: {result!r}")
                return

            self._append_log(
                "Manual image variant "
                f"{variant}: image mode selected={result.get('imageSelected') if isinstance(result, dict) else 'unknown'} "
                f"(opened={result.get('optionsOpened') if isinstance(result, dict) else 'unknown'}, "
                f"itemFound={result.get('imageItemFound') if isinstance(result, dict) else 'unknown'}, "
                f"itemClicked={result.get('imageClicked') if isinstance(result, dict) else 'unknown'}); "
                f"applying aspect option {selected_aspect_ratio} next (attempt {attempts})."
            )

            def _after_image_options(options_result):
                if not isinstance(options_result, dict) or not options_result.get("ok"):
                    self._append_log(
                        f"WARNING: Manual image variant {variant}: image options script failed; continuing. result={options_result!r}"
                    )
                else:
                    requested_summary = ", ".join(options_result.get("optionsRequested") or []) or "none"
                    applied_summary = ", ".join(options_result.get("optionsApplied") or []) or "none detected"
                    missing_summary = ", ".join(options_result.get("missingOptions") or []) or "none"
                    self._append_log(
                        f"Manual image variant {variant}: image options requested: {requested_summary}; "
                        f"applied markers: {applied_summary}; missing: {missing_summary}."
                    )

                QTimer.singleShot(450, lambda: self.browser.page().runJavaScript(populate_script, _after_populate))

            QTimer.singleShot(450, lambda: self.browser.page().runJavaScript(set_image_options_script, _after_image_options))

        def _after_populate(result):
            if result in (None, ""):
                self._append_log(
                    f"Manual image variant {variant}: prompt populate callback returned empty result; "
                    "continuing to submit."
                )
                QTimer.singleShot(450, _run_submit_attempt)
                return

            if not isinstance(result, dict) or not result.get("ok"):
                _retry_variant(f"prompt population failed: {result!r}")
                return

            self._append_log(
                f"Manual image variant {variant}: prompt populated (length={result.get('filledLength', 'unknown')}); submitting prompt."
            )
            QTimer.singleShot(450, _run_submit_attempt)

        self.browser.page().runJavaScript(set_image_mode_script, _after_set_mode)

    def _poll_for_manual_image(self) -> None:
        if self.stop_all_requested:
            self._append_log("Stop-all flag active; skipping queued job activity.")
            return

        variant = self.pending_manual_variant_for_download
        if variant is None or self.pending_manual_download_type != "image":
            return

        prompt = self.pending_manual_image_prompt or ""
        phase = "pick" if not self.manual_image_pick_clicked else "submit"
        script = f"""
            (async () => {{
                const prompt = {prompt!r};
                const phase = {phase!r};
                const submitToken = {self.manual_image_submit_token};
                const ACTION_DELAY_MS = 200;
                const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
                const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                const common = {{ bubbles: true, cancelable: true, composed: true }};
                const emulateClick = (el) => {{
                    if (!el || !isVisible(el) || el.disabled) return false;
                    try {{ el.dispatchEvent(new PointerEvent("pointerdown", common)); }} catch (_) {{}}
                    el.dispatchEvent(new MouseEvent("mousedown", common));
                    try {{ el.dispatchEvent(new PointerEvent("pointerup", common)); }} catch (_) {{}}
                    el.dispatchEvent(new MouseEvent("mouseup", common));
                    el.dispatchEvent(new MouseEvent("click", common));
                    return true;
                }};

                if (phase === "pick") {{
                    const generatedImages = [...document.querySelectorAll("img[alt='Generated image']")]
                        .filter((img) => isVisible(img));
                    if (!generatedImages.length) return {{ ok: false, status: "waiting-for-generated-image" }};

                    const firstImage = generatedImages[0];
                    const listItem = firstImage.closest("[role='listitem']");
                    const clickedImage = emulateClick(firstImage) || emulateClick(listItem) || emulateClick(firstImage.parentElement);
                    if (!clickedImage) return {{ ok: false, status: "generated-image-click-failed" }};
                    await sleep(ACTION_DELAY_MS);
                    return {{ ok: true, status: "generated-image-clicked" }};
                }}

                if (window.__grokManualImageSubmitToken === submitToken) {{
                    return {{ ok: true, status: "video-submit-already-clicked" }};
                }}

                const promptSelectors = [
                    "textarea[placeholder*='Type to customize video' i]",
                    "input[placeholder*='Type to customize video' i]",
                    "textarea[placeholder*='Type to imagine' i]",
                    "input[placeholder*='Type to imagine' i]",
                    "div.tiptap.ProseMirror[contenteditable='true']",
                    "[contenteditable='true'][aria-label*='Type to customize video' i]",
                    "[contenteditable='true'][aria-label*='Type to imagine' i]",
                    "[contenteditable='true'][data-placeholder*='Type to customize video' i]",
                    "[contenteditable='true'][data-placeholder*='Type to imagine' i]",
                ];
                const promptInput = promptSelectors.map((sel) => document.querySelector(sel)).find(Boolean);
                if (!promptInput) return {{ ok: false, status: "image-clicked-waiting-prompt-input" }};

                promptInput.focus();
                await sleep(ACTION_DELAY_MS);

                if (promptInput.isContentEditable) {{
                    const paragraph = document.createElement("p");
                    paragraph.textContent = prompt;
                    promptInput.replaceChildren(paragraph);
                }} else {{
                    const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(promptInput), "value")?.set;
                    if (setter) setter.call(promptInput, prompt);
                    else promptInput.value = prompt;
                }}
                await sleep(ACTION_DELAY_MS);

                promptInput.dispatchEvent(new Event("input", {{ bubbles: true }}));
                await sleep(ACTION_DELAY_MS);
                promptInput.dispatchEvent(new Event("change", {{ bubbles: true }}));
                await sleep(ACTION_DELAY_MS);

                const typedValue = promptInput.isContentEditable ? (promptInput.textContent || "") : (promptInput.value || "");
                if (!typedValue.trim()) return {{ ok: false, status: "prompt-fill-empty" }};

                const submitButton = [...document.querySelectorAll("button[type='submit'], button[aria-label*='submit' i], button")]
                    .find((btn) => isVisible(btn) && !btn.disabled && /submit|make\\s+video/i.test((btn.getAttribute("aria-label") || btn.textContent || "").trim()));
                if (!submitButton) return {{ ok: false, status: "prompt-filled-waiting-submit" }};

                await sleep(ACTION_DELAY_MS);
                const submitted = emulateClick(submitButton);
                if (submitted) window.__grokManualImageSubmitToken = submitToken;
                return {{
                    ok: submitted,
                    status: submitted ? "video-submit-clicked" : "submit-click-failed",
                    buttonLabel: (submitButton.getAttribute("aria-label") || submitButton.textContent || "").trim(),
                    filledLength: typedValue.length,
                }};
            }})()
        """


        def _after_poll(result):
            current_variant = self.pending_manual_variant_for_download
            if current_variant is None:
                return

            if isinstance(result, dict) and result.get("ok"):
                status = result.get("status") or "ok"
                if status == "generated-image-clicked":
                    if not self.manual_image_pick_clicked:
                        self._append_log(
                            f"Variant {current_variant}: clicked first generated image tile; preparing video prompt + submit."
                        )
                    self.manual_image_pick_clicked = True
                    self.manual_image_pick_retry_count = 0
                    self.manual_image_submit_retry_count = 0
                    QTimer.singleShot(1000, self._poll_for_manual_image)
                    return

                if status in ("video-submit-clicked", "video-submit-already-clicked"):
                    if status == "video-submit-clicked":
                        detail = result.get("buttonLabel") or "submit"
                        filled_length = result.get("filledLength")
                        if isinstance(filled_length, int):
                            message = f"video prompt submitted via '{detail}' (length={filled_length})"
                        else:
                            message = f"video prompt submitted via '{detail}'"
                    else:
                        message = "submit was already clicked earlier; waiting for video render/download"

                    if not self.manual_image_video_submit_sent:
                        self._append_log(f"Variant {current_variant}: {message}.")
                    self.manual_image_video_submit_sent = True
                    self.manual_image_submit_retry_count = 0
                    self.pending_manual_download_type = "video"
                    self._trigger_browser_video_download(current_variant)
                    return

            status = result.get("status") if isinstance(result, dict) else "callback-empty"
            if not self.manual_image_pick_clicked:
                self.manual_image_pick_retry_count += 1
                if self.manual_image_pick_retry_count >= self.MANUAL_IMAGE_PICK_RETRY_LIMIT:
                    self._append_log(
                        "WARNING: Variant "
                        f"{current_variant}: image pick validation stayed in '{status}' for "
                        f"{self.manual_image_pick_retry_count} checks; forcing prompt submit stage."
                    )
                    self.manual_image_pick_clicked = True
                    self.manual_image_pick_retry_count = 0
                    self.manual_image_submit_retry_count = 0
                    QTimer.singleShot(1000, self._poll_for_manual_image)
                    return
                self._append_log(
                    f"Variant {current_variant}: generated image not ready for pick+submit yet ({status}); retrying..."
                )
                QTimer.singleShot(3000, self._poll_for_manual_image)
                return

            self.manual_image_submit_retry_count += 1
            if self.manual_image_submit_retry_count >= self.MANUAL_IMAGE_SUBMIT_RETRY_LIMIT:
                self._append_log(
                    "WARNING: Variant "
                    f"{current_variant}: submit-stage validation stayed in '{status}' for "
                    f"{self.manual_image_submit_retry_count} checks; assuming submit succeeded and continuing to download polling."
                )
                self.manual_image_video_submit_sent = True
                self.manual_image_submit_retry_count = 0
                self.pending_manual_download_type = "video"
                self._trigger_browser_video_download(current_variant)
                return

            self._append_log(
                f"Variant {current_variant}: video submit stage not ready yet ({status}); retrying..."
            )
            QTimer.singleShot(3000, self._poll_for_manual_image)

        self.browser.page().runJavaScript(script, _after_poll)

    def _start_continue_iteration(self) -> None:
        if self.stop_all_requested:
            self._append_log("Stop-all flag active; skipping queued job activity.")
            return

        frame_path: Path | None = None
        if self.continue_from_frame_seed_image_path is not None:
            frame_path = self.continue_from_frame_seed_image_path
            self._append_log(f"Continue-from-image: using selected image: {frame_path}")
        else:
            latest_video = self._resolve_latest_video_for_continuation()
            if not latest_video:
                self._append_log("ERROR: No videos available to continue from last frame.")
                self.continue_from_frame_active = False
                self.continue_from_frame_current_source_video = ""
                return

            self.continue_from_frame_current_source_video = latest_video
            self._append_log(f"Continue-from-last-frame: extracting frame from source video: {latest_video}")
            frame_path = self._extract_last_frame(latest_video)
            if frame_path is None:
                self._append_log("ERROR: Continue-from-last-frame stopped because frame extraction failed.")
                self.continue_from_frame_active = False
                self.continue_from_frame_current_source_video = ""
                return
            self._append_log(f"Continue-from-last-frame: extracted last frame to {frame_path}")
            if not self._copy_image_to_clipboard(frame_path):
                self._append_log("ERROR: Continue-from-last-frame stopped because clipboard image copy failed.")
                self.continue_from_frame_active = False
                self.continue_from_frame_current_source_video = ""
                return

        self.last_extracted_frame_path = frame_path
        iteration = self.continue_from_frame_completed + 1
        self._append_log(
            f"Continue iteration {iteration}/{self.continue_from_frame_target_count}: using seed image {frame_path}"
        )
        browser_page_pause_ms = 200
        self._append_log(
            "Continue mode: starting image paste into the current Grok prompt area without forcing page navigation..."
        )
        QTimer.singleShot(
            9000 + browser_page_pause_ms,
            lambda: self._upload_frame_into_grok(frame_path, on_uploaded=self._wait_for_continue_upload_reload),
        )
        self._append_log(
            "Continue mode: image paste scheduled; waiting for upload/reload before prompt submission."
        )

    def _resolve_latest_video_for_continuation(self) -> str | None:
        if self.videos:
            return self.videos[-1]["video_file_path"]

        candidates: list[Path] = []
        for pattern in ("*.mp4", "*.mov", "*.webm"):
            candidates.extend(self.download_dir.glob(pattern))

        files = [path for path in candidates if path.is_file()]
        if not files:
            return None

        latest = max(files, key=lambda path: path.stat().st_mtime)
        self._append_log(
            "Continue-from-last-frame fallback: no in-session videos found, "
            f"using latest file from downloads folder: {latest}"
        )
        return str(latest)

    def _submit_next_manual_variant(self) -> None:
        if self.stop_all_requested:
            self._append_log("Stop-all flag active; skipping queued job activity.")
            return

        if not self.manual_generation_queue:
            self._append_log("Manual browser generation complete.")
            return

        item = self.manual_generation_queue.pop(0)
        remaining_count = len(self.manual_generation_queue)
        prompt = item["prompt"]
        variant = item["variant"]
        self.pending_manual_variant_for_download = variant
        self.pending_manual_download_type = "video"
        self.manual_download_click_sent = False
        action_delay_ms = 1000
        selected_quality_label = self.video_resolution.currentText().split(" ", 1)[0]
        selected_duration_label = f"{int(self.video_duration.currentData() or 10)}s"
        selected_aspect_ratio = str(self.video_aspect_ratio.currentData() or "16:9")
        self._append_log(
            f"Populating prompt for manual variant {variant} in browser, setting video options "
            f"({selected_quality_label}, {selected_aspect_ratio}), then force submitting with {action_delay_ms}ms delays between each action. "
            f"Remaining repeats after this: {remaining_count}."
        )

        escaped_prompt = repr(prompt)
        script = rf"""
            (() => {{
                try {{
                    const prompt = {escaped_prompt};
                    const promptSelectors = [
                        "textarea[placeholder*='Type to imagine' i]",
                        "input[placeholder*='Type to imagine' i]",
                        "textarea[placeholder*='Type to customize this video' i]",
                        "input[placeholder*='Type to customize this video' i]",
                        "textarea[placeholder*='Type to customize video' i]",
                        "input[placeholder*='Type to customize video' i]",
                        "textarea[placeholder*='Customize video' i]",
                        "input[placeholder*='Customize video' i]",
                        "div.tiptap.ProseMirror[contenteditable='true']",
                        "[contenteditable='true'][aria-label*='Type to imagine' i]",
                        "[contenteditable='true'][data-placeholder*='Type to imagine' i]",
                        "[contenteditable='true'][aria-label*='Type to customize this video' i]",
                        "[contenteditable='true'][data-placeholder*='Type to customize this video' i]",
                        "[contenteditable='true'][aria-label*='Type to customize video' i]",
                        "[contenteditable='true'][data-placeholder*='Type to customize video' i]",
                        "[contenteditable='true'][aria-label*='Make a video' i]",
                        "[contenteditable='true'][data-placeholder*='Customize video' i]"
                    ];

                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const inputCandidates = [];
                    promptSelectors.forEach((selector) => {{
                        const matches = document.querySelectorAll(selector);
                        for (let i = 0; i < matches.length; i += 1) inputCandidates.push(matches[i]);
                    }});
                    const input = inputCandidates.find((el) => isVisible(el));
                    if (!input) return {{ ok: false, error: "Prompt input not found" }};

                    input.focus();
                    if (input.isContentEditable) {{
                        // Only populate the field; do not synthesize Enter/submit key events.
                        const paragraph = document.createElement("p");
                        paragraph.textContent = prompt;
                        input.replaceChildren(paragraph);
                        input.dispatchEvent(new Event("input", {{ bubbles: true }}));
                        input.dispatchEvent(new Event("change", {{ bubbles: true }}));
                    }} else {{
                        const proto = Object.getPrototypeOf(input);
                        const descriptor = proto ? Object.getOwnPropertyDescriptor(proto, "value") : null;
                        const valueSetter = descriptor && descriptor.set;
                        if (valueSetter) valueSetter.call(input, prompt);
                        else input.value = prompt;
                        input.dispatchEvent(new Event("input", {{ bubbles: true }}));
                        input.dispatchEvent(new Event("change", {{ bubbles: true }}));
                    }}

                    const typedValue = input.isContentEditable ? (input.textContent || "") : (input.value || "");
                    if (!typedValue.trim()) return {{ ok: false, error: "Prompt field did not accept text" }};

                    return {{ ok: true, filledLength: typedValue.length }};
                }} catch (err) {{
                    return {{ ok: false, error: String(err && err.stack ? err.stack : err) }};
                }}
            }})()
        """

        open_options_script = r"""
            (() => {
                try {
                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const common = { bubbles: true, cancelable: true, composed: true };
                    const emulateClick = (el) => {
                        if (!el || !isVisible(el) || el.disabled) return false;
                        try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    };

                    const hasVisibleOptionsPanel = () => {
                        const popupSelectors = [
                            "[role='dialog']",
                            "[role='menu']",
                            "[role='listbox']",
                            "[data-state='open']",
                            "[data-expanded='true']",
                            "[aria-modal='true']"
                        ];
                        return popupSelectors.some((selector) => [...document.querySelectorAll(selector)]
                            .some((el) => isVisible(el) && /(video|720|10\s*s|16\s*:\s*9)/i.test((el.textContent || "").trim())));
                    };

                    const triggerCandidates = [
                        "#model-select-trigger",
                        "button[aria-haspopup='dialog']",
                        "button[aria-haspopup='menu']",
                        "button[aria-expanded='false']",
                        "button[aria-expanded='true']",
                        "[role='button'][aria-haspopup='dialog']",
                        "[role='button'][aria-haspopup='menu']",
                        "button"
                    ];

                    const byText = (el) => /(^|\s)(model|options?|settings?)($|\s)/i.test((el.textContent || "").trim());
                    let trigger = null;
                    for (const selector of triggerCandidates) {
                        const matches = [...document.querySelectorAll(selector)]
                            .filter((el) => isVisible(el) && !el.disabled)
                            .filter((el) => selector !== "button" || byText(el));
                        if (matches.length) {
                            trigger = matches[0];
                            break;
                        }
                    }

                    if (!trigger) {
                        return { ok: false, error: "Model/options trigger not found", panelVisible: hasVisibleOptionsPanel() };
                    }

                    const opened = emulateClick(trigger);
                    return {
                        ok: opened || hasVisibleOptionsPanel(),
                        opened,
                        panelVisible: hasVisibleOptionsPanel(),
                        triggerText: (trigger.textContent || "").trim(),
                        triggerId: trigger.id || ""
                    };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """

        verify_prompt_script = r"""
            (() => {
                try {
                    const promptInput = document.querySelector("textarea[placeholder*='Type to imagine' i], input[placeholder*='Type to imagine' i], textarea[placeholder*='Type to customize this video' i], input[placeholder*='Type to customize this video' i], textarea[placeholder*='Type to customize video' i], input[placeholder*='Type to customize video' i], textarea[placeholder*='Customize video' i], input[placeholder*='Customize video' i], textarea[aria-label*='Make a video' i], input[aria-label*='Make a video' i], div.tiptap.ProseMirror[contenteditable='true'], [contenteditable='true'][aria-label*='Type to imagine' i], [contenteditable='true'][data-placeholder*='Type to imagine' i], [contenteditable='true'][aria-label*='Type to customize this video' i], [contenteditable='true'][data-placeholder*='Type to customize this video' i], [contenteditable='true'][aria-label*='Type to customize video' i], [contenteditable='true'][data-placeholder*='Type to customize video' i], [contenteditable='true'][aria-label*='Make a video' i], [contenteditable='true'][data-placeholder*='Customize video' i]");
                    if (!promptInput) return { ok: false, error: "Prompt input not found during verification" };
                    const value = promptInput.isContentEditable ? (promptInput.textContent || "") : (promptInput.value || "");
                    return { ok: !!value.trim(), filledLength: value.length };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """

        set_options_script = r"""
            (() => {
                try {
                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const interactiveSelector = "button, [role='button'], [role='tab'], [role='option'], [role='menuitemradio'], [role='radio'], label, span, div";
                    const clickableAncestor = (el) => {
                        if (!el) return null;
                        if (typeof el.closest === 'function') {
                            const ancestor = el.closest("button, [role='button'], [role='tab'], [role='option'], [role='menuitemradio'], [role='radio'], label");
                            if (ancestor) return ancestor;
                        }
                        return el;
                    };
                    const matchesAny = (text, patterns) => patterns.some((pattern) => pattern.test(text));
                    const visibleTextElements = (root = document) => [...root.querySelectorAll(interactiveSelector)]
                        .filter((el) => isVisible(el) && (el.textContent || "").trim());
                    const selectedTextElements = (root = document) => visibleTextElements(root)
                        .filter((el) => {
                            const target = clickableAncestor(el);
                            if (!target) return false;
                            const ariaPressed = target.getAttribute("aria-pressed") === "true";
                            const ariaSelected = target.getAttribute("aria-selected") === "true";
                            const dataState = (target.getAttribute("data-state") || "").toLowerCase() === "checked";
                            const dataSelected = target.getAttribute("data-selected") === "true";
                            const classSelected = /\b(active|selected|checked|on)\b/i.test(target.className || "");
                            const checkedInput = !!target.querySelector("input[type='radio']:checked, input[type='checkbox']:checked");
                            return ariaPressed || ariaSelected || dataState || dataSelected || checkedInput || classSelected;
                        });

                    const emulateClick = (el) => {
                        if (!el || !isVisible(el) || el.disabled) return false;
                        const common = { bubbles: true, cancelable: true, composed: true };
                        try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    };

                    const clickByText = (patterns, root = document) => {
                        const candidate = visibleTextElements(root).find((el) => matchesAny((el.textContent || "").trim(), patterns));
                        const target = clickableAncestor(candidate);
                        if (!target) return false;
                        return emulateClick(target);
                    };

                    const hasSelectedByText = (patterns, root = document) => selectedTextElements(root)
                        .some((el) => matchesAny((el.textContent || "").trim(), patterns));

                    const promptInput = document.querySelector("textarea[placeholder*='Type to imagine' i], input[placeholder*='Type to imagine' i], textarea[placeholder*='Type to customize this video' i], input[placeholder*='Type to customize this video' i], textarea[placeholder*='Type to customize video' i], input[placeholder*='Type to customize video' i], textarea[placeholder*='Customize video' i], input[placeholder*='Customize video' i], textarea[aria-label*='Make a video' i], input[aria-label*='Make a video' i], div.tiptap.ProseMirror[contenteditable='true'], [contenteditable='true'][aria-label*='Type to imagine' i], [contenteditable='true'][data-placeholder*='Type to imagine' i], [contenteditable='true'][aria-label*='Type to customize this video' i], [contenteditable='true'][data-placeholder*='Type to customize this video' i], [contenteditable='true'][aria-label*='Type to customize video' i], [contenteditable='true'][data-placeholder*='Type to customize video' i], [contenteditable='true'][aria-label*='Make a video' i], [contenteditable='true'][data-placeholder*='Customize video' i]");
                    const composer = (promptInput && (promptInput.closest("form") || promptInput.closest("main") || promptInput.closest("section"))) || document;
                    const findVisibleButtonByAriaLabel = (ariaLabel, root = document) => {
                        const candidates = [...root.querySelectorAll(`button[aria-label='${ariaLabel}']`)];
                        return candidates.find((el) => isVisible(el) && !el.disabled) || null;
                    };
                    const isOptionButtonSelected = (button) => {
                        if (!button) return false;
                        const ariaPressed = button.getAttribute("aria-pressed") === "true";
                        const ariaSelected = button.getAttribute("aria-selected") === "true";
                        const dataSelected = button.getAttribute("data-selected") === "true";
                        const dataState = (button.getAttribute("data-state") || "").toLowerCase();
                        if (ariaPressed || ariaSelected || dataSelected || dataState === "checked" || dataState === "active") return true;
                        if (/\b(active|selected|checked|on|text-fg-primary)\b/i.test(button.className || "")) return true;
                        const selectedFill = button.querySelector(".bg-primary:not([class*='bg-primary/'])");
                        return !!selectedFill;
                    };
                    const hasSelectedByAriaLabel = (ariaLabel, root = document) => {
                        const button = findVisibleButtonByAriaLabel(ariaLabel, root);
                        return isOptionButtonSelected(button);
                    };
                    const clickVisibleButtonByAriaLabel = (ariaLabel, root = document) => {
                        const button = findVisibleButtonByAriaLabel(ariaLabel, root) || findVisibleButtonByAriaLabel(ariaLabel, document);
                        if (!button) return false;
                        return emulateClick(button);
                    };

                    const desiredQuality = "{selected_quality_label}";
                    const desiredAspect = "{selected_aspect_ratio}";
                    const desiredDuration = "{selected_duration_label}";
                    const qualityPatterns = {
                        "480p": [/480\s*p/i, /854\s*[x×]\s*480/i],
                        "720p": [/720\s*p/i, /1280\s*[x×]\s*720/i]
                    };
                    const durationPatterns = {
                        "6s": [/^6\s*s(ec(onds?)?)?$/i],
                        "10s": [/^10\s*s(ec(onds?)?)?$/i]
                    };
                    const aspectPatterns = {
                        "2:3": [/^2\s*:\s*3$/i],
                        "3:2": [/^3\s*:\s*2$/i],
                        "1:1": [/^1\s*:\s*1$/i],
                        "9:16": [/^9\s*:\s*16$/i],
                        "16:9": [/^16\s*:\s*9$/i]
                    };

                    const requiredOptions = ["video", desiredQuality, desiredDuration, desiredAspect];
                    const optionsRequested = [];
                    const optionsApplied = [];

                    const applyOption = (name, patterns, ariaLabel) => {
                        const isAlreadySelected = (ariaLabel && (hasSelectedByAriaLabel(ariaLabel, composer) || hasSelectedByAriaLabel(ariaLabel)))
                            || hasSelectedByText(patterns, composer)
                            || hasSelectedByText(patterns);
                        if (isAlreadySelected) {
                            optionsApplied.push(`${name}(already-selected)`);
                            return;
                        }
                        const clicked = (ariaLabel && (clickVisibleButtonByAriaLabel(ariaLabel, composer) || clickVisibleButtonByAriaLabel(ariaLabel)))
                            || clickByText(patterns, composer)
                            || clickByText(patterns);
                        if (clicked) {
                            optionsRequested.push(name);
                        }
                        const isNowSelected = (ariaLabel && (hasSelectedByAriaLabel(ariaLabel, composer) || hasSelectedByAriaLabel(ariaLabel)))
                            || hasSelectedByText(patterns, composer)
                            || hasSelectedByText(patterns);
                        if (clicked && !isNowSelected) {
                            if (ariaLabel) {
                                clickVisibleButtonByAriaLabel(ariaLabel, composer) || clickVisibleButtonByAriaLabel(ariaLabel);
                            } else {
                                clickByText(patterns, composer) || clickByText(patterns);
                            }
                        }
                        const selected = (ariaLabel && (hasSelectedByAriaLabel(ariaLabel, composer) || hasSelectedByAriaLabel(ariaLabel)))
                            || hasSelectedByText(patterns, composer)
                            || hasSelectedByText(patterns);
                        if (selected) optionsApplied.push(name);
                    };

                    applyOption("video", [/^video$/i], null);
                    applyOption(desiredQuality, qualityPatterns[desiredQuality] || qualityPatterns["720p"], desiredQuality);
                    applyOption(desiredDuration, durationPatterns[desiredDuration] || durationPatterns["10s"], desiredDuration);
                    applyOption(desiredAspect, aspectPatterns[desiredAspect] || aspectPatterns["16:9"], desiredAspect);

                    const missingOptions = requiredOptions.filter((option) => {
                        const patterns = option === "video"
                            ? [/^video$/i]
                            : option === desiredQuality
                                ? (qualityPatterns[desiredQuality] || qualityPatterns["720p"])
                                : option === desiredDuration
                                    ? (durationPatterns[desiredDuration] || durationPatterns["10s"])
                                    : (aspectPatterns[desiredAspect] || aspectPatterns["16:9"]);
                        const ariaLabel = option === "video" ? null : option;
                        const selectedByAria = ariaLabel && (hasSelectedByAriaLabel(ariaLabel, composer) || hasSelectedByAriaLabel(ariaLabel));
                        return !(selectedByAria || hasSelectedByText(patterns, composer) || hasSelectedByText(patterns));
                    });

                    return {
                        ok: true,
                        requiredOptions,
                        optionsRequested,
                        optionsApplied,
                        missingOptions
                    };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """
        set_options_script = set_options_script.replace('"{selected_quality_label}"', json.dumps(selected_quality_label))
        set_options_script = set_options_script.replace('"{selected_duration_label}"', json.dumps(selected_duration_label))
        set_options_script = set_options_script.replace('"{selected_aspect_ratio}"', json.dumps(selected_aspect_ratio))

        close_options_script = r"""
            (() => {
                try {
                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const common = { bubbles: true, cancelable: true, composed: true };
                    const emulateClick = (el) => {
                        if (!el || !isVisible(el) || el.disabled) return false;
                        try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    };

                    let closed = false;
                    const modelTrigger = document.querySelector("#model-select-trigger");
                    if (modelTrigger) closed = emulateClick(modelTrigger);
                    if (!closed) {
                        const escEvent = new KeyboardEvent("keydown", { key: "Escape", code: "Escape", bubbles: true });
                        document.dispatchEvent(escEvent);
                        closed = true;
                    }

                    return { ok: true, closed };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """

        submit_script = r"""
            (() => {
                try {
                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const promptInput = document.querySelector("textarea[placeholder*='Type to imagine' i], input[placeholder*='Type to imagine' i], textarea[placeholder*='Type to customize this video' i], input[placeholder*='Type to customize this video' i], textarea[placeholder*='Type to customize video' i], input[placeholder*='Type to customize video' i], textarea[placeholder*='Customize video' i], input[placeholder*='Customize video' i], textarea[aria-label*='Make a video' i], input[aria-label*='Make a video' i], div.tiptap.ProseMirror[contenteditable='true'], [contenteditable='true'][aria-label*='Type to imagine' i], [contenteditable='true'][data-placeholder*='Type to imagine' i], [contenteditable='true'][aria-label*='Type to customize this video' i], [contenteditable='true'][data-placeholder*='Type to customize this video' i], [contenteditable='true'][aria-label*='Type to customize video' i], [contenteditable='true'][data-placeholder*='Type to customize video' i], [contenteditable='true'][aria-label*='Make a video' i], [contenteditable='true'][data-placeholder*='Customize video' i]");

                    const submitSelectors = [
                        "button[type='submit'][aria-label='Submit']",
                        "button[aria-label='Submit'][type='submit']",
                        "button[type='submit']",
                        "button[aria-label='Submit']"
                    ];

                    const submitCandidates = [];
                    const collect = (root) => {
                        if (!root || typeof root.querySelectorAll !== "function") return;
                        submitSelectors.forEach((selector) => {
                            const matches = root.querySelectorAll(selector);
                            for (let i = 0; i < matches.length; i += 1) submitCandidates.push(matches[i]);
                        });
                    };

                    const composerRoot = (promptInput && typeof promptInput.closest === "function")
                        ? (promptInput.closest("form") || promptInput.closest("main") || promptInput.closest("section") || promptInput.parentElement)
                        : null;

                    collect(composerRoot);
                    collect(document);

                    const uniqueCandidates = [...new Set(submitCandidates)];
                    const submitButton = uniqueCandidates.find((el) => isVisible(el));

                    const form = (submitButton && submitButton.form)
                        || (promptInput && typeof promptInput.closest === "function" ? promptInput.closest("form") : null)
                        || (composerRoot && typeof composerRoot.closest === "function" ? composerRoot.closest("form") : null)
                        || document.querySelector("form");

                    if (!submitButton && !form) return { ok: false, error: "Submit button/form not found" };

                    if (submitButton && submitButton.disabled) {
                        submitButton.disabled = false;
                        submitButton.removeAttribute("disabled");
                        submitButton.setAttribute("aria-disabled", "false");
                    }

                    const common = { bubbles: true, cancelable: true, composed: true };
                    const emulateClick = (el) => {
                        try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                    };

                    let clicked = false;
                    

                    let formSubmitted = false;
                    if (form) {
                        const ev = new Event("submit", { bubbles: true, cancelable: true });
                        formSubmitted = form.dispatchEvent(ev); // lets React handlers run
                        formSubmitted = true;
                    }

                    return {
                        ok: true,
                        submitted: clicked || formSubmitted,
                        doubleClicked: !!submitButton,
                        formSubmitted,
                        forceEnabled: !!submitButton,
                        buttonText: submitButton ? (submitButton.textContent || "").trim() : "",
                        buttonAriaLabel: submitButton ? (submitButton.getAttribute("aria-label") || "") : ""
                    };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """

        def _continue_after_options_set(result):
            if not isinstance(result, dict) or not result.get("ok"):
                options_error = result.get("error") if isinstance(result, dict) else result
                self._append_log(
                    f"WARNING: Option application script reported an error for variant {variant}: {options_error!r}. Continuing."
                )

            options_requested = result.get("optionsRequested") if isinstance(result, dict) else []
            options_applied = result.get("optionsApplied") if isinstance(result, dict) else []
            requested_summary = ", ".join(options_requested) if options_requested else "none"
            applied_summary = ", ".join(options_applied) if options_applied else "none detected"
            self._append_log(
                f"Options staged for variant {variant}; options requested: {requested_summary}; options applied markers: {applied_summary}."
            )

            def _continue_after_options_close(close_result):
                if not isinstance(close_result, dict) or not close_result.get("ok"):
                    close_error = close_result.get("error") if isinstance(close_result, dict) else close_result
                    self._append_log(
                        f"WARNING: Closing options window reported an error for variant {variant}: {close_error!r}. Continuing."
                    )

                self._append_log(f"Options window closed for variant {variant}; submitting after {action_delay_ms}ms delay.")

                def after_delayed_submit(submit_result):
                    if not isinstance(submit_result, dict) or not submit_result.get("ok"):
                        error_detail = submit_result.get("error") if isinstance(submit_result, dict) else submit_result
                        self._append_log(
                            f"WARNING: Manual submit script reported an issue for variant {variant}: {error_detail!r}. Continuing to download polling."
                        )

                    self._append_log(
                        "Submitted manual variant "
                        f"{variant} after prompt/options staged delays (double-click submit); "
                        "waiting for generation to auto-download."
                    )
                    self._trigger_browser_video_download(variant)

                QTimer.singleShot(action_delay_ms, lambda: self.browser.page().runJavaScript(submit_script, after_delayed_submit))

            QTimer.singleShot(
                action_delay_ms,
                lambda: self.browser.page().runJavaScript(close_options_script, _continue_after_options_close),
            )

        def _continue_after_options_open(open_result):
            open_ok = isinstance(open_result, dict) and bool(open_result.get("ok"))
            panel_visible = isinstance(open_result, dict) and bool(open_result.get("panelVisible"))
            if not open_ok:
                open_error = open_result.get("error") if isinstance(open_result, dict) else open_result
                self._append_log(
                    f"WARNING: Opening options window returned an error for variant {variant}: {open_error!r}. "
                    "Continuing anyway."
                )
            elif panel_visible:
                self._append_log(f"Options panel appears visible for variant {variant}; proceeding to option selection.")

            self._append_log(f"Options window opened for variant {variant}; setting options after {action_delay_ms}ms delay.")
            QTimer.singleShot(
                action_delay_ms,
                lambda: self.browser.page().runJavaScript(set_options_script, _continue_after_options_set),
            )

        def _run_continue_mode_submit() -> None:
            continue_submit_script = r"""
                (() => {
                    try {
                        const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                        const common = { bubbles: true, cancelable: true, composed: true };
                        const emulateClick = (el) => {
                            if (!el || !isVisible(el) || el.disabled) return false;
                            try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                            el.dispatchEvent(new MouseEvent("mousedown", common));
                            try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                            el.dispatchEvent(new MouseEvent("mouseup", common));
                            el.dispatchEvent(new MouseEvent("click", common));
                            return true;
                        };

                        const buttons = [...document.querySelectorAll("button, [role='button']")].filter((el) => isVisible(el));
                        const matchers = [
                            /make\s*video/i,
                            /generate/i,
                            /submit/i,
                        ];
                        let actionButton = null;
                        for (const matcher of matchers) {
                            actionButton = buttons.find((btn) => matcher.test((btn.getAttribute("aria-label") || btn.textContent || "").trim()));
                            if (actionButton) break;
                        }
                        const clicked = actionButton ? emulateClick(actionButton) : false;
                        return {
                            ok: clicked,
                            buttonText: actionButton ? (actionButton.textContent || "").trim() : "",
                            buttonAriaLabel: actionButton ? (actionButton.getAttribute("aria-label") || "") : "",
                            error: clicked ? "" : "Could not find/click Make video button",
                        };
                    } catch (err) {
                        return { ok: false, error: String(err && err.stack ? err.stack : err) };
                    }
                })()
            """

            def _after_continue_submit(submit_result):
                if not isinstance(submit_result, dict) or not submit_result.get("ok"):
                    error_detail = submit_result.get("error") if isinstance(submit_result, dict) else submit_result
                    if error_detail not in (None, "", "callback-empty"):
                        self._append_log(
                            f"Continue-mode submit for variant {variant} reported an issue: {error_detail!r}; continuing to video download polling."
                        )
                else:
                    button_label = submit_result.get("buttonAriaLabel") or submit_result.get("buttonText") or "Make video"
                    self._append_log(f"Continue-mode submit clicked '{button_label}' for variant {variant}.")
                self._trigger_browser_video_download(variant, allow_make_video_click=False)

            QTimer.singleShot(action_delay_ms, lambda: self.browser.page().runJavaScript(continue_submit_script, _after_continue_submit))

        def after_submit(result):
            fill_ok = isinstance(result, dict) and bool(result.get("ok"))
            if not fill_ok:
                error_detail = result.get("error") if isinstance(result, dict) else result
                if error_detail not in (None, "", "callback-empty"):
                    self._append_log(
                        f"Manual prompt fill reported an issue for variant {variant}: {error_detail!r}. "
                        "Verifying current field content before continuing."
                    )

            def _after_verify_prompt(verify_result):
                verify_ok = isinstance(verify_result, dict) and bool(verify_result.get("ok"))
                if not (fill_ok or verify_ok):
                    verify_error = verify_result.get("error") if isinstance(verify_result, dict) else verify_result
                    if verify_error not in (None, "", "callback-empty"):
                        self._append_log(
                            f"Prompt fill verification did not confirm content for variant {variant}: {verify_error!r}. "
                            "Continuing with option selection and forced submit anyway."
                        )
                if self.continue_from_frame_active:
                    _run_continue_mode_submit()
                else:
                    QTimer.singleShot(
                        action_delay_ms,
                        lambda: self.browser.page().runJavaScript(open_options_script, _continue_after_options_open),
                    )

            if fill_ok:
                _after_verify_prompt({"ok": True})
                return

            QTimer.singleShot(
                250,
                lambda: self.browser.page().runJavaScript(verify_prompt_script, _after_verify_prompt),
            )

        self.browser.page().runJavaScript(script, after_submit)

    def _trigger_browser_video_download(self, variant: int, allow_make_video_click: bool = True) -> None:
        self.pending_manual_download_type = "video"
        self.manual_download_deadline = time.time() + 420
        self.manual_download_click_sent = False
        self.manual_video_start_click_sent = False
        self.manual_video_make_click_fallback_used = False
        self.manual_video_allow_make_click = allow_make_video_click
        self.manual_download_in_progress = False
        self.manual_download_started_at = time.time()
        self.manual_download_poll_timer.start(0)

    def _poll_for_manual_video(self) -> None:
        if self.stop_all_requested:
            self._append_log("Stop-all flag active; skipping queued job activity.")
            return

        variant = self.pending_manual_variant_for_download
        if variant is None:
            return

        deadline = self.manual_download_deadline or 0
        if time.time() > deadline:
            self.pending_manual_variant_for_download = None
            self.manual_download_click_sent = False
            self.manual_video_start_click_sent = False
            self.manual_video_make_click_fallback_used = False
            self.manual_video_allow_make_click = True
            self.manual_download_in_progress = False
            self.manual_download_started_at = None
            self.manual_download_deadline = None
            self._append_log(f"ERROR: Variant {variant} did not produce a downloadable video in time.")
            if self.continue_from_frame_active:
                self._append_log("Continue-from-last-frame stopped because download polling timed out.")
                self.continue_from_frame_active = False
                self.continue_from_frame_target_count = 0
                self.continue_from_frame_completed = 0
                self.continue_from_frame_prompt = ""
            return

        allow_make_video_click = "true" if (self.manual_video_allow_make_click and not self.manual_video_start_click_sent) else "false"
        script = f"""
            (() => {{
                const allowMakeVideoClick = {allow_make_video_click};
                const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                const common = {{ bubbles: true, cancelable: true, composed: true }};
                const emulateClick = (el) => {{
                    if (!el || !isVisible(el) || el.disabled) return false;
                    try {{
                        el.dispatchEvent(new PointerEvent("pointerdown", common));
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        el.dispatchEvent(new PointerEvent("pointerup", common));
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    }} catch (_) {{
                        try {{
                            el.click();
                            return true;
                        }} catch (__){{
                            return false;
                        }}
                    }}
                }};
                const percentNode = [...document.querySelectorAll("div .tabular-nums, div.tabular-nums")]
                    .find((el) => isVisible(el) && /^\\d{{1,3}}%$/.test((el.textContent || "").trim()));
                if (percentNode) {{
                    return {{ status: "progress", progressText: (percentNode.textContent || "").trim() }};
                }}

                const redoButton = [...document.querySelectorAll("button")]
                    .find((btn) => isVisible(btn) && !btn.disabled && /redo/i.test((btn.textContent || "").trim()));

                const makeVideoButton = [...document.querySelectorAll("button")]
                    .find((btn) => {{
                        if (!isVisible(btn) || btn.disabled) return false;
                        const label = (btn.getAttribute("aria-label") || btn.textContent || "").trim();
                        return /make\\s+video/i.test(label);
                    }});

                const downloadCandidates = [...document.querySelectorAll("button[aria-label='Download'][type='button'], button[aria-label='Download']")]
                    .filter((btn) => isVisible(btn) && !btn.disabled);
                const makeVideoContainer = makeVideoButton
                    ? (makeVideoButton.closest("form") || makeVideoButton.closest("section") || makeVideoButton.closest("main") || makeVideoButton.parentElement)
                    : null;
                let downloadButton = downloadCandidates.find((btn) => makeVideoContainer && makeVideoContainer.contains(btn) && btn !== makeVideoButton);
                if (!downloadButton) downloadButton = downloadCandidates[0] || null;

                if (downloadButton && (redoButton || !makeVideoButton)) {{
                    return {{
                        status: emulateClick(downloadButton) ? "download-clicked" : "download-visible",
                    }};
                }}

                if (makeVideoButton && !redoButton) {{
                    const buttonLabel = (makeVideoButton.getAttribute("aria-label") || makeVideoButton.textContent || "").trim();
                    if (!allowMakeVideoClick) {{
                        return {{ status: "make-video-awaiting-progress", buttonLabel }};
                    }}
                    return {{
                        status: emulateClick(makeVideoButton) ? "make-video-clicked" : "make-video-visible",
                        buttonLabel,
                    }};
                }}

                if (!redoButton) {{
                    return {{ status: "waiting-for-redo" }};
                }}

                const video = document.querySelector("video");
                const source = document.querySelector("video source");
                const src = (video && (video.currentSrc || video.src)) || (source && source.src) || "";
                const enoughData = !!(video && video.readyState >= 3 && Number(video.duration || 0) > 0);
                return {{
                    status: src ? (enoughData ? "video-src-ready" : "video-buffering") : "waiting",
                    src,
                    readyState: video ? video.readyState : 0,
                    duration: video ? Number(video.duration || 0) : 0,
                }};
            }})()
        """

        def after_poll(result):
            current_variant = self.pending_manual_variant_for_download
            if current_variant is None:
                return

            if not isinstance(result, dict):
                self.manual_download_poll_timer.start(3000)
                return

            status = result.get("status", "waiting")
            progress_text = (result.get("progressText") or "").strip()

            if status == "progress":
                self.manual_video_start_click_sent = True
                if progress_text:
                    self._append_log(f"Variant {current_variant} still rendering: {progress_text}")
                self.manual_download_poll_timer.start(3000)
                return

            if status == "make-video-clicked":
                label = (result.get("buttonLabel") or "Make video").strip()
                self._append_log(f"Variant {current_variant}: clicked '{label}' to start video generation.")
                self.manual_video_start_click_sent = True
                self.manual_video_make_click_fallback_used = True
                self.manual_download_poll_timer.start(3000)
                return

            if status == "make-video-awaiting-progress":
                self.manual_download_poll_timer.start(3000)
                return

            if status == "make-video-visible":
                self._append_log(f"Variant {current_variant}: '{result.get('buttonLabel') or 'Make video'}' is visible but click did not register; retrying.")
                self.manual_download_poll_timer.start(2000)
                return

            if status == "waiting-for-redo":
                self.manual_download_poll_timer.start(3000)
                return

            if status == "download-clicked":
                if not self.manual_download_click_sent:
                    self._append_log(f"Variant {current_variant} appears ready; clicked in-page Download button.")
                    self.manual_download_click_sent = True
                    self.manual_download_in_progress = True
                self.manual_download_poll_timer.start(3000)
                return

            src = result.get("src") or ""
            if status == "video-buffering":
                self.manual_download_poll_timer.start(3000)
                return

            min_wait_elapsed = self.manual_download_started_at is not None and (time.time() - self.manual_download_started_at) >= 8
            if status != "video-src-ready" or not src or not min_wait_elapsed:
                self.manual_download_poll_timer.start(3000)
                return

            trigger_download_script = f"""
                (() => {{
                    const src = {src!r};
                    const a = document.createElement("a");
                    a.href = src;
                    a.download = `grok_manual_variant_{current_variant}_${{Date.now()}}.mp4`;
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    return true;
                }})()
            """
            if not self.manual_download_click_sent:
                self.browser.page().runJavaScript(trigger_download_script)
                self._append_log(f"Variant {current_variant} video detected; browser download requested from video source.")
                self.manual_download_click_sent = True
                self.manual_download_in_progress = True

            self.manual_download_poll_timer.start(3000)

        self.browser.page().runJavaScript(script, after_poll)

    def _on_browser_download_requested(self, download) -> None:
        variant = self.pending_manual_variant_for_download
        if variant is None:
            return
        if self.manual_download_in_progress:
            self.manual_download_in_progress = False
        elif self.manual_download_click_sent:
            self._append_log("Ignoring duplicate browser download request for current manual variant.")
            download.cancel()
            return

        download_type = self.pending_manual_download_type or "video"
        extension = self._resolve_download_extension(download, download_type)
        filename = f"{download_type}_{int(time.time() * 1000)}_manual_v{variant}.{extension}"
        download.setDownloadDirectory(str(self.download_dir))
        download.setDownloadFileName(filename)
        download.accept()
        self._append_log(f"Downloading manual {download_type} variant {variant} to {self.download_dir / filename}")

        def on_state_changed(state):
            if state == download.DownloadState.DownloadCompleted:
                video_path = self.download_dir / filename
                video_size = video_path.stat().st_size if video_path.exists() else 0
                if download_type == "image":
                    self._append_log(f"Saved image: {video_path}")
                    self.pending_manual_variant_for_download = None
                    self.pending_manual_download_type = None
                    self.pending_manual_image_prompt = None
                    self.manual_image_pick_clicked = False
                    self.manual_image_video_submit_sent = False
                    self.manual_image_pick_retry_count = 0
                    self.manual_image_submit_retry_count = 0
                    self.manual_download_click_sent = False
                    self.manual_video_start_click_sent = False
                    self.manual_video_make_click_fallback_used = False
                    self.manual_video_allow_make_click = True
                    self.manual_download_in_progress = False
                    self.manual_download_started_at = None
                    self.manual_download_deadline = None
                    self._submit_next_manual_image_variant()
                    return

                image_extensions = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}
                if download_type == "video" and extension.lower() in image_extensions:
                    self._append_log(
                        f"WARNING: Variant {variant}: clicked a non-video download target ({extension}); retrying with video download button."
                    )
                    if video_path.exists():
                        video_path.unlink(missing_ok=True)
                    self.manual_download_click_sent = False
                    self.manual_download_in_progress = False
                    self.manual_download_started_at = time.time()
                    self.manual_download_poll_timer.start(1200)
                    return

                if video_size < MIN_VALID_VIDEO_BYTES:
                    self._append_log(
                        f"WARNING: Downloaded manual variant {variant} is only {video_size} bytes (< 1MB)."
                    )
                    self.pending_manual_variant_for_download = None
                    self.pending_manual_download_type = None
                    self.pending_manual_image_prompt = None
                    self.manual_image_pick_clicked = False
                    self.manual_image_video_submit_sent = False
                    self.manual_image_pick_retry_count = 0
                    self.manual_image_submit_retry_count = 0
                    self.manual_download_click_sent = False
                    self.manual_video_start_click_sent = False
                    self.manual_video_make_click_fallback_used = False
                    self.manual_video_allow_make_click = True
                    self.manual_download_in_progress = False
                    self.manual_download_started_at = None
                    self.manual_download_deadline = None

                    if self.continue_from_frame_active:
                        self._retry_continue_after_small_download(variant)
                    else:
                        self._append_log(
                            "WARNING: Undersized manual download detected outside continue-from-last-frame mode; "
                            "please use 'Continue from Last Frame' to regenerate from the extracted frame."
                        )
                    return

                self.on_video_finished(
                    {
                        "title": f"Manual Browser Video {variant}",
                        "prompt": self.manual_prompt.toPlainText().strip(),
                        "resolution": "web",
                        "video_file_path": str(video_path),
                        "source_url": "browser-session",
                    }
                )
                self._append_log("Download complete; returning embedded browser to grok.com/imagine.")
                QTimer.singleShot(0, self.show_browser_page)
                self.pending_manual_variant_for_download = None
                self.pending_manual_download_type = None
                self.pending_manual_image_prompt = None
                self.manual_image_pick_clicked = False
                self.manual_image_video_submit_sent = False
                self.manual_image_pick_retry_count = 0
                self.manual_image_submit_retry_count = 0
                self.manual_download_click_sent = False
                self.manual_video_start_click_sent = False
                self.manual_video_make_click_fallback_used = False
                self.manual_video_allow_make_click = True
                self.manual_download_in_progress = False
                self.manual_download_started_at = None
                self.manual_download_deadline = None
                if self.continue_from_frame_active:
                    self.continue_from_frame_completed += 1
                    if self.continue_from_frame_completed < self.continue_from_frame_target_count:
                        QTimer.singleShot(800, self._start_continue_iteration)
                    else:
                        self._append_log("Continue workflow complete.")
                        self.continue_from_frame_active = False
                        self.continue_from_frame_target_count = 0
                        self.continue_from_frame_completed = 0
                        self.continue_from_frame_prompt = ""
                        self.continue_from_frame_current_source_video = ""
                        self.continue_from_frame_seed_image_path = None
                else:
                    self._submit_next_manual_variant()
            elif state == download.DownloadState.DownloadInterrupted:
                self._append_log(f"ERROR: Download interrupted for manual variant {variant}.")
                self.pending_manual_variant_for_download = None
                self.pending_manual_download_type = None
                self.pending_manual_image_prompt = None
                self.manual_image_pick_clicked = False
                self.manual_image_video_submit_sent = False
                self.manual_image_pick_retry_count = 0
                self.manual_image_submit_retry_count = 0
                self.manual_download_click_sent = False
                self.manual_video_start_click_sent = False
                self.manual_video_make_click_fallback_used = False
                self.manual_video_allow_make_click = True
                self.manual_download_in_progress = False
                self.manual_download_started_at = None
                self.manual_download_deadline = None
                self.continue_from_frame_active = False
                self.continue_from_frame_target_count = 0
                self.continue_from_frame_completed = 0
                self.continue_from_frame_prompt = ""
                self.continue_from_frame_current_source_video = ""
                self.continue_from_frame_seed_image_path = None

        download.stateChanged.connect(on_state_changed)

    def stop_all_jobs(self) -> None:
        self.stop_all_requested = True
        self.manual_generation_queue.clear()
        self.manual_image_generation_queue.clear()
        self.manual_download_poll_timer.stop()
        self.continue_from_frame_reload_timeout_timer.stop()
        self.pending_manual_variant_for_download = None
        self.pending_manual_download_type = None
        self.pending_manual_image_prompt = None
        self.manual_image_pick_clicked = False
        self.manual_image_video_submit_sent = False
        self.manual_image_pick_retry_count = 0
        self.manual_image_submit_retry_count = 0
        self.manual_image_submit_token += 1
        self.manual_download_click_sent = False
        self.manual_video_start_click_sent = False
        self.manual_video_make_click_fallback_used = False
        self.manual_video_allow_make_click = True
        self.manual_download_in_progress = False
        self.manual_download_started_at = None
        self.manual_download_deadline = None

        self.continue_from_frame_active = False
        self.continue_from_frame_target_count = 0
        self.continue_from_frame_completed = 0
        self.continue_from_frame_prompt = ""
        self.continue_from_frame_current_source_video = ""
        self.continue_from_frame_seed_image_path = None
        self.continue_from_frame_waiting_for_reload = False

        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self._append_log("Stop requested: API generation worker will stop after the current request completes.")
        self._append_log("Stop all requested: cleared queued manual image/video jobs and halted polling timers.")

    def _build_video_label(self, video: dict) -> str:
        title = str(video.get("title") or "Video")
        resolution = str(video.get("resolution") or "unknown")
        return f"{title} ({resolution})"

    def _thumbnail_for_video(self, video_path: str) -> QIcon:
        source_path = Path(video_path)
        if not source_path.exists():
            return QIcon()

        safe_name = f"thumb_{source_path.stem}_{abs(hash(str(source_path.resolve()))) % 10**10}.jpg"
        thumb_path = THUMBNAILS_DIR / safe_name
        if not thumb_path.exists():
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        "00:00:01.000",
                        "-i",
                        str(source_path),
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale=144:-2",
                        str(thumb_path),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except Exception:
                return QIcon()

        pixmap = QPixmap(str(thumb_path))
        if pixmap.isNull():
            return QIcon()
        return QIcon(pixmap)

    def _refresh_video_picker(self, selected_index: int = -1) -> None:
        self.video_picker.blockSignals(True)
        self.video_picker.clear()
        for video in self.videos:
            self.video_picker.addItem(self._thumbnail_for_video(video["video_file_path"]), self._build_video_label(video))
        self.video_picker.blockSignals(False)

        if not self.videos:
            return

        if selected_index < 0 or selected_index >= len(self.videos):
            selected_index = len(self.videos) - 1
        self.video_picker.setCurrentIndex(selected_index)

    def on_video_finished(self, video: dict) -> None:
        self.videos.append(video)
        self._refresh_video_picker(selected_index=len(self.videos) - 1)
        self._append_log(f"Saved: {video['video_file_path']}")

    def open_local_video(self) -> None:
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            str(self.download_dir),
            "Video Files (*.mp4 *.mov *.mkv *.webm *.avi)",
        )
        if not video_path:
            return

        resolution = "local"
        try:
            info = self._probe_video_stream_info(Path(video_path))
            resolution = f"{info['width']}x{info['height']}"
        except Exception:
            pass

        loaded_video = {
            "title": f"Opened: {Path(video_path).stem}",
            "prompt": "opened-local-video",
            "resolution": resolution,
            "video_file_path": video_path,
            "source_url": "local-open",
        }
        self.on_video_finished(loaded_video)

    def move_selected_video(self, delta: int) -> None:
        index = self.video_picker.currentIndex()
        if index < 0 or index >= len(self.videos):
            return

        target = index + delta
        if target < 0 or target >= len(self.videos):
            return

        self.videos[index], self.videos[target] = self.videos[target], self.videos[index]
        self._refresh_video_picker(selected_index=target)
        self._append_log(f"Reordered videos: moved item to position {target + 1}.")

    def remove_selected_video(self) -> None:
        index = self.video_picker.currentIndex()
        if index < 0 or index >= len(self.videos):
            return

        removed = self.videos.pop(index)
        if not self.videos:
            self._refresh_video_picker(selected_index=-1)
            self.player.stop()
            self.player.setSource(QUrl())
            self.preview_seek_slider.blockSignals(True)
            self.preview_seek_slider.setRange(0, 0)
            self.preview_seek_slider.setValue(0)
            self.preview_seek_slider.blockSignals(False)
            self.preview_position_label.setText("00:00 / 00:00")
        else:
            self._refresh_video_picker(selected_index=min(index, len(self.videos) - 1))

        self._append_log(f"Removed video from list: {removed.get('video_file_path', 'unknown')}")

    def _format_time_ms(self, ms: int) -> str:
        total_seconds = max(0, int(ms // 1000))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _on_preview_position_changed(self, position: int) -> None:
        self.preview_seek_slider.blockSignals(True)
        self.preview_seek_slider.setValue(position)
        self.preview_seek_slider.blockSignals(False)
        self.preview_position_label.setText(
            f"{self._format_time_ms(position)} / {self._format_time_ms(self.player.duration())}"
        )

    def _on_preview_duration_changed(self, duration: int) -> None:
        self.preview_seek_slider.blockSignals(True)
        self.preview_seek_slider.setRange(0, max(0, duration))
        self.preview_seek_slider.blockSignals(False)
        self.preview_position_label.setText(
            f"{self._format_time_ms(self.player.position())} / {self._format_time_ms(duration)}"
        )

    def seek_preview(self, position: int) -> None:
        self.player.setPosition(max(0, int(position)))

    def _ensure_preview_fullscreen_overlay(self) -> None:
        if self.preview_fullscreen_overlay_btn is not None:
            return

        overlay_btn = QPushButton("🗗 Exit Fullscreen")
        overlay_btn.setToolTip("Exit fullscreen preview")
        overlay_btn.setStyleSheet(
            "background-color: rgba(15, 24, 40, 0.85); color: white; font-weight: 700;"
            "border: 1px solid #4fc3f7; border-radius: 8px; padding: 8px 12px;"
        )
        overlay_btn.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        overlay_btn.setWindowFlag(Qt.FramelessWindowHint, True)
        overlay_btn.clicked.connect(self.toggle_preview_fullscreen)
        self.preview_fullscreen_overlay_btn = overlay_btn

    def _position_preview_fullscreen_overlay(self) -> None:
        if self.preview_fullscreen_overlay_btn is None:
            return

        self.preview_fullscreen_overlay_btn.adjustSize()
        preview_frame = self.preview.frameGeometry()
        x = preview_frame.right() - self.preview_fullscreen_overlay_btn.width() - 24
        y = preview_frame.top() + 20
        self.preview_fullscreen_overlay_btn.move(max(10, x), max(10, y))

    def _on_preview_fullscreen_changed(self, fullscreen: bool) -> None:
        self.preview_fullscreen_btn.setText("🗗 Exit Fullscreen" if fullscreen else "⛶ Fullscreen")

        if fullscreen:
            self._ensure_preview_fullscreen_overlay()
            self._position_preview_fullscreen_overlay()
            if self.preview_fullscreen_overlay_btn is not None:
                self.preview_fullscreen_overlay_btn.show()
                self.preview_fullscreen_overlay_btn.raise_()
            self._append_log("Preview entered fullscreen mode.")
            return

        if self.preview_fullscreen_overlay_btn is not None:
            self.preview_fullscreen_overlay_btn.hide()
        self._append_log("Preview exited fullscreen mode.")

    def toggle_preview_fullscreen(self) -> None:
        self.preview.setFullScreen(not self.preview.isFullScreen())

    def on_generation_error(self, error: str) -> None:
        self._append_log(f"ERROR: {error}")
        QMessageBox.critical(self, "Generation Failed", error)

    def show_selected_video(self, index: int) -> None:
        if index < 0 or index >= len(self.videos):
            return
        video = self.videos[index]
        self._preview_video(video["video_file_path"])

    def _preview_video(self, file_path: str) -> None:
        self.player.setSource(QUrl.fromLocalFile(file_path))
        self.player.play()
        self._append_log(f"Selected video for preview: {file_path}")

    def play_preview(self) -> None:
        if self.player.source().isEmpty():
            self._append_log("Preview play requested, but no video is currently loaded.")
            return
        self.player.play()
        self._append_log("Preview playback started.")

    def stop_preview(self) -> None:
        self.player.stop()
        self._append_log("Preview playback stopped.")

    def _set_preview_muted(self, muted: bool) -> None:
        self.preview_muted = bool(muted)
        self.audio_output.setMuted(self.preview_muted)
        self._append_log(f"Preview audio {'muted' if self.preview_muted else 'unmuted'}.")

    def _set_preview_volume(self, value: int) -> None:
        self.preview_volume = int(value)
        self.audio_output.setVolume(self.preview_volume / 100)
        self._append_log(f"Preview volume set to {self.preview_volume}%.")

    def _toggle_prompt_source_fields(self) -> None:
        source = self.prompt_source.currentData() if hasattr(self, "prompt_source") else "manual"
        is_manual = source == "manual"
        is_openai = source == "openai"
        self.manual_prompt.setEnabled(is_manual)
        self.openai_api_key.setEnabled(is_openai)
        self.openai_chat_model.setEnabled(is_openai)
        self.chat_model.setEnabled(source == "grok")
        self.generate_btn.setText("📝 Populate Video Prompt" if is_manual else "🎬 Generate Video")
        self.generate_image_btn.setText("🖼️ Populate Image Prompt")
        self.generate_image_btn.setEnabled(is_manual)

    def _resolve_download_extension(self, download, download_type: str) -> str:
        suggested = ""
        try:
            suggested = download.downloadFileName() or ""
        except Exception:
            suggested = ""

        if "." in suggested:
            return suggested.rsplit(".", 1)[-1].lower()

        try:
            parsed = urlparse(download.url().toString())
            path_name = Path(parsed.path).name
            if "." in path_name:
                return path_name.rsplit(".", 1)[-1].lower()
        except Exception:
            pass

        return "png" if download_type == "image" else "mp4"

    def _extract_last_frame(self, video_path: str) -> Path | None:
        frame_path = self.download_dir / f"last_frame_{int(time.time() * 1000)}.png"
        self._append_log(f"Starting ffmpeg last-frame extraction from: {video_path}")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-sseof",
                    "-0.1",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    str(frame_path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            QMessageBox.critical(self, "ffmpeg Missing", "ffmpeg is required for frame extraction but was not found in PATH.")
            return None
        except subprocess.CalledProcessError as exc:
            QMessageBox.critical(self, "Frame Extraction Failed", exc.stderr[-800:] or "ffmpeg failed.")
            return None

        self._append_log(f"Completed ffmpeg last-frame extraction: {frame_path}")
        return frame_path

    def _copy_image_to_clipboard(self, frame_path: Path) -> bool:
        self._append_log(f"Copying extracted frame to clipboard: {frame_path}")

        image = QImage(str(frame_path))
        if image.isNull():
            QMessageBox.critical(self, "Frame Extraction Failed", "Frame image could not be loaded.")
            return False

        mime = QMimeData()
        mime.setImageData(image)
        mime.setText(str(frame_path))
        QGuiApplication.clipboard().setMimeData(mime)
        self._append_log("Clipboard image copy completed.")
        return True

    def _upload_frame_into_grok(self, frame_path: Path, on_uploaded=None) -> None:
        import base64

        self._append_log(f"Starting browser-side image paste for frame: {frame_path.name}")
        frame_base64 = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        upload_script = r"""
            (() => {
                const base64Data = __FRAME_BASE64__;
                const fileName = __FRAME_NAME__;
                const selectors = [
                    "textarea[placeholder*='Type to imagine' i]",
                    "input[placeholder*='Type to imagine' i]",
                    "textarea[placeholder*='Type to customize this video' i]",
                    "input[placeholder*='Type to customize this video' i]",
                    "textarea[placeholder*='Type to customize video' i]",
                    "input[placeholder*='Type to customize video' i]",
                    "textarea[placeholder*='Customize video' i]",
                    "input[placeholder*='Customize video' i]",
                    "textarea[aria-label*='Make a video' i]",
                    "input[aria-label*='Make a video' i]",
                    "div.tiptap.ProseMirror[contenteditable='true']",
                    "[contenteditable='true'][aria-label*='Type to imagine' i]",
                    "[contenteditable='true'][data-placeholder*='Type to imagine' i]",
                    "[contenteditable='true'][aria-label*='Type to customize this video' i]",
                    "[contenteditable='true'][data-placeholder*='Type to customize this video' i]",
                    "[contenteditable='true'][aria-label*='Type to customize video' i]",
                    "[contenteditable='true'][data-placeholder*='Type to customize video' i]",
                    "[contenteditable='true'][aria-label*='Make a video' i]",
                    "[contenteditable='true'][data-placeholder*='Customize video' i]"
                ];
                const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                const setInputFiles = (input, files) => {
                    try {
                        const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "files");
                        if (descriptor && typeof descriptor.set === "function") {
                            descriptor.set.call(input, files);
                            return true;
                        }
                    } catch (_) {}

                    try {
                        Object.defineProperty(input, "files", { value: files, configurable: true });
                        return true;
                    } catch (_) {
                        return false;
                    }
                };

                const dispatchFileEvents = (target, dt) => {
                    try {
                        target.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
                        target.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
                    } catch (_) {}

                    ["dragenter", "dragover", "drop"].forEach((eventName) => {
                        try {
                            target.dispatchEvent(new DragEvent(eventName, { bubbles: true, cancelable: true, dataTransfer: dt }));
                        } catch (_) {}
                    });

                    try {
                        const pasteEvent = new ClipboardEvent("paste", { bubbles: true, cancelable: true, clipboardData: dt });
                        target.dispatchEvent(pasteEvent);
                    } catch (_) {}
                };

                for (const selector of selectors) {
                    const node = [...document.querySelectorAll(selector)].find((el) => isVisible(el));
                    if (node) {
                        node.focus();
                        const binary = atob(base64Data);
                        const bytes = new Uint8Array(binary.length);
                        for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
                        const file = new File([bytes], fileName, { type: "image/png" });

                        const dt = new DataTransfer();
                        dt.items.add(file);

                        const queryBar = node.closest(".query-bar") || node.closest("form") || node.parentElement;
                        const promptRoot = queryBar?.parentElement || node.parentElement;
                        const scopedInputs = [
                            ...(promptRoot ? promptRoot.querySelectorAll("input[type='file']") : []),
                            ...(queryBar ? queryBar.querySelectorAll("input[type='file']") : []),
                        ];
                        const fileInputs = scopedInputs.length
                            ? [...new Set(scopedInputs)]
                            : [...document.querySelectorAll("input[type='file']")];

                        let populatedInputs = 0;
                        for (const input of fileInputs) {
                            try {
                                if (!setInputFiles(input, dt.files)) continue;
                                dispatchFileEvents(input, dt);
                                populatedInputs += 1;
                            } catch (_) {}
                        }

                        dispatchFileEvents(node, dt);
                        if (queryBar && queryBar !== node) dispatchFileEvents(queryBar, dt);
                        if (promptRoot && promptRoot !== queryBar && promptRoot !== node) dispatchFileEvents(promptRoot, dt);

                        return {
                            ok: populatedInputs > 0,
                            fileInputs: fileInputs.length,
                            populatedInputs,
                            selector,
                        };
                    }
                }
                return { ok: false, error: 'Prompt input not found for paste' };
            })()
        """

        upload_script = upload_script.replace("__FRAME_BASE64__", repr(frame_base64)).replace("__FRAME_NAME__", repr(frame_path.name))

        def after_focus(_result):
            if callable(on_uploaded):
                on_uploaded()

        self.browser.page().runJavaScript(upload_script, after_focus)

    def _wait_for_continue_upload_reload(self) -> None:
        self.continue_from_frame_waiting_for_reload = True
        self.continue_from_frame_reload_timeout_timer.start(10000)
        self._append_log(
            "Continue-from-last-frame: image pasted. Grok should auto-reload after upload; "
            "waiting for the new page before entering the continuation prompt..."
        )

    def _on_continue_reload_timeout(self) -> None:
        if not self.continue_from_frame_waiting_for_reload or not self.continue_from_frame_active:
            return
        self.continue_from_frame_waiting_for_reload = False
        self._append_log(
            "Timed out waiting for upload-triggered reload; continuing with prompt submission."
        )
        self._start_manual_browser_generation(self.continue_from_frame_prompt, 1)

    def continue_from_last_frame(self) -> None:
        source = self.prompt_source.currentData()
        if source != "manual":
            QMessageBox.warning(self, "Manual Mode Required", "Set Prompt Source to 'Manual prompt (no API)' for frame continuation.")
            return

        latest_video = self._resolve_latest_video_for_continuation()
        if not latest_video:
            QMessageBox.warning(self, "No Videos", "Generate or open a video first.")
            return

        manual_prompt = self.manual_prompt.toPlainText().strip()
        if not manual_prompt:
            QMessageBox.warning(self, "Missing Manual Prompt", "Enter a manual prompt for the continuation run.")
            return

        self.continue_from_frame_active = True
        self.continue_from_frame_waiting_for_reload = False
        self.continue_from_frame_reload_timeout_timer.stop()
        self.continue_from_frame_target_count = self.count.value()
        self.continue_from_frame_completed = 0
        self.continue_from_frame_prompt = manual_prompt
        self.continue_from_frame_current_source_video = ""
        self.continue_from_frame_seed_image_path = None
        self._append_log(
            f"Continue-from-last-frame started for {self.continue_from_frame_target_count} iteration(s)."
        )
        self._append_log(f"Continue-from-last-frame source video selected: {latest_video}")
        self._start_continue_iteration()

    def continue_from_local_image(self) -> None:
        source = self.prompt_source.currentData()
        if source != "manual":
            QMessageBox.warning(self, "Manual Mode Required", "Set Prompt Source to 'Manual prompt (no API)' for image continuation.")
            return

        manual_prompt = self.manual_prompt.toPlainText().strip()
        if not manual_prompt:
            QMessageBox.warning(self, "Missing Manual Prompt", "Enter a manual prompt for the continuation run.")
            return

        image_path, _ = QFileDialog.getOpenFileName(self, "Select image", str(self.download_dir), "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not image_path:
            return

        seed_image = Path(image_path)
        if not seed_image.exists():
            QMessageBox.warning(self, "Image Missing", "Selected image was not found on disk.")
            return

        self.continue_from_frame_active = True
        self.continue_from_frame_waiting_for_reload = False
        self.continue_from_frame_reload_timeout_timer.stop()
        self.continue_from_frame_target_count = self.count.value()
        self.continue_from_frame_completed = 0
        self.continue_from_frame_prompt = manual_prompt
        self.continue_from_frame_current_source_video = ""
        self.continue_from_frame_seed_image_path = seed_image

        self._append_log(
            f"Continue-from-image started for {self.continue_from_frame_target_count} iteration(s) using {seed_image}."
        )
        self._start_continue_iteration()

    def show_browser_page(self) -> None:
        self.browser.setUrl(QUrl("https://grok.com/imagine"))
        self._append_log("Navigated embedded browser to grok.com/imagine.")

    def stitch_all_videos(self) -> None:
        if self.stitch_worker and self.stitch_worker.isRunning():
            QMessageBox.information(self, "Stitch In Progress", "A stitch operation is already running.")
            return

        if len(self.videos) < 2:
            QMessageBox.warning(self, "Need More Videos", "At least two videos are required to stitch.")
            return

        timestamp = int(time.time() * 1000)
        output_file = self.download_dir / f"stitched_{timestamp}.mp4"
        stitched_base_file = self.download_dir / f"stitched_base_{timestamp}.mp4"
        video_paths = [Path(video["video_file_path"]) for video in self.videos]
        interpolate_enabled = self.stitch_interpolation_checkbox.isChecked()
        interpolation_fps = int(self.stitch_interpolation_fps.currentData())
        upscale_enabled = self.stitch_upscale_checkbox.isChecked()
        upscale_target = str(self.stitch_upscale_target.currentData())
        crossfade_enabled = self.stitch_crossfade_checkbox.isChecked()
        gpu_requested = self.stitch_gpu_checkbox.isChecked()
        gpu_enabled = gpu_requested and self._ffmpeg_supports_nvenc()
        custom_music_file = self.custom_music_file
        mute_original_audio = self.stitch_mute_original_checkbox.isChecked()
        original_audio_volume = float(self.stitch_original_audio_volume.value()) / 100.0
        music_volume = float(self.stitch_music_volume.value()) / 100.0
        audio_fade_duration = self.stitch_audio_fade_duration.value()
        if gpu_requested and not gpu_enabled:
            self._append_log("GPU encoding requested, but ffmpeg NVENC is unavailable. Falling back to CPU encoding.")

        settings_summary = (
            f"Crossfade: {'on' if crossfade_enabled else 'off'}"
            + (f" ({self.crossfade_duration.value():.1f}s)" if crossfade_enabled else "")
            + f" | Interpolation: {f'{interpolation_fps} fps' if interpolate_enabled else 'off'}"
            + f" | Upscaling: {upscale_target if upscale_enabled else 'off'}"
            + f" | Encode: {'GPU' if gpu_enabled else 'CPU'}"
            + (f" | Music: {custom_music_file.name}" if custom_music_file else " | Music: off")
        )

        started_at = time.time()
        self.stitch_progress_bar.setVisible(True)

        def update_progress(value: int, stage: str) -> None:
            bounded_value = max(0, min(100, int(value)))
            elapsed = time.time() - started_at
            eta_label = "calculating..."
            if bounded_value > 0:
                eta_seconds = max(0.0, (elapsed / bounded_value) * (100 - bounded_value))
                eta_label = f"~{eta_seconds:.0f}s"

            self.stitch_progress_bar.setValue(bounded_value)
            self.stitch_progress_label.setText(
                f"{stage} | {bounded_value}% | Elapsed: {elapsed:.1f}s | ETA: {eta_label} | {settings_summary}"
            )

        def on_stitch_failed(title: str, message: str) -> None:
            self.stitch_progress_label.setText(f"Stitch progress: failed ({message[:120]})")
            self.stitch_progress_bar.setVisible(False)
            QMessageBox.critical(self, title, message)

        def on_stitch_finished(stitched_video: dict) -> None:
            self._append_log(f"Stitched video created: {stitched_video['video_file_path']}")
            self.stitch_progress_label.setText("Stitch progress: complete")
            self.stitch_progress_bar.setValue(100)
            self.stitch_progress_bar.setVisible(False)
            self.on_video_finished(stitched_video)

        def on_stitch_complete() -> None:
            self.stitch_worker = None

        update_progress(1, "Preparing stitch pipeline...")

        self.stitch_worker = StitchWorker(
            window=self,
            video_paths=video_paths,
            output_file=output_file,
            stitched_base_file=stitched_base_file,
            crossfade_enabled=crossfade_enabled,
            crossfade_duration=self.crossfade_duration.value(),
            interpolate_enabled=interpolate_enabled,
            interpolation_fps=interpolation_fps,
            upscale_enabled=upscale_enabled,
            upscale_target=upscale_target,
            use_gpu_encoding=gpu_enabled,
            custom_music_file=custom_music_file,
            mute_original_audio=mute_original_audio,
            original_audio_volume=original_audio_volume,
            music_volume=music_volume,
            audio_fade_duration=audio_fade_duration,
        )
        self.stitch_worker.progress.connect(update_progress)
        self.stitch_worker.status.connect(self._append_log)
        self.stitch_worker.failed.connect(on_stitch_failed)
        self.stitch_worker.finished_stitch.connect(on_stitch_finished)
        self.stitch_worker.finished.connect(on_stitch_complete)
        self.stitch_worker.start()

    def _stitch_videos_concat(
        self,
        video_paths: list[Path],
        output_file: Path,
        progress_callback: Callable[[float], None] | None = None,
        use_gpu_encoding: bool = False,
    ) -> None:
        list_file = self.download_dir / f"stitch_list_{int(time.time() * 1000)}.txt"

        concat_lines = []
        for video_path in video_paths:
            quoted_path = video_path.as_posix().replace("'", "'\\''")
            concat_lines.append(f"file '{quoted_path}'")
        list_file.write_text("\n".join(concat_lines), encoding="utf-8")

        try:
            total_duration = sum(self._probe_video_duration(path) for path in video_paths)
            self._run_ffmpeg_with_progress(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_file),
                    *self._video_encoder_args(use_gpu_encoding),
                    "-c:a",
                    "aac",
                    str(output_file),
                ],
                total_duration=total_duration,
                progress_callback=progress_callback,
            )
        finally:
            if list_file.exists():
                list_file.unlink()

    def _probe_video_duration(self, video_path: Path) -> float:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        duration = float(result.stdout.strip())
        if duration <= 0:
            raise RuntimeError(f"Could not determine valid duration for {video_path.name}.")
        return duration

    def _probe_video_stream_info(self, video_path: Path) -> dict:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,duration",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            raise RuntimeError(f"No video stream found in {video_path.name}.")
        stream = streams[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        duration_raw = stream.get("duration")
        duration = float(duration_raw) if duration_raw not in (None, "N/A", "") else self._probe_video_duration(video_path)
        if width <= 0 or height <= 0 or duration <= 0:
            raise RuntimeError(f"Could not probe valid video stream info for {video_path.name}.")
        return {"width": width, "height": height, "duration": duration}

    def _video_has_audio_stream(self, video_path: Path) -> bool:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return bool(result.stdout.strip())

    def _ffmpeg_supports_nvenc(self) -> bool:
        if self._ffmpeg_nvenc_checked:
            return self._ffmpeg_nvenc_available

        self._ffmpeg_nvenc_checked = True
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            encoders_output = f"{result.stdout}\n{result.stderr}"
            self._ffmpeg_nvenc_available = "h264_nvenc" in encoders_output
        except Exception:
            self._ffmpeg_nvenc_available = False
        self.preview_fullscreen_overlay_btn: QPushButton | None = None

        return self._ffmpeg_nvenc_available

    def _video_encoder_args(self, use_gpu_encoding: bool, crf: int = 20) -> list[str]:
        if use_gpu_encoding and self._ffmpeg_supports_nvenc():
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(max(18, min(30, crf + 1))), "-b:v", "0"]
        return ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf)]

    def _run_ffmpeg_with_progress(
        self,
        ffmpeg_cmd: list[str],
        total_duration: float,
        progress_callback: Callable[[float], None] | None = None,
    ) -> None:
        command = ffmpeg_cmd[:-1] + ["-progress", "pipe:1", "-nostats", "-loglevel", "error", ffmpeg_cmd[-1]]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if progress_callback is not None:
            progress_callback(0.0)

        out_time_ms = 0
        output_lines: list[str] = []
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                output_lines.append(line)
                if len(output_lines) > 200:
                    output_lines = output_lines[-200:]
                if line.startswith("out_time_ms="):
                    try:
                        out_time_ms = int(line.split("=", 1)[1])
                    except ValueError:
                        continue
                    if total_duration > 0 and progress_callback is not None:
                        progress = (out_time_ms / 1_000_000.0) / total_duration
                        progress_callback(max(0.0, min(1.0, progress)))

        return_code = process.wait()
        if return_code != 0:
            stderr_text = "\n".join(output_lines[-80:])
            raise subprocess.CalledProcessError(return_code, command, stderr=stderr_text)

        if progress_callback is not None:
            progress_callback(1.0)

    def _stitch_videos_with_crossfade(
        self,
        video_paths: list[Path],
        output_file: Path,
        crossfade_duration: float,
        progress_callback: Callable[[float], None] | None = None,
        use_gpu_encoding: bool = False,
    ) -> None:
        stream_infos = [self._probe_video_stream_info(path) for path in video_paths]
        durations = [info["duration"] for info in stream_infos]
        has_audio = all(self._video_has_audio_stream(path) for path in video_paths)
        for path, duration in zip(video_paths, durations):
            if duration <= crossfade_duration + 0.05:
                raise RuntimeError(
                    f"Clip '{path.name}' is too short ({duration:.2f}s). Each clip must be longer than {crossfade_duration:.1f}s for crossfade stitching."
                )

        target_width = stream_infos[0]["width"]
        target_height = stream_infos[0]["height"]

        ffmpeg_cmd = ["ffmpeg", "-y"]
        for path in video_paths:
            ffmpeg_cmd.extend(["-i", str(path)])

        filter_parts: list[str] = []
        for idx in range(len(video_paths)):
            filter_parts.append(
                f"[{idx}:v]fps=24,scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,setsar=1,settb=AVTB,format=yuv420p[vsrc{idx}]"
            )
            if has_audio:
                filter_parts.append(
                    f"[{idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,aresample=async=1[asrc{idx}]"
                )

        cumulative_duration = durations[0]
        video_prev = "vsrc0"
        audio_prev = "asrc0"

        for idx in range(1, len(video_paths)):
            offset = cumulative_duration - crossfade_duration
            next_video = f"v{idx}"
            filter_parts.append(
                f"[{video_prev}][vsrc{idx}]xfade=transition=fade:duration={crossfade_duration:.3f}:offset={max(0.0, offset):.3f}[{next_video}]"
            )
            video_prev = next_video

            if has_audio:
                next_audio = f"a{idx}"
                filter_parts.append(
                    f"[{audio_prev}][asrc{idx}]acrossfade=d={crossfade_duration:.3f}:c1=tri:c2=tri[{next_audio}]"
                )
                audio_prev = next_audio

            cumulative_duration += durations[idx] - crossfade_duration

        filter_complex = ";".join(filter_parts)
        ffmpeg_cmd.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                f"[{video_prev}]",
                *self._video_encoder_args(use_gpu_encoding),
                str(output_file),
            ]
        )
        if has_audio:
            ffmpeg_cmd[-1:-1] = ["-map", f"[{audio_prev}]", "-c:a", "aac"]

        self._run_ffmpeg_with_progress(
            ffmpeg_cmd,
            total_duration=cumulative_duration,
            progress_callback=progress_callback,
        )

    def _enhance_stitched_video(
        self,
        input_file: Path,
        output_file: Path,
        interpolate: bool,
        interpolation_fps: int,
        upscale: bool,
        upscale_target: str = "2x",
        progress_callback: Callable[[float], None] | None = None,
        use_gpu_encoding: bool = False,
    ) -> None:
        if not interpolate and not upscale:
            return

        vf_filters: list[str] = []
        if interpolate:
            target_fps = 48 if interpolation_fps not in {48, 60} else interpolation_fps
            vf_filters.append(
                f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
            )
        if upscale:
            if upscale_target == "1080p":
                vf_filters.append(
                    "scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,"
                    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
                )
            elif upscale_target == "1440p":
                vf_filters.append(
                    "scale=2560:1440:force_original_aspect_ratio=decrease:flags=lanczos,"
                    "pad=2560:1440:(ow-iw)/2:(oh-ih)/2"
                )
            elif upscale_target == "4k":
                vf_filters.append(
                    "scale=3840:2160:force_original_aspect_ratio=decrease:flags=lanczos,"
                    "pad=3840:2160:(ow-iw)/2:(oh-ih)/2"
                )
            else:
                vf_filters.append("scale='min(iw*2,3840)':'min(ih*2,2160)':flags=lanczos")

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
            "-vf",
            ",".join(vf_filters),
            *self._video_encoder_args(use_gpu_encoding, crf=18),
            "-c:a",
            "copy",
            str(output_file),
        ]
        self._run_ffmpeg_with_progress(
            command,
            total_duration=self._probe_video_duration(input_file),
            progress_callback=progress_callback,
        )

    def _apply_custom_music_track(
        self,
        input_file: Path,
        output_file: Path,
        music_file: Path,
        mute_original_audio: bool,
        original_audio_volume: float,
        music_volume: float,
        fade_duration: float,
        progress_callback: Callable[[float], None] | None = None,
        use_gpu_encoding: bool = False,
    ) -> None:
        temp_output = self.download_dir / f"stitch_music_{int(time.time() * 1000)}.mp4"
        video_duration = self._probe_video_duration(input_file)
        music_duration = self._probe_video_duration(music_file)

        clamped_original_volume = max(0.0, min(2.0, original_audio_volume))
        clamped_music_volume = max(0.0, min(2.0, music_volume))
        clamped_fade = max(0.0, min(float(fade_duration), max(0.0, video_duration / 2.0)))

        trim_start = 0.0
        trim_duration = video_duration
        if music_duration > video_duration:
            extra = music_duration - video_duration
            trim_start = extra / 2.0
            trim_duration = video_duration

        has_original_audio = self._video_has_audio_stream(input_file)
        music_chain = (
            f"[1:a]atrim=start={trim_start:.6f}:duration={trim_duration:.6f},asetpts=PTS-STARTPTS,"
            f"aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            f"volume={clamped_music_volume:.3f}[music]"
        )

        filter_parts: list[str] = [music_chain]
        audio_output_label = "music"

        if has_original_audio and not mute_original_audio:
            filter_parts.append(
                f"[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                f"volume={clamped_original_volume:.3f}[orig]"
            )
            filter_parts.append("[orig][music]amix=inputs=2:duration=first:dropout_transition=0[audmix]")
            audio_output_label = "audmix"

        if clamped_fade > 0.0:
            fade_out_start = max(0.0, video_duration - clamped_fade)
            filter_parts.append(
                f"[{audio_output_label}]afade=t=in:st=0:d={clamped_fade:.3f},"
                f"afade=t=out:st={fade_out_start:.3f}:d={clamped_fade:.3f}[audfinal]"
            )
            audio_output_label = "audfinal"

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
            "-i",
            str(music_file),
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "0:v",
            "-map",
            f"[{audio_output_label}]",
            *self._video_encoder_args(use_gpu_encoding),
            "-c:a",
            "aac",
            "-shortest",
            str(temp_output),
        ]

        self._run_ffmpeg_with_progress(
            command,
            total_duration=video_duration,
            progress_callback=progress_callback,
        )

        temp_output.replace(output_file)

    def upload_selected_to_youtube(self) -> None:
        index = self.video_picker.currentIndex()
        if index < 0 or index >= len(self.videos):
            QMessageBox.warning(self, "No Video Selected", "Select a video to upload first.")
            return

        video_path = self.videos[index]["video_file_path"]
        title, description, hashtags, category_id, accepted = self._show_upload_dialog("YouTube")
        if not accepted:
            return

        client_secret_file = str(BASE_DIR / "client_secret.json")
        token_file = str(BASE_DIR / "youtube_token.json")
        if not Path(client_secret_file).exists():
            QMessageBox.critical(self, "Missing client_secret.json", f"Expected: {client_secret_file}")
            return

        self.upload_youtube_btn.setEnabled(False)
        self._append_log("Starting YouTube upload...")
        try:
            video_id = upload_video_to_youtube(
                client_secret_file=client_secret_file,
                token_file=token_file,
                video_path=video_path,
                title=title,
                description=description,
                tags=hashtags,
                category_id=category_id,
                youtube_api_key=self.youtube_api_key.text().strip(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "YouTube Upload Failed", str(exc))
            self._append_log(f"ERROR: YouTube upload failed: {exc}")
        else:
            self._append_log(f"YouTube upload complete. Video ID: {video_id}")
            QMessageBox.information(self, "YouTube Upload Complete", f"Video uploaded successfully. ID: {video_id}")
        finally:
            self.upload_youtube_btn.setEnabled(True)

    def upload_selected_to_facebook(self) -> None:
        index = self.video_picker.currentIndex()
        if index < 0 or index >= len(self.videos):
            QMessageBox.warning(self, "No Video Selected", "Select a video to upload first.")
            return

        video_path = self.videos[index]["video_file_path"]
        title, description, hashtags, _, accepted = self._show_upload_dialog("Facebook")
        if not accepted:
            return

        self.upload_facebook_btn.setEnabled(False)
        self._append_log("Starting Facebook upload...")
        try:
            video_id = upload_video_to_facebook_page(
                page_id=self.facebook_page_id.text().strip(),
                access_token=self.facebook_access_token.text().strip(),
                video_path=video_path,
                title=title,
                description=self._compose_social_text(description, hashtags),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Facebook Upload Failed", str(exc))
            self._append_log(f"ERROR: Facebook upload failed: {exc}")
        else:
            self._append_log(f"Facebook upload complete. Video ID: {video_id}")
            QMessageBox.information(self, "Facebook Upload Complete", f"Video uploaded successfully. ID: {video_id}")
        finally:
            self.upload_facebook_btn.setEnabled(True)

    def upload_selected_to_instagram(self) -> None:
        index = self.video_picker.currentIndex()
        if index < 0 or index >= len(self.videos):
            QMessageBox.warning(self, "No Video Selected", "Select a video to upload first.")
            return

        selected_video = self.videos[index]
        source_url = str(selected_video.get("source_url") or "").strip()
        if not source_url.startswith(("http://", "https://")):
            QMessageBox.warning(
                self,
                "Instagram Upload Requires URL",
                "Instagram Graph API video publishing requires a publicly reachable HTTP(S) URL. "
                "This selected video only exists locally.",
            )
            return

        _, caption, hashtags, _, accepted = self._show_upload_dialog("Instagram", title_enabled=False)
        if not accepted:
            return

        self.upload_instagram_btn.setEnabled(False)
        self._append_log("Starting Instagram upload...")
        try:
            media_id = upload_video_to_instagram_reels(
                ig_user_id=self.instagram_business_id.text().strip(),
                access_token=self.instagram_access_token.text().strip(),
                video_url=source_url,
                caption=self._compose_social_text(caption, hashtags),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Instagram Upload Failed", str(exc))
            self._append_log(f"ERROR: Instagram upload failed: {exc}")
        else:
            self._append_log(f"Instagram upload complete. Media ID: {media_id}")
            QMessageBox.information(self, "Instagram Upload Complete", f"Media published successfully. ID: {media_id}")
        finally:
            self.upload_instagram_btn.setEnabled(True)

    def _compose_social_text(self, base_text: str, hashtags: list[str]) -> str:
        tag_text = " ".join(f"#{tag.lstrip('#')}" for tag in hashtags if tag.strip())
        if not tag_text:
            return base_text.strip()
        combined = f"{base_text.strip()}\n\n{tag_text}" if base_text.strip() else tag_text
        return combined.strip()

    def _show_upload_dialog(self, platform_name: str, title_enabled: bool = True) -> tuple[str, str, list[str], str, bool]:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{platform_name} Upload Details")
        dialog_layout = QVBoxLayout(dialog)

        title_input = QLineEdit()
        if title_enabled:
            dialog_layout.addWidget(QLabel(f"{platform_name} Title"))
            title_input.setText(self.ai_social_metadata.title)
            dialog_layout.addWidget(title_input)

        dialog_layout.addWidget(QLabel(f"{platform_name} Description / Caption"))
        description_input = QPlainTextEdit()
        description_input.setPlaceholderText("Describe this upload...")
        description_input.setPlainText(self.ai_social_metadata.description)
        dialog_layout.addWidget(description_input)

        dialog_layout.addWidget(QLabel("Hashtags (comma separated, no # needed)"))
        hashtags_input = QLineEdit(", ".join(self.ai_social_metadata.hashtags))
        dialog_layout.addWidget(hashtags_input)

        category_input = QLineEdit(self.ai_social_metadata.category)
        if platform_name == "YouTube":
            dialog_layout.addWidget(QLabel("YouTube Category ID"))
            dialog_layout.addWidget(category_input)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        dialog_layout.addWidget(button_box)

        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        hashtags = [tag.strip().lstrip("#") for tag in hashtags_input.text().split(",") if tag.strip()]
        category_value = category_input.text().strip() if platform_name == "YouTube" else self.ai_social_metadata.category
        return title_input.text().strip(), description_input.toPlainText().strip(), hashtags, category_value, accepted


if __name__ == "__main__":
    _configure_qtwebengine_runtime()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
