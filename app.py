import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from PySide6.QtCore import QMimeData, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

from youtube_uploader import upload_video_to_youtube

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)
CACHE_DIR = BASE_DIR / ".qtwebengine"
MIN_VALID_VIDEO_BYTES = 1 * 1024 * 1024
API_BASE_URL = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _configure_qtwebengine_runtime() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    default_flags = [
        "--enable-gpu-rasterization",
        "--enable-zero-copy",
        "--ignore-gpu-blocklist",
        "--disable-renderer-backgrounding",
        "--autoplay-policy=no-user-gesture-required",
        f"--media-cache-size={_env_int('GROK_BROWSER_MEDIA_CACHE_BYTES', 268435456)}",
        f"--disk-cache-size={_env_int('GROK_BROWSER_DISK_CACHE_BYTES', 536870912)}",
    ]
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


class GenerateWorker(QThread):
    finished_video = Signal(dict)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, config: GrokConfig, prompt_config: PromptConfig, count: int):
        super().__init__()
        self.config = config
        self.prompt_config = prompt_config
        self.count = count

    def run(self) -> None:
        try:
            for idx in range(1, self.count + 1):
                self.status.emit(f"Generating variant {idx}/{self.count}...")
                video = self.generate_one_video(idx)
                self.finished_video.emit(video)
            self.status.emit("Generation complete.")
        except Exception as exc:
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
        source = self.prompt_config.source
        if source == "manual":
            return self.prompt_config.manual_prompt

        system = "You write highly visual prompts for short cinematic AI videos."
        user = (
            "Create one polished video prompt for a 10 second scene in 720p from this concept: "
            f"{self.prompt_config.concept}. This is variant #{variant}."
        )

        if source == "openai":
            return self.call_openai_chat(system, user)
        return self.call_grok_chat(system, user)

    def start_video_job(self, prompt: str, resolution: str) -> str:
        response = requests.post(
            f"{API_BASE_URL}/imagine/video/generations",
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.config.image_model,
                "prompt": prompt,
                "duration_seconds": 10,
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
        file_path = DOWNLOAD_DIR / f"video_{int(time.time() * 1000)}_{suffix}.mp4"
        with requests.get(video_url, stream=True, timeout=240) as response:
            response.raise_for_status()
            with open(file_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
        return file_path

    def generate_one_video(self, variant: int) -> dict:
        prompt = self.build_prompt(variant)

        video_job_id = None
        chosen_resolution = None
        for resolution in ["1280x720", "854x480"]:
            try:
                video_job_id = self.start_video_job(prompt, resolution)
                chosen_resolution = resolution
                break
            except requests.HTTPError:
                continue

        if not video_job_id:
            raise RuntimeError("Could not start a video generation job")

        result = self.poll_video_job(video_job_id)
        video_url = result.get("output", {}).get("video_url") or result.get("video_url")
        if not video_url:
            raise RuntimeError("No video URL returned")

        file_path = self.download_video(video_url, f"v{variant}")
        return {
            "title": f"Generated Video {variant}",
            "prompt": prompt,
            "resolution": chosen_resolution,
            "video_file_path": str(file_path),
            "source_url": video_url,
        }


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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grok Video Desktop Studio")
        self.resize(1500, 900)
        self.videos: list[dict] = []
        self.worker: GenerateWorker | None = None
        self.manual_generation_queue: list[dict] = []
        self.pending_manual_variant_for_download: int | None = None
        self.manual_download_deadline: float | None = None
        self.manual_download_click_sent = False
        self.manual_download_started_at: float | None = None
        self.manual_requires_image_pick = True
        self.manual_image_make_video_clicked = False
        self.pending_manual_variant_prompt: str = ""
        self.manual_video_prompt_submitted = False
        self.manual_download_poll_timer = QTimer(self)
        self.manual_download_poll_timer.setSingleShot(True)
        self.manual_download_poll_timer.timeout.connect(self._poll_for_manual_video)
        self.continue_from_frame_active = False
        self.continue_from_frame_target_count = 0
        self.continue_from_frame_completed = 0
        self.continue_from_frame_prompt = ""
        self.continue_from_frame_current_source_video = ""
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
        self._build_ui()

    def _build_ui(self) -> None:
        splitter = QSplitter()

        left = QWidget()
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel("Grok API Key (required for video generation)"))
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setText(os.getenv("GROK_API_KEY", ""))
        left_layout.addWidget(self.api_key)

        left_layout.addWidget(QLabel("Chat Model"))
        self.chat_model = QLineEdit(os.getenv("GROK_CHAT_MODEL", "grok-3-mini"))
        left_layout.addWidget(self.chat_model)

        left_layout.addWidget(QLabel("Video Model"))
        self.image_model = QLineEdit(os.getenv("GROK_VIDEO_MODEL", "grok-video-latest"))
        left_layout.addWidget(self.image_model)

        left_layout.addWidget(QLabel("Prompt Source"))
        self.prompt_source = QComboBox()
        self.prompt_source.addItem("Manual prompt (no API)", "manual")
        self.prompt_source.addItem("Grok API", "grok")
        self.prompt_source.addItem("OpenAI API", "openai")
        self.prompt_source.currentIndexChanged.connect(self._toggle_prompt_source_fields)
        left_layout.addWidget(self.prompt_source)

        left_layout.addWidget(QLabel("OpenAI API Key (for OpenAI prompt generation)"))
        self.openai_api_key = QLineEdit()
        self.openai_api_key.setEchoMode(QLineEdit.Password)
        self.openai_api_key.setText(os.getenv("OPENAI_API_KEY", ""))
        left_layout.addWidget(self.openai_api_key)

        left_layout.addWidget(QLabel("OpenAI Chat Model"))
        self.openai_chat_model = QLineEdit(os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
        left_layout.addWidget(self.openai_chat_model)

        left_layout.addWidget(QLabel("Concept"))
        self.concept = QPlainTextEdit()
        self.concept.setPlaceholderText("Describe the video idea...")
        left_layout.addWidget(self.concept)

        left_layout.addWidget(QLabel("Manual Prompt (used only when source is Manual)"))
        self.manual_prompt = QPlainTextEdit()
        self.manual_prompt.setPlaceholderText("Paste or write an exact prompt to skip prompt APIs...")
        self.manual_prompt.setPlainText(
            "abstract surreal artistic photorealistic strange random dream like scifi slow moving camera, "
            "slow moving morphing people, furniture, scenary, creatures,  50s style color tv 70mm film"
        )
        left_layout.addWidget(self.manual_prompt)

        row = QHBoxLayout()
        row.addWidget(QLabel("Count"))
        self.count = QSpinBox()
        self.count.setRange(1, 10)
        self.count.setValue(1)
        row.addWidget(self.count)
        left_layout.addLayout(row)

        self.generate_btn = QPushButton("Generate Video")
        self.generate_btn.clicked.connect(self.start_generation)
        left_layout.addWidget(self.generate_btn)

        self.open_btn = QPushButton("Open Local Video...")
        self.open_btn.clicked.connect(self.open_local_video)
        left_layout.addWidget(self.open_btn)

        self.extract_frame_btn = QPushButton("Extract Last Frame + Copy Image")
        self.extract_frame_btn.clicked.connect(self.extract_last_frame_from_selected)
        left_layout.addWidget(self.extract_frame_btn)

        self.continue_frame_btn = QPushButton("Continue from Last Frame (paste + generate)")
        self.continue_frame_btn.clicked.connect(self.continue_from_last_frame)
        left_layout.addWidget(self.continue_frame_btn)

        self.show_browser_btn = QPushButton("Show Browser (grok.com/imagine)")
        self.show_browser_btn.clicked.connect(self.show_browser_page)
        left_layout.addWidget(self.show_browser_btn)

        self.stitch_btn = QPushButton("Stitch All Videos")
        self.stitch_btn.clicked.connect(self.stitch_all_videos)
        left_layout.addWidget(self.stitch_btn)

        self.upload_youtube_btn = QPushButton("Upload Selected to YouTube")
        self.upload_youtube_btn.clicked.connect(self.upload_selected_to_youtube)
        left_layout.addWidget(self.upload_youtube_btn)

        left_layout.addWidget(QLabel("Generated Videos"))
        self.video_picker = QComboBox()
        self.video_picker.currentIndexChanged.connect(self.show_selected_video)
        left_layout.addWidget(self.video_picker)

        left_layout.addWidget(QLabel("Activity Log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        left_layout.addWidget(self.log)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)

        self.browser = QWebEngineView()
        self.browser_profile = QWebEngineProfile("grok-video-desktop-profile", self)
        self.browser.setPage(FilteredWebEnginePage(self._append_log, self.browser_profile, self.browser))
        browser_profile = self.browser_profile
        browser_profile.setPersistentStoragePath(str(CACHE_DIR / "profile"))
        browser_profile.setCachePath(str(CACHE_DIR / "cache"))
        browser_profile.setPersistentCookiesPolicy(QWebEngineProfile.ForcePersistentCookies)
        browser_profile.setHttpCacheType(QWebEngineProfile.DiskHttpCache)
        browser_profile.setHttpCacheMaximumSize(_env_int("GROK_BROWSER_DISK_CACHE_BYTES", 536870912))

        browser_settings = self.browser.settings()
        browser_settings.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
        browser_settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        browser_settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        browser_settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)

        self.browser.setUrl(QUrl("https://grok.com/imagine"))
        self.browser.loadFinished.connect(self._on_browser_load_finished)
        self.browser_profile.downloadRequested.connect(self._on_browser_download_requested)

        self._toggle_prompt_source_fields()

        splitter.addWidget(left)
        splitter.addWidget(self.browser)
        splitter.setSizes([500, 1000])

        # Keep browser visible as a fixed right-hand pane
        splitter.setChildrenCollapsible(False)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

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

        prompt_config = PromptConfig(
            source=source,
            concept=concept,
            manual_prompt=manual_prompt,
            openai_api_key=self.openai_api_key.text().strip(),
            openai_chat_model=self.openai_chat_model.text().strip() or "gpt-4o-mini",
        )

        self.worker = GenerateWorker(config, prompt_config, self.count.value())
        self.worker.status.connect(self._append_log)
        self.worker.finished_video.connect(self.on_video_finished)
        self.worker.failed.connect(self.on_generation_error)
        self.worker.start()

    def _start_manual_browser_generation(self, prompt: str, count: int) -> None:
        self.manual_generation_queue = [{"prompt": prompt, "variant": idx} for idx in range(1, count + 1)]
        self._append_log(
            "Manual mode now reuses the current browser page exactly as-is. "
            "No navigation or reload will happen."
        )
        self._append_log(f"Manual mode queued with repeat count={count}.")
        self._append_log("Attempting to populate the visible Grok prompt box on the current page...")
        self._submit_next_manual_variant()

    def _start_continue_iteration(self) -> None:
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
            f"Continue iteration {iteration}/{self.continue_from_frame_target_count}: copied last frame {frame_path}"
        )
        browser_page_pause_ms = 200
        self._append_log(
            "Continue-from-last-frame: starting image paste into the current Grok prompt area "
            "without forcing page navigation..."
        )
        QTimer.singleShot(
            900 + browser_page_pause_ms,
            lambda: self._upload_frame_into_grok(frame_path, on_uploaded=self._wait_for_continue_upload_reload),
        )
        self._append_log(
            "Continue-from-last-frame: image paste scheduled; waiting for upload/reload before prompt submission."
        )

    def _resolve_latest_video_for_continuation(self) -> str | None:
        if self.videos:
            return self.videos[-1]["video_file_path"]

        candidates: list[Path] = []
        for pattern in ("*.mp4", "*.mov", "*.webm"):
            candidates.extend(DOWNLOAD_DIR.glob(pattern))

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
        if not self.manual_generation_queue:
            self._append_log("Manual browser generation complete.")
            return

        item = self.manual_generation_queue.pop(0)
        remaining_count = len(self.manual_generation_queue)
        prompt = item["prompt"]
        variant = item["variant"]
        self.pending_manual_variant_prompt = prompt
        self.pending_manual_variant_for_download = variant
        self.manual_download_click_sent = False
        action_delay_ms = 2000
        self._append_log(
            f"Populating prompt for manual variant {variant} in browser, setting image options, "
            f"then force submitting with {action_delay_ms}ms delays between each action. Remaining repeats after this: {remaining_count}."
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
                    const clickVisibleButtonByAriaLabel = (ariaLabel) => {
                        const button = [...document.querySelectorAll(`button[aria-label='${ariaLabel}']`)]
                            .find((el) => isVisible(el) && !el.disabled);
                        if (!button) return false;
                        return emulateClick(button);
                    };

                    const requiredOptions = ["image"];
                    const optionsRequested = [];
                    const optionsApplied = [];

                    const applyOption = (name, patterns, ariaLabel) => {
                        const isAlreadySelected = hasSelectedByText(patterns, composer) || hasSelectedByText(patterns);
                        if (isAlreadySelected) {
                            optionsApplied.push(`${name}(already-selected)`);
                            return;
                        }
                        const clicked = (ariaLabel && clickVisibleButtonByAriaLabel(ariaLabel))
                            || clickByText(patterns, composer)
                            || clickByText(patterns);
                        if (clicked) {
                            optionsRequested.push(name);
                        }
                        const isNowSelected = hasSelectedByText(patterns, composer) || hasSelectedByText(patterns);
                        if (clicked && !isNowSelected) {
                            clickByText(patterns, composer) || clickByText(patterns);
                        }
                        const selected = hasSelectedByText(patterns, composer) || hasSelectedByText(patterns);
                        if (selected) optionsApplied.push(name);
                    };

                    applyOption("image", [/^image$/i], null);

                    const missingOptions = requiredOptions.filter((option) => {
                        const patterns = [/^image$/i];
                        return !(hasSelectedByText(patterns, composer) || hasSelectedByText(patterns));
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
                        f"{variant} after prompt/options staged delays; "
                        "waiting for image results and then triggering video generation from the first image."
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
                            try { el.dispatchEvent(new MouseEvent("mouseover", common)); } catch (_) {}
                            try { el.dispatchEvent(new MouseEvent("mousemove", common)); } catch (_) {}
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
                    self._append_log(
                        f"WARNING: Continue-mode submit for variant {variant} reported an issue: {error_detail!r}."
                    )
                else:
                    button_label = submit_result.get("buttonAriaLabel") or submit_result.get("buttonText") or "Make video"
                    self._append_log(f"Continue-mode submit clicked '{button_label}' for variant {variant}.")
                self._trigger_browser_video_download(variant, require_image_pick=False)

            QTimer.singleShot(action_delay_ms, lambda: self.browser.page().runJavaScript(continue_submit_script, _after_continue_submit))

        def after_submit(result):
            fill_ok = isinstance(result, dict) and bool(result.get("ok"))
            if not fill_ok:
                error_detail = result.get("error") if isinstance(result, dict) else result
                self._append_log(
                    f"WARNING: Manual prompt fill reported an error for variant {variant}: {error_detail!r}. "
                    "Verifying current field content before continuing."
                )

            def _after_verify_prompt(verify_result):
                verify_ok = isinstance(verify_result, dict) and bool(verify_result.get("ok"))
                if not (fill_ok or verify_ok):
                    verify_error = verify_result.get("error") if isinstance(verify_result, dict) else verify_result
                    self._append_log(
                        f"WARNING: Prompt fill verification did not confirm content for variant {variant}: {verify_error!r}. "
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

    def _trigger_browser_video_download(self, variant: int, require_image_pick: bool = True) -> None:
        self.manual_download_deadline = time.time() + 420
        self.manual_download_click_sent = False
        self.manual_download_started_at = time.time()
        self.manual_requires_image_pick = require_image_pick
        self.manual_image_make_video_clicked = not require_image_pick
        self.manual_video_prompt_submitted = not require_image_pick
        self.manual_download_poll_timer.start(0)

    def _poll_for_manual_video(self) -> None:
        variant = self.pending_manual_variant_for_download
        if variant is None:
            return

        deadline = self.manual_download_deadline or 0
        if time.time() > deadline:
            self.pending_manual_variant_for_download = None
            self.manual_download_click_sent = False
            self.manual_download_started_at = None
            self.manual_download_deadline = None
            self.manual_requires_image_pick = True
            self.manual_image_make_video_clicked = False
            self.manual_video_prompt_submitted = False
            self._append_log(f"ERROR: Variant {variant} did not produce a downloadable video in time.")
            if self.continue_from_frame_active:
                self._append_log("Continue-from-last-frame stopped because download polling timed out.")
                self.continue_from_frame_active = False
                self.continue_from_frame_target_count = 0
                self.continue_from_frame_completed = 0
                self.continue_from_frame_prompt = ""
            return

        if self.manual_requires_image_pick and not self.manual_image_make_video_clicked:
            pick_image_script = r"""
                (() => {
                    try {
                        const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));

                        const videoPromptSelectors = [
                            "textarea[placeholder*='Type to customize this video' i]",
                            "input[placeholder*='Type to customize this video' i]",
                            "textarea[placeholder*='Type to customize video' i]",
                            "input[placeholder*='Type to customize video' i]",
                            "textarea[placeholder*='Customize video' i]",
                            "input[placeholder*='Customize video' i]",
                            "[contenteditable='true'][aria-label*='Type to customize this video' i]",
                            "[contenteditable='true'][data-placeholder*='Type to customize this video' i]",
                            "[contenteditable='true'][aria-label*='Type to customize video' i]",
                            "[contenteditable='true'][data-placeholder*='Type to customize video' i]",
                            "[contenteditable='true'][aria-label*='Make a video' i]"
                        ];

                        const alreadyOnVideoComposer = videoPromptSelectors.some((selector) =>
                            [...document.querySelectorAll(selector)].some((node) => isVisible(node))
                        ) || [...document.querySelectorAll("button, [role='button']")].some((node) => {
                            if (!isVisible(node)) return false;
                            const label = `${(node.getAttribute?.("aria-label") || "")} ${(node.textContent || "")}`.toLowerCase();
                            return /\bmake\s+video\b/.test(label);
                        });
                        if (alreadyOnVideoComposer) {
                            return { ok: true, clicked: true, count: 0, skippedImagePick: true };
                        }

                        const common = { bubbles: true, cancelable: true, composed: true };
                        const emulateClick = (el) => {
                            if (!el || !isVisible(el) || el.disabled) return false;
                            try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                            try { el.dispatchEvent(new MouseEvent("mouseover", common)); } catch (_) {}
                            try { el.dispatchEvent(new MouseEvent("mousemove", common)); } catch (_) {}
                            el.dispatchEvent(new MouseEvent("mousedown", common));
                            try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                            el.dispatchEvent(new MouseEvent("mouseup", common));
                            el.dispatchEvent(new MouseEvent("click", common));
                            return true;
                        };

                        const getClickableImageTarget = (listItem) => {
                            if (!listItem) return null;

                            const imageNodes = [
                                ...listItem.querySelectorAll("img"),
                                ...listItem.querySelectorAll("picture img"),
                                ...listItem.querySelectorAll("canvas"),
                            ].filter((node) => isVisible(node));

                            const nonVideoImages = imageNodes.filter((node) => {
                                const alt = (node.getAttribute?.("alt") || "").toLowerCase();
                                const ariaLabel = (node.getAttribute?.("aria-label") || "").toLowerCase();
                                return !/play|video/.test(`${alt} ${ariaLabel}`);
                            });

                            const targetImage = nonVideoImages[0] || imageNodes[0] || null;
                            if (!targetImage) return null;

                            const clickableAncestor = targetImage.closest("a, button, [role='button'], [tabindex], .relative.group, [class~='relative'][class~='group'], media-post-masonry-card, [data-testid*='media-post' i]");
                            if (clickableAncestor && isVisible(clickableAncestor) && clickableAncestor !== listItem) {
                                const label = `${(clickableAncestor.getAttribute("aria-label") || "")} ${(clickableAncestor.textContent || "")}`.toLowerCase();
                                if (!/play/.test(label)) return clickableAncestor;
                            }

                            return targetImage;
                        };

                        const listItems = [...document.querySelectorAll("div[role='listitem']")]
                            .filter((item) => isVisible(item));
                        if (!listItems.length) return { ok: true, clicked: false, count: 0 };

                        const mediaListItems = listItems.filter((item) => {
                            const hasMedia = !!item.querySelector("picture img, img, canvas, video");
                            const inHistoryOrNav = !!item.closest("nav, aside, [aria-label*='history' i], [data-testid*='history' i]");
                            return hasMedia && !inHistoryOrNav;
                        });

                        const firstItem = mediaListItems[0] || listItems[0];
                        const findBestCardTarget = (listItem) => {
                            if (!listItem) return null;
                            const candidates = [
                                ...listItem.querySelectorAll("[class*='cursor-pointer']"),
                                ...listItem.querySelectorAll("[class*='pointer-events-auto']"),
                                ...listItem.querySelectorAll("[class*='media-post-masonry-card']"),
                                ...listItem.querySelectorAll("[class*='group/media-post-masonry-card']"),
                                ...listItem.querySelectorAll("[data-testid*='media-post' i]"),
                                ...listItem.querySelectorAll("a, button, [role='button']"),
                            ].filter((node) => isVisible(node));

                            const nonControl = candidates.find((node) => {
                                if (!node) return false;
                                const label = `${(node.getAttribute?.("aria-label") || "")} ${(node.textContent || "")}`.toLowerCase();
                                return !/unsave|save|download|share|compose|post|redo|menu|more/.test(label);
                            });

                            return nonControl || candidates[0] || null;
                        };

                        const cardTarget =
                            findBestCardTarget(firstItem) ||
                            getClickableImageTarget(firstItem) ||
                            firstItem.querySelector(".relative.group") ||
                            firstItem.querySelector("[class~='relative'][class~='group']") ||
                            firstItem.querySelector("media-post-masonry-card") ||
                            firstItem.querySelector("[data-testid*='media-post' i]") ||
                            firstItem.querySelector("a") ||
                            firstItem;

                        const clicked = emulateClick(cardTarget);
                        if (clicked) {
                            firstItem.setAttribute("data-manual-image-open-clicked", "1");
                        }
                        return { ok: true, clicked, count: listItems.length };
                    } catch (err) {
                        return { ok: false, error: String(err && err.stack ? err.stack : err) };
                    }
                })()
            """

            def after_pick_image(result):
                ok = isinstance(result, dict) and bool(result.get("ok", True))
                clicked = isinstance(result, dict) and bool(result.get("clicked"))
                skipped_image_pick = isinstance(result, dict) and bool(result.get("skippedImagePick"))
                if not ok:
                    error_detail = result.get("error") if isinstance(result, dict) else result
                    self._append_log(
                        f"WARNING: Could not pick first generated image for variant {variant}: {error_detail!r}. Retrying."
                    )
                elif clicked:
                    self.manual_image_make_video_clicked = True
                    if skipped_image_pick:
                        self._append_log(
                            f"Variant {variant}: video generation prompt is already visible; skipping image-pick step."
                        )
                    else:
                        self._append_log(
                            f"Variant {variant}: opened the first generated image/list item to load the video generation page."
                        )

                self.manual_download_poll_timer.start(3000)

            self.browser.page().runJavaScript(pick_image_script, after_pick_image)
            return

        if self.manual_requires_image_pick and self.manual_image_make_video_clicked and not self.manual_video_prompt_submitted:
            escaped_prompt = repr(self.pending_manual_variant_prompt)
            submit_video_prompt_script = rf"""
                (() => {{
                    try {{
                        const prompt = {escaped_prompt};
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

                        const promptSelectors = [
                            "textarea[placeholder*='Type to customize this video' i]",
                            "input[placeholder*='Type to customize this video' i]",
                            "textarea[placeholder*='Type to customize video' i]",
                            "input[placeholder*='Type to customize video' i]",
                            "textarea[placeholder*='Customize video' i]",
                            "input[placeholder*='Customize video' i]",
                            "[contenteditable='true'][aria-label*='Type to customize this video' i]",
                            "[contenteditable='true'][data-placeholder*='Type to customize this video' i]",
                            "[contenteditable='true'][aria-label*='Type to customize video' i]",
                            "[contenteditable='true'][data-placeholder*='Type to customize video' i]",
                            "[contenteditable='true'][aria-label*='Make a video' i]"
                        ];

                        const inputCandidates = [];
                        promptSelectors.forEach((selector) => {{
                            const matches = document.querySelectorAll(selector);
                            for (let i = 0; i < matches.length; i += 1) inputCandidates.push(matches[i]);
                        }});
                        const input = inputCandidates.find((el) => isVisible(el));
                        if (!input) return {{ ok: true, submitted: false, reason: "video-prompt-input-not-found" }};

                        input.focus();
                        if (input.isContentEditable) {{
                            const paragraph = document.createElement("p");
                            paragraph.textContent = prompt;
                            input.replaceChildren(paragraph);
                        }} else {{
                            const proto = Object.getPrototypeOf(input);
                            const descriptor = proto ? Object.getOwnPropertyDescriptor(proto, "value") : null;
                            const valueSetter = descriptor && descriptor.set;
                            if (valueSetter) valueSetter.call(input, prompt);
                            else input.value = prompt;
                        }}
                        input.dispatchEvent(new Event("input", {{ bubbles: true }}));
                        input.dispatchEvent(new Event("change", {{ bubbles: true }}));

                        const buttons = [...document.querySelectorAll("button, [role='button']")].filter((el) => isVisible(el) && !el.disabled);
                        const generateButton = buttons.find((btn) => /generate/i.test((btn.textContent || "").trim()) || /make\s*video/i.test((btn.textContent || "").trim()));
                        if (!generateButton) return {{ ok: true, submitted: false, reason: "video-generate-button-not-found" }};

                        return {{ ok: true, submitted: emulateClick(generateButton) }};
                    }} catch (err) {{
                        return {{ ok: false, error: String(err && err.stack ? err.stack : err) }};
                    }}
                }})()
            """

            def after_submit_video_prompt(result):
                ok = isinstance(result, dict) and bool(result.get("ok", True))
                submitted = isinstance(result, dict) and bool(result.get("submitted"))
                if not ok:
                    error_detail = result.get("error") if isinstance(result, dict) else result
                    self._append_log(
                        f"WARNING: Could not submit video prompt for variant {variant}: {error_detail!r}. Retrying."
                    )
                elif submitted:
                    self.manual_video_prompt_submitted = True
                    self._append_log(
                        f"Variant {variant}: entered the video prompt on the generation page and clicked Generate Video."
                    )

                self.manual_download_poll_timer.start(3000)

            self.browser.page().runJavaScript(submit_video_prompt_script, after_submit_video_prompt)
            return

        script = """
            (() => {
                const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                const common = { bubbles: true, cancelable: true, composed: true };
                const emulateClick = (el) => {
                    if (!el || !isVisible(el) || el.disabled) return false;
                    try {
                        el.dispatchEvent(new PointerEvent("pointerdown", common));
                        el.dispatchEvent(new MouseEvent("mouseover", common));
                        el.dispatchEvent(new MouseEvent("mousemove", common));
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        el.dispatchEvent(new PointerEvent("pointerup", common));
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    } catch (_) {
                        try {
                            el.click();
                            return true;
                        } catch (__){
                            return false;
                        }
                    }
                };
                const percentNode = [...document.querySelectorAll("div .tabular-nums, div.tabular-nums")]
                    .find((el) => isVisible(el) && /^\\d{1,3}%$/.test((el.textContent || "").trim()));
                if (percentNode) {
                    return { status: "progress", progressText: (percentNode.textContent || "").trim() };
                }

                const redoButton = [...document.querySelectorAll("button")]
                    .find((btn) => isVisible(btn) && !btn.disabled && /redo/i.test((btn.textContent || "").trim()));

                const downloadButton = [...document.querySelectorAll("button[aria-label='Download'], button")]
                    .find((btn) => isVisible(btn) && !btn.disabled && /download/i.test((btn.getAttribute("aria-label") || btn.textContent || "").trim()));

                const mediaListItem = [...document.querySelectorAll("div[role='listitem']")]
                    .find((item) => isVisible(item) && (item.querySelector("video") || item.querySelector("img[alt*='Generated image' i]")));

                const resolveMediaOpenTarget = (listItem) => {
                    if (!listItem) return null;
                    const clickableChildren = [
                        ...listItem.querySelectorAll("[class*='cursor-pointer']"),
                        ...listItem.querySelectorAll("[class*='pointer-events-auto']"),
                        ...listItem.querySelectorAll("[class*='media-post-masonry-card']"),
                        ...listItem.querySelectorAll("[class*='group/media-post-masonry-card']"),
                        ...listItem.querySelectorAll("a, button, [role='button']"),
                    ].filter((el) => isVisible(el));

                    const preferred = clickableChildren.find((el) => {
                        const label = `${(el.getAttribute("aria-label") || "")} ${(el.textContent || "")}`.toLowerCase();
                        return !/unsave|save|download|share|compose|post|redo|menu|more/.test(label);
                    });

                    return preferred || clickableChildren[0] || listItem;
                };

                if (!redoButton) {
                    return { status: "waiting-for-redo" };
                }

                if (downloadButton) {
                    return {
                        status: emulateClick(downloadButton) ? "download-clicked" : "download-visible",
                    };
                }

                if (mediaListItem) {
                    const openTarget = resolveMediaOpenTarget(mediaListItem);
                    const clickedBefore = mediaListItem.getAttribute("data-manual-video-open-clicked") === "1";
                    const clicked = clickedBefore ? false : emulateClick(openTarget);
                    if (clicked) {
                        mediaListItem.setAttribute("data-manual-video-open-clicked", "1");
                    }
                    return {
                        status: clicked ? "video-page-open-clicked" : (clickedBefore ? "video-page-open-already-attempted" : "video-page-listitem-visible"),
                    };
                }

                const video = document.querySelector("video");
                const source = document.querySelector("video source");
                const src = (video && (video.currentSrc || video.src)) || (source && source.src) || "";
                const enoughData = !!(video && video.readyState >= 3 && Number(video.duration || 0) > 0);
                return {
                    status: src ? (enoughData ? "video-src-ready" : "video-buffering") : "waiting",
                    src,
                    readyState: video ? video.readyState : 0,
                    duration: video ? Number(video.duration || 0) : 0,
                };
            })()
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
                if progress_text:
                    self._append_log(f"Variant {current_variant} still rendering: {progress_text}")
                self.manual_download_poll_timer.start(3000)
                return

            if status == "waiting-for-redo":
                self.manual_download_poll_timer.start(3000)
                return

            if status == "download-clicked":
                if not self.manual_download_click_sent:
                    self._append_log(f"Variant {current_variant} appears ready; clicked in-page Download button.")
                    self.manual_download_click_sent = True
                self.manual_download_poll_timer.start(3000)
                return

            if status in {"video-page-open-clicked", "video-page-open-already-attempted", "video-page-listitem-visible"}:
                if status == "video-page-open-clicked":
                    self._append_log(
                        f"Variant {current_variant}: opened generated media list item to load the video page and Download button."
                    )
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
            self.browser.page().runJavaScript(trigger_download_script)
            if not self.manual_download_click_sent:
                self._append_log(f"Variant {current_variant} video detected; browser download requested from video source.")
                self.manual_download_click_sent = True

            self.manual_download_poll_timer.start(3000)

        self.browser.page().runJavaScript(script, after_poll)

    def _on_browser_download_requested(self, download) -> None:
        variant = self.pending_manual_variant_for_download
        if variant is None:
            return

        filename = f"video_{int(time.time() * 1000)}_manual_v{variant}.mp4"
        download.setDownloadDirectory(str(DOWNLOAD_DIR))
        download.setDownloadFileName(filename)
        download.accept()
        self._append_log(f"Downloading manual variant {variant} to {DOWNLOAD_DIR / filename}")

        def on_state_changed(state):
            if state == download.DownloadState.DownloadCompleted:
                video_path = DOWNLOAD_DIR / filename
                video_size = video_path.stat().st_size if video_path.exists() else 0
                if video_size < MIN_VALID_VIDEO_BYTES:
                    self._append_log(
                        f"WARNING: Downloaded manual variant {variant} is only {video_size} bytes (< 1MB)."
                    )
                    self.pending_manual_variant_for_download = None
                    self.manual_download_click_sent = False
                    self.manual_download_started_at = None
                    self.manual_download_deadline = None
                    self.manual_requires_image_pick = True
                    self.manual_image_make_video_clicked = False
                    self.manual_video_prompt_submitted = False
                    self.pending_manual_variant_prompt = ""

                    if self.continue_from_frame_active:
                        self._retry_continue_after_small_download(variant)
                    else:
                        self._append_log(
                            "WARNING: Undersized manual download detected outside continue-from-last-frame mode; "
                            "please use 'Continue from Last Frame' to regenerate from the extracted frame."
                        )
                    return

                self.videos.append(
                    {
                        "title": f"Manual Browser Video {variant}",
                        "prompt": self.manual_prompt.toPlainText().strip(),
                        "resolution": "web",
                        "video_file_path": str(video_path),
                        "source_url": "browser-session",
                    }
                )
                self.video_picker.addItem(f"Manual Browser Video {variant} (web)")
                self.video_picker.setCurrentIndex(self.video_picker.count() - 1)
                self._append_log(f"Saved: {video_path}")
                self._append_log("Download complete; returning embedded browser to grok.com/imagine.")
                QTimer.singleShot(0, self.show_browser_page)
                self.pending_manual_variant_for_download = None
                self.manual_download_click_sent = False
                self.manual_download_started_at = None
                self.manual_download_deadline = None
                self.manual_requires_image_pick = True
                self.manual_image_make_video_clicked = False
                self.manual_video_prompt_submitted = False
                self.pending_manual_variant_prompt = ""
                if self.continue_from_frame_active:
                    self.continue_from_frame_completed += 1
                    if self.continue_from_frame_completed < self.continue_from_frame_target_count:
                        QTimer.singleShot(800, self._start_continue_iteration)
                    else:
                        self._append_log("Continue-from-last-frame workflow complete.")
                        self.continue_from_frame_active = False
                        self.continue_from_frame_target_count = 0
                        self.continue_from_frame_completed = 0
                        self.continue_from_frame_prompt = ""
                        self.continue_from_frame_current_source_video = ""
                else:
                    self._submit_next_manual_variant()
            elif state == download.DownloadState.DownloadInterrupted:
                self._append_log(f"ERROR: Download interrupted for manual variant {variant}.")
                self.pending_manual_variant_for_download = None
                self.manual_download_click_sent = False
                self.manual_download_started_at = None
                self.manual_download_deadline = None
                self.manual_requires_image_pick = True
                self.manual_image_make_video_clicked = False
                self.manual_video_prompt_submitted = False
                self.pending_manual_variant_prompt = ""
                self.continue_from_frame_active = False
                self.continue_from_frame_target_count = 0
                self.continue_from_frame_completed = 0
                self.continue_from_frame_prompt = ""
                self.continue_from_frame_current_source_video = ""

        download.stateChanged.connect(on_state_changed)

    def on_video_finished(self, video: dict) -> None:
        self.videos.append(video)
        label = f"{video['title']} ({video['resolution']})"
        self.video_picker.addItem(label)
        self.video_picker.setCurrentIndex(self.video_picker.count() - 1)
        self._append_log(f"Saved: {video['video_file_path']}")

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
        self.player.stop()
        self._append_log(f"Selected video (preview disabled): {file_path}")

    def open_local_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select video", str(DOWNLOAD_DIR), "Videos (*.mp4 *.mov *.webm)")
        if not file_path:
            return
        self._preview_video(file_path)
        self._append_log(f"Opened local file (preview disabled): {file_path}")

    def _toggle_prompt_source_fields(self) -> None:
        source = self.prompt_source.currentData() if hasattr(self, "prompt_source") else "manual"
        is_manual = source == "manual"
        is_openai = source == "openai"
        self.manual_prompt.setEnabled(is_manual)
        self.openai_api_key.setEnabled(is_openai)
        self.openai_chat_model.setEnabled(is_openai)
        self.chat_model.setEnabled(source == "grok")
        self.generate_btn.setText("Populate Prompt in Browser" if is_manual else "Generate Video")

    def extract_last_frame_from_selected(self) -> None:
        index = self.video_picker.currentIndex()
        if index < 0 or index >= len(self.videos):
            QMessageBox.warning(self, "No Video Selected", "Select a generated video first.")
            return

        video_path = self.videos[index]["video_file_path"]
        frame_path = self._extract_last_frame(video_path)
        if frame_path is None:
            return

        if not self._copy_image_to_clipboard(frame_path):
            return

        self.last_extracted_frame_path = frame_path
        self.browser.setUrl(QUrl.fromLocalFile(str(frame_path)))
        self._append_log(
            "Extracted last frame and copied it to clipboard as an image. "
            f"Saved to: {frame_path}. You can now paste it into Grok's prompt tab."
        )

    def _extract_last_frame(self, video_path: str) -> Path | None:
        frame_path = DOWNLOAD_DIR / f"last_frame_{int(time.time() * 1000)}.png"
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
        self.continue_from_frame_reload_timeout_timer.start(2000)
        self._append_log(
            "Continue-from-last-frame: image pasted. Grok should auto-reload after upload; "
            "waiting for the new page before entering the continuation prompt..."
        )

    def _on_continue_reload_timeout(self) -> None:
        if not self.continue_from_frame_waiting_for_reload or not self.continue_from_frame_active:
            return
        self.continue_from_frame_waiting_for_reload = False
        self._append_log(
            "WARNING: Timed out waiting for upload-triggered reload. Continuing anyway with prompt submission."
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
        self._append_log(
            f"Continue-from-last-frame started for {self.continue_from_frame_target_count} iteration(s)."
        )
        self._append_log(f"Continue-from-last-frame source video selected: {latest_video}")
        self._start_continue_iteration()

    def show_browser_page(self) -> None:
        self.browser.setUrl(QUrl("https://grok.com/imagine"))
        self._append_log("Navigated embedded browser to grok.com/imagine.")

    def stitch_all_videos(self) -> None:
        if len(self.videos) < 2:
            QMessageBox.warning(self, "Need More Videos", "At least two videos are required to stitch.")
            return

        list_file = DOWNLOAD_DIR / f"stitch_list_{int(time.time() * 1000)}.txt"
        output_file = DOWNLOAD_DIR / f"stitched_{int(time.time() * 1000)}.mp4"

        concat_lines = []
        for video in self.videos:
            quoted_path = Path(video["video_file_path"]).as_posix().replace("'", "'\\''")
            concat_lines.append(f"file '{quoted_path}'")
        list_file.write_text("\n".join(concat_lines), encoding="utf-8")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_file),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-c:a",
                    "aac",
                    str(output_file),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            QMessageBox.critical(self, "ffmpeg Missing", "ffmpeg is required for stitching but was not found in PATH.")
            return
        except subprocess.CalledProcessError as exc:
            QMessageBox.critical(self, "Stitch Failed", exc.stderr[-800:] or "ffmpeg failed.")
            return
        finally:
            if list_file.exists():
                list_file.unlink()

        stitched_video = {
            "title": "Stitched Video",
            "prompt": "stitched",
            "resolution": "mixed",
            "video_file_path": str(output_file),
            "source_url": "local-stitch",
        }
        self.on_video_finished(stitched_video)

    def upload_selected_to_youtube(self) -> None:
        index = self.video_picker.currentIndex()
        if index < 0 or index >= len(self.videos):
            QMessageBox.warning(self, "No Video Selected", "Select a video to upload first.")
            return

        video_path = self.videos[index]["video_file_path"]
        title, description, accepted = self._show_youtube_upload_dialog()
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
                tags=["grok", "ai", "generated-video"],
            )
        except Exception as exc:
            QMessageBox.critical(self, "YouTube Upload Failed", str(exc))
            self._append_log(f"ERROR: YouTube upload failed: {exc}")
        else:
            self._append_log(f"YouTube upload complete. Video ID: {video_id}")
            QMessageBox.information(self, "YouTube Upload Complete", f"Video uploaded successfully. ID: {video_id}")
        finally:
            self.upload_youtube_btn.setEnabled(True)

    def _show_youtube_upload_dialog(self) -> tuple[str, str, bool]:
        dialog = QDialog(self)
        dialog.setWindowTitle("YouTube Upload Details")
        dialog_layout = QVBoxLayout(dialog)

        dialog_layout.addWidget(QLabel("YouTube Title"))
        title_input = QLineEdit()
        title_input.setText("AI Generated Video")
        dialog_layout.addWidget(title_input)

        dialog_layout.addWidget(QLabel("YouTube Description"))
        description_input = QPlainTextEdit()
        description_input.setPlaceholderText("Describe this upload...")
        dialog_layout.addWidget(description_input)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        dialog_layout.addWidget(button_box)

        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        return title_input.text().strip(), description_input.toPlainText().strip(), accepted


if __name__ == "__main__":
    _configure_qtwebengine_runtime()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
