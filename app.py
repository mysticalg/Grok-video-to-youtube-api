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
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
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

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)
CACHE_DIR = BASE_DIR / ".qtwebengine"
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
        for resolution in ["1280x720", "640x420"]:
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
    )

    def __init__(self, on_console_message, parent=None):
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
        self.manual_download_poll_timer = QTimer(self)
        self.manual_download_poll_timer.setSingleShot(True)
        self.manual_download_poll_timer.timeout.connect(self._poll_for_manual_video)
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

        left_layout.addWidget(QLabel("Generated Videos"))
        self.video_picker = QComboBox()
        self.video_picker.currentIndexChanged.connect(self.show_selected_video)
        left_layout.addWidget(self.video_picker)

        left_layout.addWidget(QLabel("Activity Log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        left_layout.addWidget(self.log)

        left_layout.addWidget(QLabel("Video Preview"))
        self.video_preview = QVideoWidget()
        self.video_preview.setMinimumHeight(220)
        left_layout.addWidget(self.video_preview)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_preview)

        self.browser = QWebEngineView()
        self.browser.setPage(FilteredWebEnginePage(self._append_log, self.browser))
        browser_profile = self.browser.page().profile()
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

        self.browser.setUrl(QUrl("https://grok.com"))
        self.browser.page().profile().downloadRequested.connect(self._on_browser_download_requested)

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

        self.generate_btn.setEnabled(False)
        self.worker = GenerateWorker(config, prompt_config, self.count.value())
        self.worker.status.connect(self._append_log)
        self.worker.finished_video.connect(self.on_video_finished)
        self.worker.failed.connect(self.on_generation_error)
        self.worker.finished.connect(lambda: self.generate_btn.setEnabled(True))
        self.worker.start()

    def _start_manual_browser_generation(self, prompt: str, count: int) -> None:
        self.manual_generation_queue = [{"prompt": prompt, "variant": 1}]
        self.generate_btn.setEnabled(False)
        self._append_log(
            "Manual mode now reuses the current browser page exactly as-is. "
            "No navigation or reload will happen."
        )
        if count > 1:
            self._append_log("Manual mode populates one prompt per click; ignoring Count and filling once.")
        self._append_log("Attempting to populate the visible 'Type to imagine' prompt box on the current page...")
        self._submit_next_manual_variant()

    def _submit_next_manual_variant(self) -> None:
        if not self.manual_generation_queue:
            self.generate_btn.setEnabled(True)
            self._append_log("Manual browser generation complete.")
            return

        item = self.manual_generation_queue.pop(0)
        prompt = item["prompt"]
        variant = item["variant"]
        self.pending_manual_variant_for_download = variant
        self._append_log(f"Populating prompt for manual variant {variant} in browser and submitting...")

        escaped_prompt = repr(prompt)
        script = rf"""
            (() => {{
                try {{
                    const prompt = {escaped_prompt};
                    const promptSelectors = [
                        "textarea[placeholder*='Type to imagine' i]",
                        "input[placeholder*='Type to imagine' i]",
                        "div.tiptap.ProseMirror[contenteditable='true']",
                        "[contenteditable='true'][aria-label*='Type to imagine' i]",
                        "[contenteditable='true'][data-placeholder*='Type to imagine' i]"
                    ];

                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const input = promptSelectors
                        .flatMap((selector) => [...document.querySelectorAll(selector)])
                        .find((el) => isVisible(el));
                    if (!input) return {{ ok: false, error: "Type to imagine input not found" }};

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
                        const valueSetter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
                        if (valueSetter) valueSetter.call(input, prompt);
                        else input.value = prompt;
                        input.dispatchEvent(new Event("input", {{ bubbles: true }}));
                        input.dispatchEvent(new Event("change", {{ bubbles: true }}));
                    }}

                    const typedValue = input.isContentEditable ? (input.textContent || "") : (input.value || "");
                    if (!typedValue.trim()) return {{ ok: false, error: "Prompt field did not accept text" }};

                    const interactiveSelector = "button, [role='button'], [role='tab'], [role='option'], [role='menuitemradio'], [role='radio'], label, span, div";
                    const clickableAncestor = (el) => el?.closest?.("button, [role='button'], [role='tab'], [role='option'], [role='menuitemradio'], [role='radio'], label") || el;
                    const matchesAny = (text, patterns) => patterns.some((pattern) => pattern.test(text));
                    const visibleTextElements = (root = document) => [...root.querySelectorAll(interactiveSelector)]
                        .filter((el) => isVisible(el) && (el.textContent || "").trim());

                    const clickByText = (patterns, root = document) => {{
                        const candidate = visibleTextElements(root).find((el) => matchesAny((el.textContent || "").trim(), patterns));
                        const target = clickableAncestor(candidate);
                        if (!target) return false;
                        target.click();
                        return true;
                    }};

                    const composer = input.closest("form") || input.parentElement || document;
                    clickByText([/video/i], composer) || clickByText([/video/i]);
                    clickByText([/720\s*p/i, /1280\s*[x×]\s*720/i], composer) || clickByText([/720\s*p/i, /1280\s*[x×]\s*720/i]);
                    clickByText([/10\s*s(ec(onds?)?)?/i], composer) || clickByText([/10\s*s(ec(onds?)?)?/i]);

                    const submitSelectors = [
                        "button[type='submit'][aria-label='Submit']",
                        "button[type='submit']",
                        "button[aria-label='Submit']"
                    ];

                    const submitButton = submitSelectors
                        .flatMap((selector) => [...document.querySelectorAll(selector)])
                        .find((el) => isVisible(el) && !el.disabled)
                        || visibleTextElements(composer)
                            .map((el) => clickableAncestor(el))
                            .find((el) => el && isVisible(el) && !el.disabled && matchesAny((el.textContent || "").trim(), [/submit/i, /generate/i, /create/i]));
                    if (!submitButton) return {{ ok: false, error: "Submit button not found or disabled" }};

                    submitButton.click();

                    return {{ ok: true, filledLength: typedValue.length, submitted: true, optionsRequested: ["video", "720p", "10s"] }};
                }} catch (err) {{
                    return {{ ok: false, error: String(err && err.stack ? err.stack : err) }};
                }}
            }})()
        """

        def after_submit(result):
            if not isinstance(result, dict) or not result.get("ok"):
                self.pending_manual_variant_for_download = None
                error_detail = result.get("error") if isinstance(result, dict) else result
                self._append_log(f"ERROR: Manual prompt fill failed for variant {variant}: {error_detail!r}")
                self.generate_btn.setEnabled(True)
                return
            self._append_log(f"Prompt populated for variant {variant}; requested Video + 720p + 10s, submitted, and now waiting for generation to auto-download.")
            self._trigger_browser_video_download(variant)

        self.browser.page().runJavaScript(script, after_submit)

    def _trigger_browser_video_download(self, variant: int) -> None:
        self.manual_download_deadline = time.time() + 420
        self.manual_download_poll_timer.start(0)

    def _poll_for_manual_video(self) -> None:
        variant = self.pending_manual_variant_for_download
        if variant is None:
            return

        deadline = self.manual_download_deadline or 0
        if time.time() > deadline:
            self.pending_manual_variant_for_download = None
            self._append_log(f"ERROR: Variant {variant} did not produce a downloadable video in time.")
            self.generate_btn.setEnabled(True)
            return

        script = """
            (() => {
                const video = document.querySelector("video");
                const source = document.querySelector("video source");
                return (video && (video.currentSrc || video.src)) || (source && source.src) || "";
            })()
        """

        def after_poll(result):
            current_variant = self.pending_manual_variant_for_download
            if current_variant is None:
                return

            src = result if isinstance(result, str) else ""
            if not src:
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
            self._append_log(f"Variant {current_variant} video detected; browser download requested.")

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
                self.pending_manual_variant_for_download = None
                self._submit_next_manual_variant()
            elif state == download.DownloadState.DownloadInterrupted:
                self._append_log(f"ERROR: Download interrupted for manual variant {variant}.")
                self.pending_manual_variant_for_download = None
                self.generate_btn.setEnabled(True)

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
        self.player.play()
        self._append_log(f"Previewing video: {file_path}")

    def open_local_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select video", str(DOWNLOAD_DIR), "Videos (*.mp4 *.mov *.webm)")
        if not file_path:
            return
        self._preview_video(file_path)
        self._append_log(f"Opened local file: {file_path}")

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
        frame_path = DOWNLOAD_DIR / f"last_frame_{int(time.time() * 1000)}.png"

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
            return
        except subprocess.CalledProcessError as exc:
            QMessageBox.critical(self, "Frame Extraction Failed", exc.stderr[-800:] or "ffmpeg failed.")
            return

        image = QImage(str(frame_path))
        if image.isNull():
            QMessageBox.critical(self, "Frame Extraction Failed", "Frame image could not be loaded.")
            return

        mime = QMimeData()
        mime.setImageData(image)
        mime.setText(str(frame_path))
        QGuiApplication.clipboard().setMimeData(mime)

        self.browser.setUrl(QUrl.fromLocalFile(str(frame_path)))
        self._append_log(
            "Extracted last frame and copied it to clipboard as an image. "
            f"Saved to: {frame_path}. You can now paste it into Grok's 'Type to imagine' tab."
        )


if __name__ == "__main__":
    _configure_qtwebengine_runtime()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
