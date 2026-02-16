from __future__ import annotations

import os
import platform
import re
import json
import sys
import time
from pathlib import Path

import requests


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:  # noqa: BLE001
        py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise RuntimeError(
            "Playwright (Python) is required for web automation. "
            "Install with: pip install -r requirements-web.txt && python -m playwright install chromium firefox webkit. "
            f"Current Python: {py} on {platform.system()}. "
            "If greenlet wheel build fails on Windows, use Python 3.11 or 3.12 (64-bit)."
        ) from exc
    return sync_playwright


def _get_browser_type(playwright):
    browser_name = os.getenv("GROK_PLAYWRIGHT_BROWSER", "chromium").strip().lower()
    supported = {
        "chromium": playwright.chromium,
        "firefox": playwright.firefox,
        "webkit": playwright.webkit,
    }
    if browser_name not in supported:
        supported_names = ", ".join(sorted(supported))
        raise RuntimeError(
            f"Unsupported GROK_PLAYWRIGHT_BROWSER='{browser_name}'. "
            f"Use one of: {supported_names}."
        )
    return supported[browser_name], browser_name


def _get_selectors() -> dict[str, str]:
    return {
        "imagine_url": os.getenv("GROK_IMAGINE_URL", "https://grok.com/imagine"),
        "prompt": os.getenv("GROK_IMAGINE_PROMPT_SELECTOR", "textarea"),
        "submit": os.getenv("GROK_IMAGINE_SUBMIT_SELECTOR", "button[type='submit'][aria-label='Submit']"),
        "video": os.getenv("GROK_IMAGINE_VIDEO_SELECTOR", "video"),
    }


def _resolve_prompt_locator(page, preferred_selector: str, timeout_s: int):
    candidates = [
        preferred_selector,
        "textarea",
        "input[placeholder*='Type to imagine' i]",
        "textarea[placeholder*='Type to imagine' i]",
        "[contenteditable='true'][aria-label*='Type to imagine' i]",
        "[contenteditable='true'][data-placeholder*='Type to imagine' i]",
        "[contenteditable='true']",
        "input[type='text']",
    ]

    last_error = None
    for selector in candidates:
        locator = page.locator(selector).first
        try:
            locator.wait_for(state="visible", timeout=max(2000, int(timeout_s * 1000 / len(candidates))))
            return locator, selector
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    raise RuntimeError(
        "Could not find Imagine prompt input. "
        "Set GROK_IMAGINE_PROMPT_SELECTOR to the exact selector for your page "
        "(e.g., input[placeholder*='Type to imagine'])."
    ) from last_error


def _fill_prompt(locator, prompt: str) -> None:
    tag_name = (locator.evaluate("el => el.tagName") or "").lower()
    is_editable = bool(locator.evaluate("el => !!el.isContentEditable"))

    if tag_name in {"textarea", "input"}:
        locator.fill(prompt)
        return

    if is_editable:
        locator.click()
        locator.evaluate("el => { el.innerHTML=''; el.textContent=''; }")
        locator.type(prompt, delay=10)
        return

    # fallback: attempt generic fill then type
    try:
        locator.fill(prompt)
    except Exception:  # noqa: BLE001
        locator.click()
        locator.type(prompt, delay=10)


def _resolve_submit_locator(page, preferred_selector: str, timeout_s: int):
    candidates = [
        preferred_selector,
        "button[type='submit'][aria-label='Submit']",
        "form button[type='submit']",
        "button[aria-label='Submit']",
        "button:has(svg)",
        "button:has-text('Generate')",
    ]

    last_error = None
    for selector in candidates:
        locator = page.locator(selector).first
        try:
            locator.wait_for(state="visible", timeout=max(2000, int(timeout_s * 1000 / len(candidates))))
            locator.wait_for(state="attached", timeout=2000)
            return locator, selector
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    raise RuntimeError(
        "Could not find Imagine submit button. "
        "Set GROK_IMAGINE_SUBMIT_SELECTOR to the exact submit selector for your page "
        "(e.g., button[type='submit'][aria-label='Submit'])."
    ) from last_error


def _click_submit(locator, timeout_s: int) -> None:
    locator.wait_for(state="visible", timeout=max(2000, timeout_s * 1000))

    # Grok keeps the submit button disabled until prompt input is accepted.
    locator.page.wait_for_function(
        "el => !!el && !el.disabled",
        arg=locator.element_handle(),
        timeout=max(3000, timeout_s * 1000),
    )

    locator.scroll_into_view_if_needed()
    locator.click()


def manual_login_and_save(storage_state_path: Path, timeout_s: int = 300) -> None:
    sync_playwright = _require_playwright()
    selectors = _get_selectors()
    storage_state_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser_type, browser_name = _get_browser_type(p)
        browser = browser_type.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(selectors["imagine_url"], wait_until="domcontentloaded")
        _resolve_prompt_locator(page, selectors["prompt"], timeout_s=timeout_s)
        context.storage_state(path=str(storage_state_path))
        print(f"Saved Grok web login session with Playwright browser: {browser_name}")
        browser.close()


def generate_video_via_web(
    storage_state_path: Path,
    prompt: str,
    output_path: Path,
    timeout_s: int = 360,
) -> Path:
    sync_playwright = _require_playwright()
    selectors = _get_selectors()

    if not storage_state_path.exists():
        raise RuntimeError("No saved web login session found. Run manual web login first.")

    with sync_playwright() as p:
        browser_type, browser_name = _get_browser_type(p)
        browser = browser_type.launch(headless=True)
        context = browser.new_context(storage_state=str(storage_state_path))
        page = context.new_page()
        page.goto(selectors["imagine_url"], wait_until="domcontentloaded")

        prompt_locator, used_selector = _resolve_prompt_locator(page, selectors["prompt"], timeout_s=timeout_s)
        _fill_prompt(prompt_locator, prompt)

        submit_locator, used_submit_selector = _resolve_submit_locator(page, selectors["submit"], timeout_s=timeout_s)
        _click_submit(submit_locator, timeout_s=timeout_s)

        page.wait_for_selector(selectors["video"], timeout=timeout_s * 1000)
        video_el = page.locator(selectors["video"]).first
        video_url = video_el.get_attribute("src")
        if not video_url:
            video_url = page.locator(f"{selectors['video']} source").first.get_attribute("src")
        if not video_url:
            raise RuntimeError(
                "Generated video element found but no source URL was discovered. "
                f"Prompt selector used: {used_selector}. Submit selector used: {used_submit_selector}"
            )

        response = context.request.get(video_url)
        if not response.ok:
            raise RuntimeError(f"Could not download generated video from web session: {response.status}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.body())
        print(f"Downloaded Grok video using Playwright browser: {browser_name}")
        browser.close()

    return output_path


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return cleaned or "flow"


def _openai_training_summary(access_token: str, model: str, payload: dict) -> dict:
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
    endpoint = f"{api_base}/responses"
    instruction = (
        "You are an automation analyst. Convert the captured browser training log into a deterministic replay plan. "
        "Return strict JSON matching this schema: "
        '{"process_name": str, "goal": str, "steps": [{"index": int, "action": "click|fill|press", "selector": str, '
        '"value": str, "notes": str}], "risks": [str]}. '
        "Keep selectors CSS-compatible and preserve action order."
    )
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": instruction}]},
                {"role": "user", "content": [{"type": "input_text", "text": json.dumps(payload)}]},
            ],
            "store": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    text = ""
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text += content.get("text", "")
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def train_browser_flow(
    start_url: str,
    output_dir: Path,
    storage_state_path: Path | None = None,
    timeout_s: int = 900,
    screenshot_every_event: bool = True,
) -> Path:
    """Record a browser flow by observing user clicks/inputs and persist the raw trace.

    User flow: call function, guide the browser manually, then close the browser tab/window to stop training.
    """

    sync_playwright = _require_playwright()
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    events: list[dict] = []
    started_at = time.time()
    event_counter = {"value": 0}

    with sync_playwright() as p:
        browser_type, browser_name = _get_browser_type(p)
        context_options = {}
        if storage_state_path and storage_state_path.exists():
            context_options["storage_state"] = str(storage_state_path)
        browser = browser_type.launch(headless=False)
        context = browser.new_context(**context_options)
        page = context.new_page()

        def _record_event(source, event):
            event_counter["value"] += 1
            event["index"] = event_counter["value"]
            event["source"] = source
            event["ts"] = round(time.time() - started_at, 3)
            if screenshot_every_event:
                screenshot_name = f"step-{event['index']:03d}.png"
                screenshot_path = screenshots_dir / screenshot_name
                try:
                    page.screenshot(path=str(screenshot_path), full_page=True)
                    event["screenshot"] = str(Path("screenshots") / screenshot_name)
                except Exception:  # noqa: BLE001
                    event["screenshot"] = ""
            events.append(event)

        context.expose_binding("pyRecordEvent", _record_event)

        page.add_init_script(
            r"""
            (() => {
              const cssPath = (el) => {
                if (!el || !(el instanceof Element)) return '';
                const parts = [];
                let node = el;
                while (node && node.nodeType === Node.ELEMENT_NODE && parts.length < 6) {
                  let part = node.nodeName.toLowerCase();
                  if (node.id) {
                    part += `#${node.id}`;
                    parts.unshift(part);
                    break;
                  }
                  const className = (node.className || '').toString().trim().split(/\s+/).filter(Boolean).slice(0,2).join('.');
                  if (className) part += `.${className}`;
                  const parent = node.parentElement;
                  if (parent) {
                    const siblings = Array.from(parent.children).filter((x) => x.nodeName === node.nodeName);
                    if (siblings.length > 1) {
                      part += `:nth-of-type(${siblings.indexOf(node) + 1})`;
                    }
                  }
                  parts.unshift(part);
                  node = node.parentElement;
                }
                return parts.join(' > ');
              };

              const scrub = (value) => (value || '').toString().slice(0, 400);
              document.addEventListener('click', (event) => {
                const el = event.target;
                window.pyRecordEvent({
                  action: 'click',
                  selector: cssPath(el),
                  text: scrub(el && el.innerText),
                  tag: (el && el.tagName || '').toLowerCase(),
                });
              }, true);

              document.addEventListener('change', (event) => {
                const el = event.target;
                if (!el) return;
                const tag = (el.tagName || '').toLowerCase();
                if (!['input', 'textarea', 'select'].includes(tag)) return;
                window.pyRecordEvent({
                  action: 'fill',
                  selector: cssPath(el),
                  value: scrub(el.value),
                  tag,
                });
              }, true);

              document.addEventListener('keydown', (event) => {
                if (!event.key || !['Enter', 'Tab', 'Escape'].includes(event.key)) return;
                const el = event.target;
                window.pyRecordEvent({
                  action: 'press',
                  selector: cssPath(el),
                  value: event.key,
                  tag: (el && el.tagName || '').toLowerCase(),
                });
              }, true);
            })();
            """
        )

        page.goto(start_url, wait_until="domcontentloaded")
        print("Training started. Interact with the browser, then close the browser window to stop training.")
        page.wait_for_event("close", timeout=timeout_s * 1000)
        context.close()
        browser.close()

        trace = {
            "start_url": start_url,
            "browser": browser_name,
            "created_at": int(time.time()),
            "events": events,
        }
        trace_path = output_dir / "raw_training_trace.json"
        trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        return trace_path


def build_trained_process(trace_path: Path, access_token: str, model: str = "gpt-5.1-codex") -> Path:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    summary = _openai_training_summary(access_token=access_token, model=model, payload=trace)
    process_name = _slugify(summary.get("process_name") or "trained-process")
    output_path = trace_path.parent / f"{process_name}.process.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def run_trained_process(
    process_path: Path,
    start_url: str,
    output_dir: Path,
    storage_state_path: Path | None = None,
    timeout_s: int = 180,
) -> Path:
    sync_playwright = _require_playwright()
    process = json.loads(process_path.read_text(encoding="utf-8"))
    steps = process.get("steps", [])
    output_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser_type, _browser_name = _get_browser_type(p)
        context_options = {}
        if storage_state_path and storage_state_path.exists():
            context_options["storage_state"] = str(storage_state_path)
        browser = browser_type.launch(headless=False)
        context = browser.new_context(**context_options)
        page = context.new_page()
        page.goto(start_url, wait_until="domcontentloaded")

        execution_log: list[dict] = []
        for idx, step in enumerate(steps, start=1):
            selector = step.get("selector", "")
            action = step.get("action", "")
            value = step.get("value", "")
            status = "ok"
            error = ""
            try:
                locator = page.locator(selector).first
                locator.wait_for(state="visible", timeout=timeout_s * 1000)
                if action == "click":
                    locator.click()
                elif action == "fill":
                    locator.fill(value)
                elif action == "press":
                    locator.press(value or "Enter")
                else:
                    status = "skipped"
                    error = f"Unsupported action: {action}"
            except Exception as exc:  # noqa: BLE001
                status = "error"
                error = str(exc)

            shot = output_dir / f"run-step-{idx:03d}.png"
            try:
                page.screenshot(path=str(shot), full_page=True)
            except Exception:  # noqa: BLE001
                pass
            execution_log.append(
                {
                    "index": idx,
                    "action": action,
                    "selector": selector,
                    "value": value,
                    "status": status,
                    "error": error,
                    "screenshot": shot.name,
                }
            )

        report_path = output_dir / "run_report.json"
        report_path.write_text(json.dumps(execution_log, indent=2), encoding="utf-8")
        context.close()
        browser.close()
        return report_path
