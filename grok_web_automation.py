from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


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
