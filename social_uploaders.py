from __future__ import annotations

import time
from pathlib import Path

import requests

GRAPH_API_BASE = "https://graph.facebook.com/v21.0"


def upload_video_to_facebook_page(
    page_id: str,
    access_token: str,
    video_path: str,
    title: str,
    description: str,
) -> str:
    if not page_id.strip():
        raise ValueError("Facebook Page ID is required.")
    if not access_token.strip():
        raise ValueError("Facebook access token is required.")

    with Path(video_path).open("rb") as video_file:
        response = requests.post(
            f"{GRAPH_API_BASE}/{page_id}/videos",
            data={
                "access_token": access_token,
                "title": title,
                "description": description,
                "published": "false",
            },
            files={"source": video_file},
            timeout=600,
        )

    if not response.ok:
        raise RuntimeError(f"Facebook upload failed: {response.status_code} {response.text[:500]}")

    payload = response.json()
    video_id = payload.get("id")
    if not video_id:
        raise RuntimeError(f"Facebook upload did not return a video id: {payload}")
    return str(video_id)


def upload_video_to_instagram_reels(
    ig_user_id: str,
    access_token: str,
    video_url: str,
    caption: str,
    publish_timeout_s: int = 180,
) -> str:
    if not ig_user_id.strip():
        raise ValueError("Instagram Business Account ID is required.")
    if not access_token.strip():
        raise ValueError("Instagram access token is required.")
    if not video_url.strip().lower().startswith(("http://", "https://")):
        raise ValueError("Instagram upload requires a public HTTP(S) video URL.")

    create_resp = requests.post(
        f"{GRAPH_API_BASE}/{ig_user_id}/media",
        data={
            "access_token": access_token,
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true",
        },
        timeout=120,
    )
    if not create_resp.ok:
        raise RuntimeError(f"Instagram container creation failed: {create_resp.status_code} {create_resp.text[:500]}")

    creation_id = create_resp.json().get("id")
    if not creation_id:
        raise RuntimeError(f"Instagram container id missing: {create_resp.text[:500]}")

    deadline = time.time() + publish_timeout_s
    while time.time() < deadline:
        status_resp = requests.get(
            f"{GRAPH_API_BASE}/{creation_id}",
            params={"access_token": access_token, "fields": "status_code"},
            timeout=60,
        )
        if not status_resp.ok:
            raise RuntimeError(f"Instagram status check failed: {status_resp.status_code} {status_resp.text[:500]}")

        status_code = (status_resp.json().get("status_code") or "").upper()
        if status_code in {"FINISHED", "PUBLISHED"}:
            break
        if status_code in {"ERROR", "EXPIRED"}:
            raise RuntimeError(f"Instagram media processing failed with status: {status_code}")
        time.sleep(3)
    else:
        raise RuntimeError("Instagram media container did not finish processing before timeout.")

    publish_resp = requests.post(
        f"{GRAPH_API_BASE}/{ig_user_id}/media_publish",
        data={"access_token": access_token, "creation_id": creation_id},
        timeout=120,
    )
    if not publish_resp.ok:
        raise RuntimeError(f"Instagram publish failed: {publish_resp.status_code} {publish_resp.text[:500]}")

    media_id = publish_resp.json().get("id")
    if not media_id:
        raise RuntimeError(f"Instagram publish did not return media id: {publish_resp.text[:500]}")
    return str(media_id)
