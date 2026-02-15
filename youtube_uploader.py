from __future__ import annotations

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def upload_video_to_youtube(
    client_secret_file: str,
    token_file: str,
    video_path: str,
    title: str,
    description: str,
    tags: list[str],
    youtube_api_key: str = "",
    category_id: str = "22",
) -> str:
    creds = None
    token_path = Path(token_file)

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())

    youtube = build("youtube", "v3", credentials=creds, developerKey=(youtube_api_key or None))

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": str(category_id or "22"),
            },
            "status": {"privacyStatus": "private"},
        },
        media_body=MediaFileUpload(video_path, chunksize=-1, resumable=True),
    )

    response = None
    while response is None:
        _, response = request.next_chunk()

    return response["id"]
