import os

import requests
from dotenv import find_dotenv, load_dotenv

from video_summarizer.backend.configs.config import ApiSettings

load_dotenv()
app_env = os.environ.get("APP_ENV")
env_file = f"{app_env}.env"
load_dotenv(find_dotenv(env_file))


def main(method: str, data: dict):
    endpoint = os.environ.get("ENDPOINT")
    prefix = ApiSettings.load_settings().api_prefix
    token_method = ApiSettings.load_settings().token_method

    url = f"{endpoint}{prefix}"
    token_url = f"{url}{token_method}"

    auth_headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    auth_data = {
        "grant_type": "",
        "username": os.environ.get("_USERNAME"),
        "password": os.environ.get("_PASSWORD"),
        "scope": "",
        "client_id": "",
        "client_secret": "",
    }

    auth_response = requests.post(
        token_url,
        headers=auth_headers,
        data=auth_data,
    )

    token = auth_response.json().get("access_token")

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    resp = requests.post(f"{url}{method}", headers=headers, json=data)

    return resp


def card(video_title, video_url, summary):
    """Generate Bootstrap card html content."""

    html = f"""
    <div class="card">
        <div class="card-body">
            <h5 class="card-title">{video_title}</h5>
            <h6 class="card-subtitle mb-2 text-muted">{video_url}</h6>
            <p class="card-text">{summary}</p>
            <a href="{video_url}" title="Watch this video on YouTube" class="btn btn-primary"><font color="white">Watch</font></a>
            <a href="#" title="Chat this video with ChatGPT" class="btn btn-primary"><font color="white">Chat</font></a>
        </div>
    </div>
    """
    return html


def clean_titles(k: str):
    s = k.strip().title().split("_")
    return " ".join(s)


def format_summary(video, return_html: bool = True) -> tuple[list[str], bool]:
    """Formats the content into a user friendly format.

    Args:
    ---
    response: Response from the API/database
    return_html: Whether to return html or markdown
    """
    if return_html:
        video_title = video.get("video_title")
        video_url = video.get("video_url")
        video_summary = video.get("summary")
        video_summary = video_summary.replace("\n", "<br>")

        r = card(
            video_title=video_title,
            video_url=video_url,
            summary=video_summary,
        )
        return r, return_html

    else:
        res = []
        for k, v in video.items():
            s = clean_titles(k)
            r = f"**{s}**: {v}\n\n"
            res.append(r)

        return res, return_html
