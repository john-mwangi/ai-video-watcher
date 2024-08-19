import os

import pytest
import requests
from dotenv import find_dotenv, load_dotenv

from video_summarizer.backend.configs.config import ApiSettings
from video_summarizer.backend.main import load_urls
from video_summarizer.frontend.utils import validate_url

load_dotenv()

app_env = os.environ.get("APP_ENV")
env_file = f"{app_env}.env"
load_dotenv(find_dotenv(filename=env_file))


endpoint = os.environ.get("ENDPOINT")
uname = os.environ.get("_USERNAME")
pwd = os.environ.get("_PASSWORD")
api_prefix = ApiSettings.load_settings().api_prefix
token_method = ApiSettings.load_settings().token_method
in_pipeline = True if os.environ.get("CIRCLECI") is not None else False

token_url = f"{endpoint}{api_prefix}{token_method}"
video_1 = "https://www.youtube.com/watch?v=TRjq7t2Ms5I"
video_2 = "https://www.youtube.com/watch?v=JEBDfGqrAUA"


def get_access_token(username: str, password: str):
    token_headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    token_json = {
        "grant_type": "",
        "username": username,
        "password": password,
        "scope": "",
        "client_id": "",
        "client_secret": "",
    }

    token_response = requests.post(
        url=token_url, headers=token_headers, data=token_json
    )

    return token_response


@pytest.mark.skipif(condition=in_pipeline, reason="Not applicable")
def test_authentication(username: str = uname, password: str = pwd):
    """Tests then authentication/token method"""

    token_response = get_access_token(username, password)
    assert token_response.status_code == 200


@pytest.mark.skipif(condition=in_pipeline, reason="Not applicable")
def test_endpoint(method: str = "/summarize_video"):
    """Tests the core endpoint's method"""

    token_response = get_access_token(uname, pwd)
    token_data = token_response.json()
    access_token = token_data["access_token"]

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    data = {
        "channels": [],
        "videos": [video_1, video_2],
        "limit_transcript": 0.25,
        "top_n": 2,
        "sort_by": "newest",
    }

    url = f"{endpoint}{api_prefix}{method}"
    response = requests.post(url, json=data, headers=headers)
    assert response.status_code == 200


def test_urls_from_channel():
    "Test the number of URLs extracted from the frontend"

    channels: list = []
    videos: list = []
    top_n: int = 3
    video_urls = {"channels": channels, "top_n": top_n, "videos": videos}

    yt_urls = load_urls(video_urls, sort_by="newest")
    assert len(yt_urls) == 0 and isinstance(yt_urls, set)

    channels = [
        "https://www.youtube.com/@ArjanCodes",
        "https://www.youtube.com/@CBSNews",
    ]
    videos = [
        "https://www.youtube.com/watch?v=JEBDfGqrAUA",
        "https://www.youtube.com/watch?v=TRjq7t2Ms5I",
    ]
    videos_len = len(videos)

    video_urls.update({"channels": channels, "videos": videos})

    yt_urls = load_urls(video_urls, sort_by="newest")

    assert len(yt_urls) == len(channels) * top_n + videos_len and isinstance(
        yt_urls, set
    )


def test_url_validation():
    "Test URL format & domain"
    assert (
        validate_url("https://docs.pydantic.dev/latest/api/networks/") == False
    )

    assert validate_url("xxx") == False

    assert validate_url("docs.pydantic.dev/latest/api/networks/") == False

    assert validate_url(None) == False

    assert validate_url("") == False

    assert validate_url("https://www.youtube.com") == True


def test_save_summaries():
    "Test saved data includes all video keys"
    pass


def test_api_response():
    "Test response includes all video keys"
    pass


def test_transcription():
    "Test that videos transcript is correctly generated with all keys"
    pass


def test_genai_summary():
    "Test that a summary from the Gen AI tool is being generated"
    pass
