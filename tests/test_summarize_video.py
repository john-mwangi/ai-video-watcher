"""Tests the /summarise_video endpoint"""

import os

import requests
from dotenv import find_dotenv, load_dotenv

load_dotenv()
APP_ENV = os.environ.get("APP_ENV")

load_dotenv(find_dotenv(f"{APP_ENV}.env"))

username = os.environ.get("_USERNAME")
password = os.environ.get("_PASSWORD")


def get_access_token(user, pwd):
    url = "http://0.0.0.0:12000/api/v1/token"

    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "grant_type": "",
        "username": user,
        "password": pwd,
        "scope": "",
        "client_id": "",
        "client_secret": "",
    }

    response = requests.post(url, headers=headers, data=data)
    token = response.json()["access_token"]
    return token


def test_summarise_video():
    token = get_access_token(username, password)
    header = {"Authorization": f"Bearer {token}"}
    url = "http://0.0.0.0:12000/api/v1/summarize_video"
    video_1 = "https://www.youtube.com/watch?v=TRjq7t2Ms5I"
    video_2 = "https://www.youtube.com/watch?v=IUTFrexghsQ"

    data = {
        "channels": [],
        "videos": [video_1, video_2],
        "limit_transcript": 0.25,
        "top_n": 2,
        "sort_by": "newest",
    }

    response = requests.post(url=url, json=data, headers=header)

    assert response.status_code == 200
