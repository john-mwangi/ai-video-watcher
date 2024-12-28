"""Tests the /summarise_video endpoint"""

import os

import requests
from dotenv import find_dotenv, load_dotenv

from video_summarizer.backend.utils.utils import get_mongodb_client

load_dotenv()
APP_ENV = os.environ.get("APP_ENV")

load_dotenv(find_dotenv(f"{APP_ENV}.env"))

username = os.environ.get("_USERNAME")
password = os.environ.get("_PASSWORD")
video_id = "IUTFrexghsQ"
limit_transcript = 0

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

def delete_video(video_id: str = video_id) -> None:
    """Deletes a video from Mongodb"""
    
    collection_name = "summaries"
    client, db_name = get_mongodb_client()
    
    db = client[db_name]
    collection = db[collection_name]
    
    collection.delete_many(filter={"video_id": video_id})

def test_summarise_new_video():
    token = get_access_token(username, password)
    header = {"Authorization": f"Bearer {token}"}
    url = "http://0.0.0.0:12000/api/v1/summarize_video"
    video = f"https://www.youtube.com/watch?v={video_id}"

    data = {
        "channels": [],
        "videos": [video],
        "limit_transcript": limit_transcript,
        "top_n": 2,
        "sort_by": "newest",
    }

    delete_video(video_id)
    response = requests.post(url=url, json=data, headers=header)
    status = response.json()["status"]

    assert response.status_code == 200
    assert status == "VIDEO_SUMMARISED_SUCCESSFULLY"
    
def test_summarise_existing_video():
    token = get_access_token(username, password)
    header = {"Authorization": f"Bearer {token}"}
    url = "http://0.0.0.0:12000/api/v1/summarize_video"
    video = f"https://www.youtube.com/watch?v={video_id}"

    data = {
        "channels": [],
        "videos": [video],
        "limit_transcript": limit_transcript,
        "top_n": 2,
        "sort_by": "newest",
    }

    response = requests.post(url=url, json=data, headers=header)
    status = response.json()["status"]

    assert response.status_code == 200
    assert status == "VIDEO_RETRIEVED_SUCCESSFULLY"
