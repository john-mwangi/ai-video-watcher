"""This module tests the functionality for extracting a transcript from a YouTube video"""

from video_summarizer.backend.src import extract_transcript

URL = "https://www.youtube.com/watch?v=JEBDfGqrAUA"
VIDEO_ID = "JEBDfGqrAUA"


def test_get_video():
    video_id = extract_transcript.get_video_id(URL)
    assert video_id == VIDEO_ID


def test_get_transcript_from_db():
    result = extract_transcript.get_transcript_from_db(VIDEO_ID)
    assert len(result) == 5

    result = extract_transcript.get_transcript_from_db("VIDEO_ID")
    assert result is None


def test_get_video_transcript():
    transcript = extract_transcript.get_video_transcript(VIDEO_ID)
    assert len(transcript) == 604
