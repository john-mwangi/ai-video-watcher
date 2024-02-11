from enum import Enum
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

params_path = ROOT_DIR / f"video_summarizer/configs/params.yaml"
video_keys = ["video_id", "video_url", "video_title", "summary"]
class statuses(Enum):
    SUCCESS = 200
    ERROR = 400
    NOT_FOUND = 404


class Params(BaseSettings):
    MODEL: str
    CHUNK_SIZE: int
    SUMMARY_LIMIT: int
    BULLETS: int
    BATCH_CHUNKS: int

    def load(path: Path = params_path):
        with open(path, mode="r") as f:
            params = yaml.safe_load(f).get("model_params")

        return Params(**params)


if __name__ == "__main__":
    print(Params.load().BATCH_CHUNKS)
    print(Params.load())
    print(dict(Params.load()))
