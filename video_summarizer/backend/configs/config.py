from enum import Enum
from pathlib import Path

import yaml
from fastapi import status
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
params_path = ROOT_DIR / f"video_summarizer/backend/configs/params.yaml"

video_keys = ["video_id", "video_url", "video_title", "summary"]


class statuses(Enum):
    SUCCESS = status.HTTP_200_OK
    ERROR = status.HTTP_400_BAD_REQUEST
    NOT_FOUND = status.HTTP_404_NOT_FOUND


class ModelParams(BaseSettings):
    MODEL: str
    CHUNK_SIZE: int
    SUMMARY_LIMIT: int
    BULLETS: int
    BATCH_CHUNKS: int

    def load(path: Path = params_path):
        with open(path, mode="r") as f:
            params = yaml.safe_load(f).get("model_params")

        return ModelParams(**params)


class ApiSettings(BaseSettings):
    api_prefix: str
    algorithm: str
    access_token_expire_minutes: int
    token_method: str

    def load_settings():
        with open(params_path, "r") as f:
            api_settings = yaml.safe_load(f)["endpoint"]
            return ApiSettings(**api_settings)


prompt_template = """system: You are a helpful assistant who provides useful summaries 
    to a video transcript. The format of the video transcript is `timestamp - dialogue`.

    user: {question}
    assistant:
    """

augmented_prompt = """system: You are a helpful assistant. Please answer the
    question using the context below:
    
    Context: 
    {context}
    
    user: {question}
    assistant:
    """

if __name__ == "__main__":
    print(ROOT_DIR)
    print(ModelParams.load().BATCH_CHUNKS)
    print(ModelParams.load())
    print(dict(ModelParams.load()))
