import yaml
from pydantic import BaseModel, HttpUrl, ValidationError

from video_summarizer.backend.configs.config import params_path
from video_summarizer.backend.utils.utils import logger


class Url(BaseModel):
    url: HttpUrl


def validate_url(url: str) -> bool:
    with open(params_path, mode="r") as f:
        supported_websites = yaml.safe_load(f).get("supported_websites")

    is_valid = False

    try:
        ValidUrl = Url(url=url)
        if (
            ValidUrl.url.host not in supported_websites
            or supported_websites is None
        ):
            logger.warning("This website is not supported")
        else:
            is_valid = True
    except ValidationError:
        logger.warning(f"Invalid url: {url}")

    return is_valid


def extract_channels_and_videos(urls: list[str]):
    channels = []
    videos = []

    for url in urls:
        if url.__contains__("@"):
            channels.append(url)
        else:
            videos.append(url)

    return channels, videos


if __name__ == "__main__":
    assert (
        validate_url("https://docs.pydantic.dev/latest/api/networks/") == True
    )

    assert validate_url("xxx") == False

    assert validate_url("docs.pydantic.dev/latest/api/networks/") == False

    assert validate_url(None) == False
