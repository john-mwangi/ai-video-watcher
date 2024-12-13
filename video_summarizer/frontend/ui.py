import streamlit as st
from streamlit_tags import st_tags

from video_summarizer.backend.configs.config import WWW_DIR
from video_summarizer.frontend import utils
from video_summarizer.frontend.server import format_summary, main

# https://getbootstrap.com/docs/5.0/getting-started/introduction/
css = """<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">"""
st.markdown(css, unsafe_allow_html=True)

st.sidebar.title("ChatGPT Video Summarizer")

st.sidebar.image(f"{WWW_DIR}/Gemini_Generated_Image.jpeg", width=None)
st.sidebar.divider()

sort_by = st.sidebar.selectbox(
    label="Sort By",
    options=["Newest", "Popular", "Oldest"],
    help="Criteria to sort channel videos",
)

top_n = st.sidebar.number_input(
    label="Top N Videos",
    value=2,
    step=1,
    min_value=1,
    max_value=5,
    help="Retrieves this number of video from a channel to summarise",
)

limit_transcript = st.sidebar.number_input(
    label="Limit Transcript",
    value=0.25,
    step=0.1,
    help="Portion of the video transcript to summarise",
)

submit = st.sidebar.button(label="Submit")

urls: list[str] = st_tags(label="### YOUTUBE VIDEOS")
st.write("_Enter a list of YouTube channels or videos._")
st.divider()


def render_content(summaries):
    """Displays the content of several videos"""

    for summary in summaries:
        for video in summary:
            result, is_html = format_summary(video, return_html=False)

            st.markdown("".join(result), unsafe_allow_html=is_html)

            if is_html:
                st.write("\n\n")

            else:
                cols = st.columns(8)

                watch_btn = cols[0].button(
                    label="Watch",
                    help="Watch this video on YouTube",
                    key="watch_" + video["video_id"],
                    on_click=user_action_dialog,
                    kwargs={"action": "watch"},
                )

                chat_btn = cols[1].button(
                    label="Chat",
                    help="Chat this video with ChatGPT",
                    key="chat_" + video["video_id"],
                    on_click=user_action_dialog,
                    kwargs={"action": "chat"},
                )

                st.divider()


@st.dialog(title="Your video")
def user_action_dialog(action: str, url: str = None):
    """Create a modal to handle a user action"""
    st.write(f"You are about to {action} a video...")


if "result" in st.session_state:
    render_content(st.session_state.result)

if submit:
    url_validations = [utils.validate_url(url) for url in urls]
    is_valid = False if not url_validations else all(url_validations)

    if not is_valid:
        st.markdown("One of the urls submitted was invalid or not supported")

    else:
        channels, videos = utils.extract_channels_and_videos(urls)
        data = {
            "channels": channels,
            "videos": videos,
            "limit_transcript": limit_transcript,
            "top_n": top_n,
            "sort_by": sort_by.lower(),
        }

        response = main(method="/summarize_video", data=data)
        if response.status_code in (401, 403):
            st.error("Incorrect username or password!")
        else:
            content = response.json()
            summaries = content.get("data").get("summaries")
            st.session_state.result = summaries
            render_content(summaries)
