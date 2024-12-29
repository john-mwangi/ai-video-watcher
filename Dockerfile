FROM python:3.10-slim

ENV APP_HOME /app
ENV APP_NAME video_summarizer

WORKDIR ${APP_HOME}

COPY . .

# CMD ls -la && pwd
RUN apt-get update && \
    apt-get install -y gcc git && \
    pip install -r requirements.txt

CMD bash -c "\
    cd ${APP_NAME}/backend/ && \
    uvicorn api:app --host 0.0.0.0 --port 12000 & \
    cd ${APP_HOME}/${APP_NAME}/frontend/ && \
    streamlit run ui.py --server.address=0.0.0.0 --server.port=8501 & \
    wait \
    "

# DOCKER_BUILDKIT=1 docker build -f Dockerfile -t video_summarizer .
# docker run -p 8501:8501 -p 12000:12000 video_summarizer