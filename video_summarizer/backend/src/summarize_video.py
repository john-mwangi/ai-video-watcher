from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from tqdm import tqdm

from video_summarizer.backend.configs import configs
from video_summarizer.backend.src.extract_transcript import (
    get_transcript_from_db,
    get_video_title,
)
from video_summarizer.backend.utils.utils import get_mongodb_client, logger


def init_model(template: str):
    """Initialise an LLM"""

    prompt_template = PromptTemplate(
        input_variables=["question"],
        template=template,
    )

    model = LLMChain(
        llm=ChatOpenAI(model=configs.ModelParams.load().MODEL),
        prompt=prompt_template,
    )

    return model


def chunk_a_list(data: list[str], chunk_size: int) -> list[list[str]]:
    """Converts a long list to a smaller one by combining its items"""

    result = []
    sublist = []

    for _, t in enumerate(data):
        if t.strip():
            sublist.append(t)
        if len(sublist) == chunk_size:
            result.append(sublist)
            sublist = []
    if sublist:
        result.append(sublist)

    return result


def check_if_summarised(video_id: str) -> tuple[bool, None | str]:
    """Checks if a video has already been summarised"""

    is_summarised = False
    data = None

    client, db = get_mongodb_client()

    with client:
        db = client[db]
        summaries = db.summaries
        result = summaries.find_one({"video_id": video_id})

    if result is not None:
        is_summarised = True

        data = {}
        for k in configs.video_keys:
            data[k] = result.get(k)

    return is_summarised, data


def summarize_transcript(transcript, bullets, model, limit) -> str:
    """Provides a summary for a video transcript"""

    question = f"""Consider the following video transcript: 
    
    {transcript} 
    
    Begin by providing a concise single sentence summary of the what 
    is being discussed in the transcript. Then follow it with bullet 
    points of the {bullets} most important points along with their associated 
    timestamps. The total number of words should not be more than {limit}.
    """

    summary = model.predict(question=question)
    return summary


def summarize_list_of_transcripts(transcripts, bullets, model, limit):
    """Summarize a list of transcripts into a list of summaries"""

    summaries = [
        summarize_transcript(transcript, bullets, model, limit)
        for transcript in tqdm(transcripts)
    ]
    return summaries


def summarize_list_of_summaries(summaries, chunk_size, bullets, model, limit):
    """Summarise a list of summaries into a summary"""

    # group list of summaries into chunks
    chunked_summaries = [
        summaries[i : i + chunk_size]
        for i in range(0, len(summaries), chunk_size)
    ]

    # join chunks into a single prompt/string and send to model
    combined_summaries = [
        summarize_transcript(" ".join(chunk), bullets, model, limit)
        for chunk in tqdm(chunked_summaries)
    ]

    # repeat
    return summarize_transcript(
        " ".join(combined_summaries), bullets, model, limit
    )


def save_summary(data: dict | list[dict]):
    """Saves data to a MongoDB database"""

    client, _DB = get_mongodb_client()

    with client:
        db = client[_DB]  # Establish db connection
        summaries = db.summaries  # Create a collection

        # Insert a record(s)
        if isinstance(data, dict):
            result = summaries.insert_one(data)
            logger.info(
                f"Record {result.inserted_id} successfully added to {summaries.name} collection in {db.name} db"
            )
        elif isinstance(data, list):
            result = summaries.insert_many(data)
            logger.info(
                f"Record {result.inserted_ids} successfully added to {summaries.name} collection in {db.name} db"
            )
        else:
            raise ValueError(f"Cannot save type: {type(data)}")


def main(LIMIT_TRANSCRIPT: int | float | None, video_id: str):
    load_dotenv()

    model = init_model(configs.prompt_template)
    msgs = []

    ModelParams = configs.ModelParams

    is_summarised, data = check_if_summarised(video_id)

    if is_summarised:
        logger.info(f"{video_id=}' has already been summarised")
        msgs.append(data)

    else:

        video_url = f"https://www.youtube.com/watch?v={video_id}"

        data = {
            "video_id": video_id,
            "video_url": video_url,
            "video_title": get_video_title(video_url),
            "params": dict(ModelParams.load()),
        }

        missing_keys = []
        for k in configs.video_keys:
            if k not in data and k != "summary":
                missing_keys.append(k)

        if missing_keys:
            raise ValueError(f"Some keys are not included: {missing_keys=}")

        logger.info(f"Summarising {video_id=} ...")
        result = get_transcript_from_db(video_id=video_id)
        transcript = result.get("transcript")

        # Chunk the entire transcript into list of lines
        transcripts = chunk_a_list(transcript, ModelParams.load().CHUNK_SIZE)

        if (LIMIT_TRANSCRIPT is not None) & (LIMIT_TRANSCRIPT > 1):
            transcripts = transcripts[:LIMIT_TRANSCRIPT]

        elif LIMIT_TRANSCRIPT <= 1:
            length = len(transcripts) * LIMIT_TRANSCRIPT
            transcripts = transcripts[: int(length)]

        else:
            raise ValueError("incorrect value for LIMIT_TRANSCRIPT")

        # Summary of summaries: recursively chunk the list & summarise until len(summaries) == 1
        # Summarize each transcript
        list_of_summaries = summarize_list_of_transcripts(
            transcripts,
            ModelParams.load().BULLETS,
            model,
            ModelParams.load().SUMMARY_LIMIT,
        )

        # Combine summaries in chunks and summarize them iteratively until a single summary is obtained
        while len(list_of_summaries) > 1:
            list_of_summaries = [
                summarize_list_of_summaries(
                    list_of_summaries,
                    ModelParams.load().CHUNK_SIZE,
                    ModelParams.load().BULLETS,
                    model,
                    ModelParams.load().SUMMARY_LIMIT,
                )
            ]

        # Output the final summary
        assert len(list_of_summaries) == 1, "Not a summary of summaries"
        msg = list_of_summaries[0]

        data.update({"summary": msg})

        save_summary(data)

        res = {k: v for k, v in data.items() if k in configs.video_keys}
        msgs.append(res)

    return msgs


if __name__ == "__main__":

    check_if_summarised(video_id="TRjq7t2Ms5I")
    check_if_summarised(video_id="JEBDfGqrAUA")
    exit()

    video_1 = {
        "video_id": "TRjq7t2Ms5I",
        "video_url": "https://www.youtube.com/watch?v=TRjq7t2Ms5I",
        "video_title": "Building Production-Ready RAG Applications: Jerry Liu",
        "summary": "Jerry's presentation addresses the challenges and advancements in implementing language model reasoning in real-world applications, focusing on enhancing data retrieval and integration for question answering and conversational agents.\n\n- Jerry introduces the topic and announces a raffle (0:00:14 - 0:00:26).\n- He discusses innovative uses of generative models (0:00:31 - 0:00:41).\n- The significance of retrieval augmentation and context integration is outlined (0:00:53 - 0:01:11).\n- Fine-tuning neural networks for knowledge embedding is introduced (0:01:13 - 0:01:22).\n- Challenges in productionizing applications are identified, especially regarding response quality and vector database retrieval (0:02:54 - 0:03:12).",
        "params": {
            "MODEL": "gpt-4-1106-preview",
            "CHUNK_SIZE": 10,
            "SUMMARY_LIMIT": 150,
            "BULLETS": 5,
            "BATCH_CHUNKS": 2,
        },
    }

    video_2 = {
        "video_id": "JEBDfGqrAUA",
        "video_url": "https://www.youtube.com/watch?v=JEBDfGqrAUA",
        "video_title": "Vector Search RAG Tutorial – Combine Your Data with LLMs with Advanced Search",
        "summary": "The video outlines a course on leveraging vector search and embeddings with large language models like GPT-4 for practical projects such as semantic search and question-answering applications.\n\n- Introduction to a course on vector search and embeddings with large language models like GPT-4. (0:00:00)\n- The first project focuses on creating a semantic search feature for movie queries. (0:00:14)\n- Discussion on building a question-answering app using Vector search and the RAG architecture. (0:00:25)\n- Usage of Python and JavaScript for semantic similarity searches in MongoDB's Atlas Vector search. (0:00:51)\n- Explanation of vector embeddings and their role in organizing digital data by similarity. (0:01:19)",
    }

    data = [video_1, video_2]
    save_summary(data)
