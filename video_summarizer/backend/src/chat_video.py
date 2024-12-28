"""Module for chatting with a video via a RAG"""

import json
import os
import time
from uuid import uuid4

import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone, PodSpec
from pinecone.data.index import Index
from sqlalchemy import create_engine, text
from tqdm.auto import tqdm

from video_summarizer.backend.configs.config import Provider, augmented_prompt
from video_summarizer.backend.src.summarize_video import init_model
from video_summarizer.backend.utils.utils import get_mongodb_client, logger


def get_document(video_id: str, collection_name: str = "transcripts"):
    """Get a document related to a video from Mongodb"""

    client, db_name = get_mongodb_client()
    db = client[db_name]
    collection = db[collection_name]
    document = collection.find_one({"video_id": video_id})
    return document

def get_embeddings(data: pd.DataFrame):
    """Peform embeddings on a document

    Args
    ---
    - data: data to peform embedding on

    Returns
    ---
    - ids: uuid of the mebdding
    - embeds: the embedding result
    - metadata: metadata of the embedding
    """
            
    texts = [str(x["text"]) for _, x in data.iterrows()]
    ids = [uuid4().hex for _ in range(len(texts))]

    embeds = OpenAIEmbeddings().embed_documents(texts)

    metadata = [
        {"text": x["text"], "timestamp": str(x["timestamp"])}
        for _, x in data.iterrows()
    ]
    
    return ids, embeds, metadata

class PineconeRAG:
    """Class for using Pinecone as the vector store"""
    
    def __init__(self, api_key, environment):
        self.api_key = api_key
        self.environment = environment
        self.pc = Pinecone(api_key=self.api_key, environment=self.environment)

    def get_create_pinecone_index(self, index_name: str):
        """Retrieve an index from Pinecode vectorstore"""

        pc = self.pc
        available_idx = [i.get("name") for i in pc.list_indexes().get("indexes")]

        if index_name not in available_idx:
            pc.create_index(
                name=index_name,
                dimension=1536,  # https://www.pinecone.io/learn/openai-embeddings-v3/
                metric="cosine",
                spec=PodSpec(environment="gcp-starter"),
            )

            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        # get the created index
        index = pc.Index(index_name)
        logger.info(f"Pinecone index stats:\n {index.describe_index_stats()}")

        return index

    def upsert_documents_to_pinecone(
        self, idx: Index, video_id: str, index_name: str
    ):
        """Inserts document to an index associated with a video id"""

        # check that the index for this video is not empty
        stats = idx.describe_index_stats()
        total_vector_count = stats.get("total_vector_count")

        if total_vector_count > 0:
            logger.info(f"{index_name=} is already populated")
            return None

        # convert transcript to dataframe
        doc = get_document(video_id)
        transcript = doc.get("transcript")

        df = pd.DataFrame(transcript)
        if df.isnull().values.any():
            logger.error("df contains null values")
            return None
        
        data = df[0].str.extract(r"\n(\d+:\d{2}:\d{2})\s-\s(.*)")
        data.columns = ["timestamp", "text"]

        # upload transcript to pinecone index
        batch_size = 100
        for i in tqdm(range(0, len(data), batch_size)):
            i_end = min(len(data), i + batch_size)
            batch = data.iloc[i:i_end]
        
            ids, embeds, metadata = get_embeddings(batch)
            idx.upsert(vectors=zip(ids, embeds, metadata))

        logger.info(f"Successfully added content to {index_name=}")


    def query_vectorstore(
        self,
        query: str,
        embeddings: OpenAIEmbeddings,
        index: Index,
        k: int = 5,
        include_timestamp: bool = False,
    ) -> str:

        vector = embeddings.embed_query(query)

        query_res = index.query(
            vector=vector, top_k=k, include_metadata=True, include_values=False
        )

        logger.info(f"{query=}")
        logger.info(f"{query_res=}")

        context = [
            f'{d["metadata"]["text"]} - {d["metadata"]["timestamp"]}'
            for d in query_res.get("matches")
        ]

        if include_timestamp is False:
            context = [
                f'{d["metadata"]["text"]}' for d in query_res.get("matches")
            ]

        return "\n".join(context)


    def get_context(
        self,
        query: str,
        video_id: str,
        delete_index=False,
        embeddings=OpenAIEmbeddings(),
        k=15,
    ):
        """Given a video id and a query, retrieves the vectors that match the query.

        Args:
        ---
        query: A user provided query or question
        video_id: The video id to query from
        delete_index: Whether to replace the current index with that of a new video id
        embeddings: The vector embeddings to use
        k: The number of lines of a transcript to use. The higher the number, the richer the context

        Returns:
        ---
        Lines from the transctipt that closest match the query
        """

        index_name = video_id.lower()

        # delete the index
        if delete_index:
            try:
                pc = self.pc
                pc.delete_index(index_name)
                logger.info(f"Successfully deleted {index_name=}")
            except:
                logger.info(f"{index_name=} already deleted")

        # create an index
        index = self.get_create_pinecone_index(index_name=index_name)

        # insert content to vectorstore
        self.upsert_documents_to_pinecone(
            idx=index,
            video_id=video_id,
            index_name=index_name,
            embeddings=embeddings,
        )

        # use RAG ("text" is from the metadata)
        # vectorstore = PineconeVectorStore(
        #     index=index, embedding=embeddings, text_key="text"
        # )
        # query_res = vectorstore.similarity_search({"query": query, "k": 3})
        # query_res = vectorstore.similarity_search(query=query)

        return self.query_vectorstore(query, embeddings=embeddings, index=index, k=k)

class PgVectorRAG:
    """Class for using pgvector as the vector store"""
    
    def __init__(self, host, port, username, password, database):
        self.uri = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        self.engine = create_engine(self.uri)
    
    def upsert_document(self, video_id):
        doc = get_document(video_id)
        transcript = doc.get("transcript")

        df = pd.DataFrame(transcript)
        if df.isnull().values.any():
            logger.error("df contains null values")
            return None
        
        data = df[0].str.extract(r"\n(\d+:\d{2}:\d{2})\s-\s(.*)")
        data.columns = ["timestamp", "text"]

        batch_size = 100
        engine = self.engine
        
        for i in tqdm(range(0, len(data), batch_size)):
            i_end = min(len(data), i + batch_size)
            batch = data.iloc[i:i_end]
            
            ids, embeds, metadata = get_embeddings(batch)
        
            query1 = f"""
            INSERT INTO items (id, embedding, metadata) VALUES 
            (1, '[1,2,3]', '{{"text": "search on the data using an approximate", "timestamp": "0:04:13"}}'), 
            (2, '[4,5,6]', '{{"text": "search on the data using an approximate", "timestamp": "0:04:13"}}')
            ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding;
            """
            
            meta = {"text": "search on the data using an approximate", "timestamp": "0:04:13"}
            v1 = [1,2,3]
            v2 = [3,4,5]
            
            query2 = f"""
            INSERT INTO items (id, embedding, metadata) VALUES 
            (1, '{str(v1)}', '{json.dumps(meta)}'), 
            (2, '{str(v2)}', '{json.dumps(meta)}')
            ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding;
            """
            
            query3 = f"""
            INSERT INTO items (id, embedding, metadata) VALUES 
            {", ".join(f"({id}, '{str(embed)}', '{json.dumps(meta)}')" 
                    for id, embed, meta in zip(ids, embeds, metadata))}
            ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding;
            """
            
            print("embeddings vector len:", len(embeds[0]))
            
            with engine.begin() as conn:
                conn.execute(text(query3))
                
            exit()

    # query pgvector
    
def main(query: str, video_id: str, model: Provider, vectorstore: Provider, delete_index: bool = False):
    from dotenv import load_dotenv
    load_dotenv()

    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
    PG_HOST = os.environ.get("_PG_HOST")
    PG_PORT = os.environ.get("_PG_PORT")
    PG_USERNAME = os.environ.get("_PG_USERNAME")
    PG_PASSWORD = os.environ.get("_PG_PASSWORD")
    PG_DATABASE = os.environ.get("_MONGO_DB")
    
    if vectorstore == Provider.pinecone.name:
        pinecone_rag = PineconeRAG(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        context = pinecone_rag.get_context(query, video_id=video_id, delete_index=delete_index)
        logger.info(f"{context=}")

    if vectorstore == Provider.pgvector.name:
        pgvector_rag = PgVectorRAG(
            host=PG_HOST, 
            port=PG_PORT, 
            username=PG_USERNAME, 
            password=PG_PASSWORD, 
            database=PG_DATABASE,
            )
        
        pgvector_rag.upsert_document(video_id)
    
    if model != Provider.ollama.name:
        model = init_model(template=augmented_prompt)
        logger.info("Connecting to ChatGPT...")
        res = model.predict(question=query, context=context)

    logger.info(res)

    return res


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # VIDEO_ID = "JEBDfGqrAUA"
    parser.add_argument(
        "--video_id", help="The video id to chat with", required=True
    )
    
    parser.add_argument(
        "--delete_index",
        help="Delete the Pinecone index",
        action="store_true",
        default=False,
    )
    
    parser.add_argument(
        "--model",
        help="Model provider (openai, anthropic, ollama)",
        default=Provider.ollama.name
    )
    
    parser.add_argument(
        "--vectorstore",
        help="The vector store. One of [pinecone, pgvector]",
        default=Provider.pgvector.name
    )

    args = parser.parse_args()

    logger.info(args)

    QUERY = "What is a vector store?"
    res = main(QUERY, video_id=args.video_id, vectorstore=args.vectorstore, model=args.model, delete_index=args.delete_index)
    print(res)
