import re
from datetime import datetime
import pinecone
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd

device = 'cude' if torch.cuda.is_available() else 'cpu'

model  = SentenceTransformer('all-MiniLM-L6-v2', device=device)


def extract_hashtags(content):
    if isinstance(content, str):
        return re.findall(r'#\w+', content)
    return []

def extract_phases_of_day(publish_date):
    dt_object = datetime.strptime(publish_date, "%m/%d/%Y %H:%M")
    hour = dt_object.hour

    if 6<= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 23:
        return "Evening"
    else:
        return "Night"


pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

# Define index parameters
index_name = "tweet-embeddings"
index = pinecone.Index(index_name)


def upsert_records(df):
    records = [
        {
            "id": str(row["tweet_id"]),
            "values": row["embedding"],
            "metadata": {
                "region": row["region"],
                "hashtags": row["hashtags"],
                "language": row["language"],
                "publish_date": row["publish_date"],
                "phases_of_day": row["Phases of day"],
            },
        }
        for _, row in df.iterrows()
    ]

    # Insert data into Pinecone
    index.upsert(records)


chunk_size = 1000
columns = ["tweet_id", "content", "language", "region", "updates", "publish_date"]
files = ["./russian-troll-tweets/IRAhandle_tweets_1.csv"]
for file in files:
    for df in pd.read_csv(file, chunksize=chunk_size):
        df = df[columns]
        df = df[(df["language"] == "English") & (~(df["region"] == "Unknown") | df["region"].isnull())]
        df["embeddings"] = df["content"].apply(model.encode)
        df["hashtags"] = df["content"].apply(extract_hashtags)
        df["phase_of_date"] = df["publish_date"].apply(extract_phases_of_day)
        upsert_records(df)
