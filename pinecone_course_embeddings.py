"""
Author: Selim Ben Haj Braiek
Project: Semantic Search Engine with Pinecone and Sentence Transformers
Description: This script demonstrates how to create a semantic search engine using Pinecone and Sentence Transformers.
  
"""

import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec


# STEP 1 — Load environment variables

load_dotenv(find_dotenv(), override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in .env file!")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)



# STEP 2 — Load and preprocess data

DATA_PATH = "course_descriptions.csv"
print(f"Loading dataset from: {DATA_PATH}")
files = pd.read_csv(DATA_PATH, encoding="ANSI")

# Create a detailed course description string
def create_course_description(row):
    return (
        f"The course name is {row['course_name']}, "
        f"the slug is {row['course_slug']}, "
        f"the technology is {row['course_technology']}, "
        f"and the course topic is {row['course_topic']}."
    )

files["course_description_new"] = files.apply(create_course_description, axis=1)
print(f" Created {len(files)} new course descriptions.")


# STEP 3 — Setup Pinecone index
INDEX_NAME = "my-index"
DIMENSION = 384
METRIC = "cosine"

# Delete existing index if exists
existing_indexes = [index.name for index in pc.list_indexes()]
if INDEX_NAME in existing_indexes:
    pc.delete_index(INDEX_NAME)
    print(f"Existing index '{INDEX_NAME}' deleted.")
else:
    print(f"Index '{INDEX_NAME}' not found, creating new one.")

# Create new index
pc.create_index(
    name=INDEX_NAME,
    dimension=DIMENSION,
    metric=METRIC,
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

index = pc.Index(INDEX_NAME)
print(f" Pinecone index '{INDEX_NAME}' ready.")


# STEP 4 — Generate embeddings

print("Loading sentence transformer model...")
model = SentenceTransformer("multi-qa-distilbert-cos-v1")

def create_embeddings(row):
    combined_text = " ".join(
        str(row[field])
        for field in ["course_description", "course_description_new", "course_description_short"]
        if field in row and pd.notna(row[field])
    )
    return model.encode(combined_text, show_progress_bar=False)

files["embedding"] = files.apply(create_embeddings, axis=1)
print("Embeddings generated successfully.")


# STEP 5 — Upload (Upsert) to Pinecone

print(" Uploading vectors to Pinecone...")
vectors_to_upsert = [
    (str(row["course_name"]), row["embedding"].tolist())
    for _, row in files.iterrows()
]

BATCH_SIZE = 100
for i in range(0, len(vectors_to_upsert), BATCH_SIZE):
    batch = vectors_to_upsert[i : i + BATCH_SIZE]
    index.upsert(vectors=batch)
print(" All data upserted to Pinecone index!")


# STEP 6 — Run a test query
query = "clustering"
query_embedding = model.encode(query, show_progress_bar=False).tolist()

query_results = index.query(
    vector=[query_embedding],
    top_k=12,
    include_values=True,
)

SCORE_THRESHOLD = 0.3
print(f"\n Query results for: '{query}' (score ≥ {SCORE_THRESHOLD})\n")

for match in query_results["matches"]:
    if match["score"] >= SCORE_THRESHOLD:
        print(f"📘 Matched item ID: {match['id']}, Score: {match['score']:.3f}")

print("\n Query completed successfully.")
