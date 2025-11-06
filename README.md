# Semantic-Search-Engine-with-Pinecone-and-Sentence-Transformers

Semantic Course Search Engine with Pinecone and Sentence Transformers

This project demonstrates how to build a **semantic search engine** for course descriptions using **Sentence Transformers** for text embeddings and **Pinecone** as a vector database.  

By converting course information into vector embeddings, the system can retrieve courses that are **semantically similar** — even if the query doesn’t share exact words.  

## Features

- Generate semantic embeddings for course descriptions  
- Search and retrieve similar courses using Pinecone  
- Automatically create detailed course descriptions  
- Store and query data in a vector index (serverless Pinecone)  
- Simple Python script — no extra setup required  


## How It Works

1. **Load the Dataset**  
   A CSV file (`course_descriptions.csv`) containing columns like:
   - `course_name`
   - `course_slug`
   - `course_technology`
   - `course_topic`
   - `course_description_short`

2. **Generate Descriptions**  
   For each row, the script combines these fields into a detailed descriptive text.

3. **Create Embeddings**  
   The `SentenceTransformer` model (`multi-qa-distilbert-cos-v1`) encodes each course description into a 384-dimensional vector.

4. **Upload to Pinecone**  
   The vectors are uploaded (upserted) into a **Pinecone serverless index**, allowing fast similarity searches.

5. **Run Semantic Queries**  
   Query any phrase like `"clustering"` or `"deep learning"` to find related courses.


### Clone the Repository

```bash
git clone https://github.com/selimbenhajbraiek/Semantic-Search-Engine-with-Pinecone-and-Sentence-Transformers.git
cd pinecone_course_embeddings
