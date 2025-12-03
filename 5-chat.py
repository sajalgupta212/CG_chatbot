# 5-chat.py
import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import lancedb
import numpy as np
import argparse

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = "lancedb_db"
TABLE_NAME = "sp_blocks_vectors"

# Must match what you stored earlier
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"   # recommended Groq model

def retrieve_context(query, top_k=5):
    """Vector search over LanceDB."""
    embedder = SentenceTransformer(EMBED_MODEL)
    qvec = embedder.encode([query])[0].astype(np.float32).tolist()

    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE_NAME)

    # Important: specify correct vector column
    results = tbl.search(qvec, vector_column="embedding").limit(top_k).to_pandas()

    if results.empty:
        return ""

    # Use `text` column, not block_text
    context = "\n\n---\n\n".join(results["text"].tolist())
    return context


def call_groq(prompt):
    client = Groq(api_key=GROQ_KEY)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert SQL lineage and ETL logic assistant. "
                    "Always answer strictly based on the provided context. "
                    "If the answer is not in context, say so."
                )
            },
            {"role": "user", "content": prompt}
        ],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    query = args.query
    print("USER QUERY →", query)

    context = retrieve_context(query)

    if not context.strip():
        print("❌ No relevant context found in LanceDB.")
        exit()

    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}"

    answer = call_groq(prompt)
    print("\n========================\nFINAL ANSWER:\n")
    print(answer)
