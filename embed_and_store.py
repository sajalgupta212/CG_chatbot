import os
import lancedb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load the local embedding model once
_model = None


def load_model(model_path="models/all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        print(f"📦 Loading local embedding model from: {model_path}")
        _model = SentenceTransformer(model_path)
    return _model


def embed_texts(texts):
    """
    Generate embeddings using the local SentenceTransformer model.
    """
    model = load_model()
    print(f"🧠 Generating {len(texts)} embeddings...")
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return vectors


def store_embeddings_lancedb(db_path, table_name, chunks, vectors):
    import lancedb

    db = lancedb.connect(db_path)

    # Build valid row-wise data for LanceDB
    data = []
    for i, (text, embedding) in enumerate(zip(chunks, vectors)):
        data.append({
            "chunk_id": i,
            "text": text,
            "embedding": embedding
        })

    if table_name in db.table_names():
        print("🟡 Table exists — appending...")
        tbl = db.open_table(table_name)
        tbl.add(data)
    else:
        print("🟢 Creating new table...")
        db.create_table(table_name, data)

    print("✅ Stored successfully in LanceDB")
