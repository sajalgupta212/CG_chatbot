import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "lancedb_db"
TABLE_NAME = "sp_blocks_vectors"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_text(value):
    """Ensure the chunk text is always converted into a readable string."""
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        # Convert dict → "key: value\nkey: value"
        return "\n".join(f"{k}: {v}" for k, v in value.items())

    if isinstance(value, list):
        # Convert list → "item1\nitem2"
        return "\n".join(str(v) for v in value)

    # Fallback for other types
    return str(value)


def main():
    print("📁 Connecting to LanceDB...")
    db = lancedb.connect(DB_PATH)

    if TABLE_NAME not in db.table_names():
        print(f"❌ Table '{TABLE_NAME}' not found!")
        print(f"Available tables: {db.table_names()}")
        return

    table = db.open_table(TABLE_NAME)

    question = input("💬 Enter your question about the SQL logic: ")

    print("📦 Loading embedding model...")
    model = SentenceTransformer(MODEL)
    query_vector = model.encode(question, convert_to_numpy=True).astype(np.float32)

    print("🔍 Running vector search...")

    try:
        results = (
            table.search(query_vector)
            .limit(5)
            .to_list()
        )
    except Exception as e:
        print("❌ Search error:", e)
        print("\n🔍 Dumping a sample problematic row for debugging:")
        sample = table.head(1)[0]
        print(sample)
        return

    print("\n=== Top Results ===")
    for r in results:
        print("\n---")
        print(normalize_text(r.get("text", "")))  # safe access + normalize


if __name__ == "__main__":
    main()
