# 4-search.py
import argparse
from sentence_transformers import SentenceTransformer
import lancedb
import numpy as np

DB_PATH = "lancedb_db"
TABLE_NAME = "sp_blocks_vectors"
EMBED_MODEL = "all-MiniLM-L6-v2"

def search(query, top_k=5):
    embedder = SentenceTransformer(EMBED_MODEL)
    qvec = embedder.encode([query])[0].astype(np.float32).tolist()
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE_NAME)
    results = tbl.search(qvec).limit(top_k).to_pandas()
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()
    print("USER QUERY →", args.query)
    results = search(args.query, top_k=5)
    print(f"✅ Found {len(results)} results\n")
    for i, r in results.iterrows():
        score = r.get("score") or r.get("_distance") or r.get("_dist") or r.get("vector_score") or 0
        print(f"[{i+1}] Score={float(score):.4f}\n{r['block_text'][:600]}\n")
