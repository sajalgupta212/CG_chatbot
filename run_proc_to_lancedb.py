import os
from agents.mapping_extractor import MappingExtractorAgent
from embed_and_store import load_model, embed_texts, store_embeddings_lancedb

from dotenv import load_dotenv
load_dotenv()


def main():
    print("🔍 Enter the stored procedure name: ", end="")
    proc_name = input().strip()

    if not proc_name:
        print("❗ Procedure name is required.")
        return

    # Load embedding model
    load_model()

    # Initialize agent
    agent = MappingExtractorAgent()
    ddl = agent.fetch_procedure_text(
        os.getenv("SNOWFLAKE_DATABASE"),
        os.getenv("SNOWFLAKE_SCHEMA"),
        proc_name
    )

    if not ddl:
        print("❌ No DDL found.")
        return

    print("📦 Chunking...")
    chunks = agent.chunk_sql_text(ddl)
    print(f"✅ Generated {len(chunks)} chunks.")

    texts = [c["text"] for c in chunks]

    print("🧮 Generating embeddings...")
    vectors = embed_texts(texts)

    print("📁 Storing in LanceDB...")
    store_embeddings_lancedb(
        db_path="lancedb_db",
        table_name="sp_blocks_vectors",
        chunks=chunks,
        vectors=vectors
    )

    print("🎉 DONE!")


if __name__ == "__main__":
    main()
