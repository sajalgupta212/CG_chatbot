# agents/mapping_extractor.py

import os
import textwrap
from dotenv import load_dotenv
from utils.snowflake_connection import get_connection
import pandas as pd

load_dotenv()


class MappingExtractorAgent:
    def __init__(self):
        """
        Initialize the agent and create a Snowflake connection using JWT authentication.
        """
        try:
            self.conn = get_connection()
            print("✅ Snowflake connection initialized")
        except Exception as e:
            print(f"⚠️ Warning: Snowflake connection not initialized: {e}")
            self.conn = None

    # ------------------------------
    # FETCH PROCEDURE DDL
    # ------------------------------
    def fetch_procedure_text(self, db, schema, proc_name):
        if not self.conn:
            raise ValueError("Snowflake connection not initialized.")

        query = f"""
            SELECT DDL_TEXT
            FROM "{db}"."{schema}"."DDL_METADATA"
            WHERE UPPER(OBJECT_NAME) = UPPER('{proc_name}')
            LIMIT 1;
        """

        print("🧠 Running query:")
        print(textwrap.indent(query, "    "))

        cur = self.conn.cursor()
        cur.execute(query)
        row = cur.fetchone()
        cur.close()

        return row[0] if row else None

    # ------------------------------
    # CHUNKING METHOD (REQUIRED)
    # ------------------------------
    def chunk_sql_text(self, ddl_text, max_len=500):
        lines = ddl_text.split("\n")
        chunks = []
        buffer = []

        for line in lines:
            buffer.append(line)
            if sum(len(l) for l in buffer) > max_len:
                chunks.append({"text": "\n".join(buffer)})
                buffer = []

        if buffer:
            chunks.append({"text": "\n".join(buffer)})

        return chunks

    # ------------------------------
    # FETCH QUERY HISTORY FOR CRUD MATRIX
    # ------------------------------
    def fetch_query_history(self, days=30):
        """
        Returns query history DataFrame for CRUD matrix.
        Currently a stub with sample data; replace with real Snowflake query later.
        """
        # Example columns: object_name, crud, user_name, timestamp
        data = [
            {"object_name": "SALES_TABLE", "crud": "READ", "user_name": "alice"},
            {"object_name": "SALES_TABLE", "crud": "UPDATE", "user_name": "bob"},
            {"object_name": "CUSTOMER_VIEW", "crud": "READ", "user_name": "alice"},
            {"object_name": "INVENTORY_TABLE", "crud": "INSERT", "user_name": "charlie"},
            {"object_name": "ORDERS_TABLE", "crud": "DELETE", "user_name": "bob"},
        ]
        df = pd.DataFrame(data)
        return df


# ------------------------------
# QUICK TEST
# ------------------------------
if __name__ == "__main__":
    agent = MappingExtractorAgent()
    if agent.conn:
        try:
            cur = agent.conn.cursor()
            cur.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
            db, schema = cur.fetchone()
            print(f"✅ Snowflake connected successfully! Current database: {db}, schema: {schema}")
            cur.close()
            agent.conn.close()
        except Exception as e:
            print(f"❌ Snowflake connection test failed: {e}")
    else:
        print("❌ Snowflake connection not initialized.")
