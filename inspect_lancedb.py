import lancedb

DB_PATH = "lancedb_db"

def main():
    print("📁 Connecting to LanceDB…")
    db = lancedb.connect(DB_PATH)

    tables = db.table_names()
    print("\n📋 Tables inside the database:")
    for name in tables:
        print(f" - {name}")

    if not tables:
        print("\n❌ No tables found in the database.")
        return

    # If only one table exists, auto-select it
    if len(tables) == 1:
        table_name = tables[0]
        print(f"\n📌 Auto-selecting the only table: {table_name}")
    else:
        table_name = input("\nEnter table name to preview rows: ").strip()
        if not table_name:
            print("❌ Please enter a valid table name next time.")
            return
        if table_name not in tables:
            print("❌ Table not found!")
            return

    table = db.open_table(table_name)

    print(f"\n🔍 Previewing first 5 rows from '{table_name}':\n")
    for row in table.head(5):
        print(row)
        print("---")

if __name__ == "__main__":
    main()
