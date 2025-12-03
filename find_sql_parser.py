import os
import re

PATTERNS = [
    r"class\s+SQLExtractor",
    r"def\s+extract_sql",
    r"sqlglot",
    r"parse\(",
    r"lineage",
    r"CRUD",
    r"table_name",
]

ignore_dirs = {"venv", ".git", "__pycache__"}

def search(root="."):
    print("🔍 Searching for SQL parsing logic...\n")

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]

        for file in filenames:
            if not file.endswith(".py"):
                continue

            path = os.path.join(dirpath, file)
            try:
                content = open(path, "r", errors="ignore").read()
            except:
                continue

            for pattern in PATTERNS:
                if re.search(pattern, content):
                    print(f"📌 Found match in: {path}  (pattern: {pattern})")
                    break


if __name__ == "__main__":
    search()
