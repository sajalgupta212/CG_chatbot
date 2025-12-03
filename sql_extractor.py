import re
import sqlglot
from sqlglot.expressions import Table, Column, Identifier
from sqlglot.errors import ParseError


class SQLExtractor:
    def __init__(self):
        pass

    # -------------------------------------------------
    # CLEAN SNOWFLAKE PROCEDURE TEXT (supports JS/SP)
    # -------------------------------------------------
    def extract_sql_blocks(self, text):
        """
        Extract all SQL statements even if procedure is JavaScript.
        Covers:
        - EXECUTE IMMEDIATE
        - $$ ... $$
        - BEGIN ... END
        - Semi-colon SQL statements
        """

        sql_blocks = []

        # 1. Extract SQL within EXECUTE IMMEDIATE '...'
        exec_sql = re.findall(r"EXECUTE\s+IMMEDIATE\s+'(.*?)'", text, flags=re.S | re.I)
        sql_blocks.extend(exec_sql)

        # 2. Extract SQL inside $$ ... $$
        dollar_sql = re.findall(r"\$\$(.*?)\$\$", text, flags=re.S)
        sql_blocks.extend(dollar_sql)

        # 3. Extract simple SQL commands (fallback)
        basic_sql = re.findall(r"(SELECT.*?;|INSERT.*?;|UPDATE.*?;|DELETE.*?;|MERGE.*?;)",
                               text, flags=re.S | re.I)
        sql_blocks.extend(basic_sql)

        # 4. If nothing found, assume the text itself is SQL
        if not sql_blocks:
            sql_blocks = [text]

        return [block.strip() for block in sql_blocks]

    # -------------------------------------------------
    # PARSE SQL AND EXTRACT TABLES + COLUMNS
    # -------------------------------------------------
    def parse_sql(self, sql_text):
        try:
            parsed = sqlglot.parse_one(sql_text, read="snowflake")
        except ParseError:
            return {
                "tables": [],
                "columns_read": [],
                "columns_written": [],
                "statements": []
            }

        tables = []
        columns_read = []
        columns_written = []

        # Extract tables
        for table in parsed.find_all(Table):
            tables.append(table.sql())

        # Extract columns
        for col in parsed.find_all(Column):
            columns_read.append(col.sql())

        # Detect writes (INSERT, UPDATE, MERGE)
        if parsed.find("Insert"):
            for col in parsed.find_all(Column):
                columns_written.append(col.sql())

        if parsed.find("Update"):
            for col in parsed.find_all(Identifier):
                columns_written.append(col.sql())

        if parsed.find("Merge"):
            for col in parsed.find_all(Identifier):
                columns_written.append(col.sql())

        return {
            "tables": list(set(tables)),
            "columns_read": list(set(columns_read)),
            "columns_written": list(set(columns_written)),
            "statements": [parsed.sql()]
        }

    # -------------------------------------------------
    # MAIN FUNCTION: FULL EXTRACTION
    # -------------------------------------------------
    def extract_all(self, raw_text):
        sql_blocks = self.extract_sql_blocks(raw_text)

        all_tables = set()
        all_read = set()
        all_written = set()
        statements = []

        for block in sql_blocks:
            parsed = self.parse_sql(block)

            all_tables.update(parsed["tables"])
            all_read.update(parsed["columns_read"])
            all_written.update(parsed["columns_written"])
            statements.extend(parsed["statements"])

        return {
            "tables": list(all_tables),
            "columns_read": list(all_read),
            "columns_written": list(all_written),
            "statements": statements
        }
