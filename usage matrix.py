import pandas as pd


def build_crud_matrix(tables, read_cols, write_cols):
    """
    Returns a CRUD matrix:
    - C: INSERT / CTAS
    - R: SELECT
    - U: UPDATE
    - D: DELETE
    """

    crud = {}

    for table in tables:
        crud[table] = {
            "C": "INSERT INTO" in table or "CREATE TABLE" in table,
            "R": any(table in col for col in read_cols),
            "U": any(table in col for col in write_cols),
            "D": False  # DELETE detection can be added
        }

    df = pd.DataFrame(crud).T
    df.index.name = "Table"
    return df


def build_usage_matrix(tables):
    """
    Like a heatmap / occurrence matrix.
    """
    return pd.DataFrame({
        "Table": tables,
        "Usage Count": [1] * len(tables)
    })
