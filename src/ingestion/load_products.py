"""Loads data/csv/products.csv into a SQLite database at data/database/products.db."""
import sqlite3
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "csv" / "products.csv"
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "database" / "products.db"


def build_products_db(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> int:
    df = pd.read_csv(csv_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("products", conn, if_exists="replace", index=False)
    return len(df)


if __name__ == "__main__":
    n = build_products_db()
    print(f"Loaded {n} products into {DB_PATH}")
