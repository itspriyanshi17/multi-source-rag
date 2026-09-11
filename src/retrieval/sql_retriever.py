"""Text-to-SQL retrieval over the products table, exposed as an LLM tool function."""
from pathlib import Path

from src.retrieval.sql_safety import run_validated_select

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "database" / "products.db"

SCHEMA_DESCRIPTION = """Table: products
Columns:
  product_id     TEXT    e.g. 'P1001'
  name           TEXT    e.g. 'NovaBook Pro 15'
  category       TEXT    one of: Laptops, Smartphones, Headphones, Monitors, Smartwatches,
                          Tablets, Cameras, Speakers, Accessories, Gaming Consoles
  brand          TEXT    e.g. 'NovaTech', 'Pulsar', 'Corevia', 'Pulse', 'Aurora', 'Zenbyte'
  price_usd      REAL    e.g. 899.4
  stock          INTEGER units currently in stock
  rating         REAL    average customer rating out of 5
  release_year   INTEGER e.g. 2024
  specs_summary  TEXT    short spec string, e.g. '16GB RAM, 512GB SSD, ...'
  description    TEXT    one-line description
"""


def run_sql_query(sql: str) -> str:
    """Runs a read-only SQL SELECT query against the NovaTech products table and returns the
    results. Use this for questions about product prices, stock levels, categories, brands,
    ratings, release years, or specs — anything requiring filtering, sorting, or counting over
    the product catalog.

    Args:
        sql: A single SQLite SELECT statement. Must reference the `products` table. Example:
            "SELECT name, price_usd, stock FROM products WHERE category='Laptops' AND
            price_usd < 1000 AND stock > 0 ORDER BY price_usd ASC"

    Returns:
        A formatted string of the matching rows, or an error message if the query is invalid.
    """
    return run_validated_select(DB_PATH, sql)


if __name__ == "__main__":
    print(SCHEMA_DESCRIPTION)
    print(run_sql_query("SELECT name, price_usd, stock FROM products WHERE category='Laptops' AND price_usd < 1000 ORDER BY price_usd ASC"))
