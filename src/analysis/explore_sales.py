from pathlib import Path
import duckdb

PROJECT = Path(__file__).resolve().parents[2]
SILVER = PROJECT / "data" / "silver" / "retail_silver.duckdb"
BRONZE = PROJECT / "data" / "bronze" / "retail_bronze.duckdb"

queries = {
    "1. Sales and revenue by category": """
        SELECT cat_id,
               SUM(units) AS units,
               ROUND(100.0 * SUM(units) / SUM(SUM(units)) OVER (), 1) AS pct_units,
               ROUND(SUM(units * sell_price), 0) AS revenue,
               ROUND(100.0 * SUM(units * sell_price)
                     / SUM(SUM(units * sell_price)) OVER (), 1) AS pct_revenue
        FROM silver_sales
        GROUP BY cat_id
        ORDER BY units DESC
    """,
    "2. Sales by state and store": """
        SELECT state_id, store_id, SUM(units) AS units
        FROM silver_sales
        GROUP BY state_id, store_id
        ORDER BY state_id, units DESC
    """,
    "3. Sales by year (2011 and 2016 are partial years)": """
        SELECT YEAR(date) AS year,
               COUNT(DISTINCT date) AS days,
               SUM(units) AS units,
               ROUND(SUM(units) / COUNT(DISTINCT date), 0) AS units_per_day
        FROM silver_sales
        GROUP BY year
        ORDER BY year
    """,
    "4. Average daily sales by day of week": """
        SELECT dayname(date) AS weekday,
               ROUND(SUM(units) / COUNT(DISTINCT date), 0) AS avg_units_per_day
        FROM silver_sales
        GROUP BY weekday, isodow(date)
        ORDER BY isodow(date)
    """,
    "5. The five lowest-sales days": """
        SELECT date, dayname(date) AS weekday, SUM(units) AS units
        FROM silver_sales
        GROUP BY date
        ORDER BY units
        LIMIT 5
    """,
    "6. Investigating very low prices (under $0.10)": """
        SELECT item_id, store_id,
               MIN(sell_price) AS min_price,
               MAX(sell_price) AS max_price,
               COUNT(*) AS weeks_listed
        FROM bronze.bronze_sell_prices
        WHERE item_id IN (
            SELECT DISTINCT item_id
            FROM bronze.bronze_sell_prices
            WHERE sell_price < 0.10
        )
        GROUP BY item_id, store_id
        ORDER BY min_price
        LIMIT 10
    """,
}


def main():
    con = duckdb.connect(str(SILVER), read_only=True)
    try:
        con.execute(f"ATTACH '{BRONZE.as_posix()}' AS bronze (READ_ONLY)")
        for title, query in queries.items():
            print(f"\n=== {title} ===")
            con.sql(query).show(max_rows=20)
    finally:
        con.close()


if __name__ == "__main__":
    main()
