from pathlib import Path
import duckdb

PROJECT = Path(__file__).resolve().parents[2]
BRONZE = PROJECT / "data" / "bronze" / "retail_bronze.duckdb"
SILVER = PROJECT / "data" / "silver" / "retail_silver.duckdb"


def show(con, title, query):
    result = con.execute(query)
    columns = [c[0] for c in result.description]
    values = result.fetchone()
    print(f"\n--- {title} ---")
    for col, val in zip(columns, values):
        print(f"  {col}: {val}")


def main():
    con = duckdb.connect(str(SILVER), read_only=True)
    try:
        con.execute(f"ATTACH '{BRONZE.as_posix()}' AS bronze (READ_ONLY)")

        show(con, "Structure", """
            SELECT COUNT(*) AS total_rows,
                   COUNT(DISTINCT id) AS unique_series,
                   COUNT(DISTINCT date) AS unique_dates,
                   MIN(date) AS first_date,
                   MAX(date) AS last_date
            FROM silver_sales
        """)

        show(con, "Duplicate series-days", """
            SELECT COUNT(*) AS duplicate_groups
            FROM (
                SELECT id, date
                FROM silver_sales
                GROUP BY id, date
                HAVING COUNT(*) > 1
            ) dup
        """)

        show(con, "Validity", """
            SELECT COUNT(*) FILTER (WHERE date IS NULL) AS null_dates,
                   COUNT(*) FILTER (WHERE units IS NULL) AS null_units,
                   COUNT(*) FILTER (WHERE units < 0) AS negative_units
            FROM silver_sales
        """)

        show(con, "Prices", """
            SELECT COUNT(*) FILTER (WHERE sell_price IS NULL) AS rows_without_price,
                   ROUND(100.0 * COUNT(*) FILTER (WHERE sell_price IS NULL)
                         / COUNT(*), 2) AS pct_without_price,
                   COUNT(*) FILTER (WHERE sell_price IS NULL AND units > 0)
                       AS sales_without_price
            FROM silver_sales
        """)

        silver_total = con.execute(
            "SELECT SUM(units) FROM silver_sales"
        ).fetchone()[0]
        bronze_sums = con.execute(
            "SELECT SUM(COLUMNS('^d_[0-9]+$')) FROM bronze.bronze_sales_evaluation"
        ).fetchone()
        bronze_total = sum(bronze_sums)

        print("\n--- Reconciliation ---")
        print(f"  bronze_total_units: {bronze_total:,}")
        print(f"  silver_total_units: {silver_total:,}")
        print(f"  match: {bronze_total == silver_total}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
