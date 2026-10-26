from pathlib import Path
import duckdb

PROJECT = Path(__file__).resolve().parents[2]
BRONZE = PROJECT / "data" / "bronze" / "retail_bronze.duckdb"
SILVER = PROJECT / "data" / "silver" / "retail_silver.duckdb"


def main():
    SILVER.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(SILVER))

    try:
        con.execute("SET preserve_insertion_order = false")
        con.execute(f"ATTACH '{BRONZE.as_posix()}' AS bronze (READ_ONLY)")

        print("Building silver_sales (this may take a few minutes)...")
        con.execute("""
            CREATE OR REPLACE TABLE silver_sales AS
            SELECT
                s.id,
                s.item_id,
                s.dept_id,
                s.cat_id,
                s.store_id,
                s.state_id,
                s.d,
                c.date,
                c.wm_yr_wk,
                CAST(s.units AS INTEGER) AS units,
                p.sell_price
            FROM (
                UNPIVOT bronze.bronze_sales_evaluation
                ON COLUMNS('^d_[0-9]+$')
                INTO NAME d VALUE units
            ) s
            JOIN bronze.bronze_calendar c
              ON c.d = s.d
            LEFT JOIN bronze.bronze_sell_prices p
              ON p.store_id = s.store_id
             AND p.item_id  = s.item_id
             AND p.wm_yr_wk = c.wm_yr_wk
        """)

        rows = con.execute("SELECT COUNT(*) FROM silver_sales").fetchone()[0]
        print(f"  silver_sales rows: {rows:,}")
        print(f"Database: {SILVER}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
