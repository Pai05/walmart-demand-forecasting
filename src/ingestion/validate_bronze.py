from pathlib import Path
import duckdb

PROJECT = Path(__file__).resolve().parents[2]
DB = PROJECT / "data" / "bronze" / "retail_bronze.duckdb"

checks = {
    "Calendar keys": """
        SELECT COUNT(*) AS total_rows,
               COUNT(DISTINCT d) AS unique_days,
               COUNT(DISTINCT date) AS unique_dates,
               COUNT(*) FILTER (WHERE d IS NULL OR date IS NULL) AS null_keys
        FROM bronze_calendar
    """,
    "Validation sales IDs": """
        SELECT COUNT(*) AS total_rows,
               COUNT(DISTINCT id) AS unique_ids,
               COUNT(*) FILTER (WHERE id IS NULL OR item_id IS NULL
                                  OR store_id IS NULL) AS null_keys
        FROM bronze_sales_validation
    """,
    "Evaluation sales IDs": """
        SELECT COUNT(*) AS total_rows,
               COUNT(DISTINCT id) AS unique_ids,
               COUNT(*) FILTER (WHERE id IS NULL OR item_id IS NULL
                                  OR store_id IS NULL) AS null_keys
        FROM bronze_sales_evaluation
    """,
    "Duplicate price keys": """
        SELECT COUNT(*) AS duplicate_key_groups
        FROM (
            SELECT store_id, item_id, wm_yr_wk
            FROM bronze_sell_prices
            GROUP BY store_id, item_id, wm_yr_wk
            HAVING COUNT(*) > 1
        ) dup
    """,
    "Price quality": """
        SELECT COUNT(*) FILTER (WHERE store_id IS NULL OR item_id IS NULL
                                  OR wm_yr_wk IS NULL) AS null_keys,
               COUNT(*) FILTER (WHERE sell_price IS NULL) AS null_prices,
               COUNT(*) FILTER (WHERE sell_price <= 0) AS non_positive_prices,
               MIN(sell_price) AS min_price,
               MAX(sell_price) AS max_price
        FROM bronze_sell_prices
    """,
    "Submission IDs": """
        SELECT COUNT(*) AS total_rows,
               COUNT(DISTINCT id) AS unique_ids,
               COUNT(*) FILTER (WHERE id IS NULL) AS null_ids
        FROM bronze_sample_submission
    """,
}


def main():
    con = duckdb.connect(str(DB), read_only=True)
    try:
        for name, query in checks.items():
            result = con.execute(query)
            columns = [c[0] for c in result.description]
            values = result.fetchone()
            print(f"\n--- {name} ---")
            for col, val in zip(columns, values):
                print(f"  {col}: {val}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
