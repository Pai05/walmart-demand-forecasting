from datetime import timedelta
from pathlib import Path
import duckdb

PROJECT = Path(__file__).resolve().parents[2]
SILVER = PROJECT / "data" / "silver" / "retail_silver.duckdb"
GOLD = PROJECT / "data" / "gold" / "retail_gold.duckdb"
TEST_DAYS = 28

FORECAST_LONG = """
    (UNPIVOT baseline_forecasts
     ON zero, naive_last_day, seasonal_naive, moving_avg_28
     INTO NAME model VALUE forecast)
"""

METRICS = """
    ROUND(AVG(ABS(forecast - actual)), 3) AS mae,
    ROUND(SQRT(AVG(POW(forecast - actual, 2))), 3) AS rmse,
    ROUND(100.0 * (SUM(forecast) / SUM(actual) - 1), 1) AS bias_pct
"""


def main():
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(GOLD))
    try:
        con.execute(f"ATTACH '{SILVER.as_posix()}' AS silver (READ_ONLY)")

        last_date = con.execute(
            "SELECT MAX(date) FROM silver.silver_sales"
        ).fetchone()[0]
        train_end = last_date - timedelta(days=TEST_DAYS)
        print(f"Training data: up to {train_end}")
        print(f"Test period:   {train_end + timedelta(days=1)} to {last_date}")

        con.execute(f"""
            CREATE OR REPLACE TABLE baseline_forecasts AS
            WITH test AS (
                SELECT id, cat_id, store_id, date,
                       CAST(units AS DOUBLE) AS actual
                FROM silver.silver_sales
                WHERE date > DATE '{train_end}'
            ),
            last_day AS (
                SELECT id, CAST(units AS DOUBLE) AS forecast
                FROM silver.silver_sales
                WHERE date = DATE '{train_end}'
            ),
            last_week AS (
                SELECT id, dayofweek(date) AS dow,
                       CAST(units AS DOUBLE) AS forecast
                FROM silver.silver_sales
                WHERE date > DATE '{train_end}' - INTERVAL 7 DAY
                  AND date <= DATE '{train_end}'
            ),
            last_28 AS (
                SELECT id, AVG(units) AS forecast
                FROM silver.silver_sales
                WHERE date > DATE '{train_end}' - INTERVAL 28 DAY
                  AND date <= DATE '{train_end}'
                GROUP BY id
            )
            SELECT t.id, t.cat_id, t.store_id, t.date, t.actual,
                   CAST(0 AS DOUBLE) AS zero,
                   d.forecast AS naive_last_day,
                   w.forecast AS seasonal_naive,
                   m.forecast AS moving_avg_28
            FROM test t
            JOIN last_day d ON d.id = t.id
            JOIN last_week w ON w.id = t.id AND w.dow = dayofweek(t.date)
            JOIN last_28 m ON m.id = t.id
        """)

        rows = con.execute("SELECT COUNT(*) FROM baseline_forecasts").fetchone()[0]
        print(f"Forecast rows: {rows:,}")

        print("\n=== 1. Overall accuracy (sorted by RMSE, best first) ===")
        con.sql(f"""
            SELECT model, {METRICS}
            FROM {FORECAST_LONG}
            GROUP BY model
            ORDER BY rmse
        """).show()

        print("\n=== 2. Accuracy by category ===")
        con.sql(f"""
            SELECT cat_id, model, {METRICS}
            FROM {FORECAST_LONG}
            GROUP BY cat_id, model
            ORDER BY cat_id, rmse
        """).show(max_rows=20)
    finally:
        con.close()


if __name__ == "__main__":
    main()
