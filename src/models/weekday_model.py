from datetime import timedelta
from pathlib import Path
import duckdb

PROJECT = Path(__file__).resolve().parents[2]
SILVER = PROJECT / "data" / "silver" / "retail_silver.duckdb"
GOLD = PROJECT / "data" / "gold" / "retail_gold.duckdb"
TEST_DAYS = 28
PROFILE_DAYS = 84

FORECAST_LONG = """
    (UNPIVOT forecasts_v2
     ON zero, naive_last_day, seasonal_naive, moving_avg_28, ma28_weekday
     INTO NAME model VALUE forecast)
"""

METRICS = """
    ROUND(AVG(ABS(forecast - actual)), 3) AS mae,
    ROUND(SQRT(AVG(POW(forecast - actual, 2))), 3) AS rmse,
    ROUND(100.0 * (SUM(forecast) / SUM(actual) - 1), 1) AS bias_pct
"""


def main():
    con = duckdb.connect(str(GOLD))
    try:
        con.execute(f"ATTACH '{SILVER.as_posix()}' AS silver (READ_ONLY)")

        last_date = con.execute(
            "SELECT MAX(date) FROM silver.silver_sales"
        ).fetchone()[0]
        train_end = last_date - timedelta(days=TEST_DAYS)
        print(f"Weekday profile learned from the {PROFILE_DAYS} days up to {train_end}")

        con.execute(f"""
            CREATE OR REPLACE TABLE weekday_profile AS
            WITH daily AS (
                SELECT store_id, cat_id, date, SUM(units) AS units
                FROM silver.silver_sales
                WHERE date > DATE '{train_end}' - INTERVAL {PROFILE_DAYS} DAY
                  AND date <= DATE '{train_end}'
                GROUP BY store_id, cat_id, date
            ),
            by_dow AS (
                SELECT store_id, cat_id, dayofweek(date) AS dow,
                       AVG(units) AS dow_avg
                FROM daily
                GROUP BY store_id, cat_id, dow
            )
            SELECT store_id, cat_id, dow,
                   dow_avg / AVG(dow_avg) OVER (PARTITION BY store_id, cat_id)
                       AS multiplier
            FROM by_dow
        """)

        con.execute("""
            CREATE OR REPLACE TABLE forecasts_v2 AS
            SELECT b.*,
                   b.moving_avg_28 * p.multiplier AS ma28_weekday
            FROM baseline_forecasts b
            JOIN weekday_profile p
              ON p.store_id = b.store_id
             AND p.cat_id = b.cat_id
             AND p.dow = dayofweek(b.date)
        """)

        rows = con.execute("SELECT COUNT(*) FROM forecasts_v2").fetchone()[0]
        print(f"Forecast rows: {rows:,}")

        print("\n=== 1. Example weekday profile: store CA_3 ===")
        con.sql("""
            SELECT cat_id,
                   ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][dow + 1] AS weekday,
                   ROUND(multiplier, 2) AS multiplier
            FROM weekday_profile
            WHERE store_id = 'CA_3'
            ORDER BY cat_id, (dow + 6) % 7
        """).show(max_rows=25)

        print("\n=== 2. Overall accuracy (sorted by RMSE, best first) ===")
        con.sql(f"""
            SELECT model, {METRICS}
            FROM {FORECAST_LONG}
            GROUP BY model
            ORDER BY rmse
        """).show()

        print("\n=== 3. Old best vs new model, by category ===")
        con.sql(f"""
            SELECT cat_id, model, {METRICS}
            FROM {FORECAST_LONG}
            WHERE model IN ('moving_avg_28', 'ma28_weekday')
            GROUP BY cat_id, model
            ORDER BY cat_id, rmse
        """).show()
    finally:
        con.close()


if __name__ == "__main__":
    main()
