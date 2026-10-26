from datetime import timedelta
from pathlib import Path
import duckdb

PROJECT = Path(__file__).resolve().parents[2]
SILVER = PROJECT / "data" / "silver" / "retail_silver.duckdb"
GOLD = PROJECT / "data" / "gold" / "retail_gold.duckdb"
TEST_DAYS = 28
WINDOW = 28
DAMPING = 0.5

FORECAST_LONG = """
    (UNPIVOT forecasts_v3
     ON moving_avg_28, ma28_weekday, ma28_trend, ma28_damped
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

        con.execute(f"""
            CREATE OR REPLACE TABLE trend_factors AS
            WITH windows AS (
                SELECT store_id, cat_id, units, sell_price,
                       CASE
                           WHEN date > DATE '{train_end}' - INTERVAL {WINDOW} DAY
                               THEN 'recent'
                           ELSE 'previous'
                       END AS period
                FROM silver.silver_sales
                WHERE date > DATE '{train_end}' - INTERVAL {2 * WINDOW} DAY
                  AND date <= DATE '{train_end}'
            ),
            rates AS (
                SELECT store_id, cat_id,
                       SUM(units) FILTER (WHERE period = 'recent')
                         / COUNT(*) FILTER (WHERE period = 'recent'
                                              AND sell_price IS NOT NULL)
                         AS recent_rate,
                       SUM(units) FILTER (WHERE period = 'previous')
                         / COUNT(*) FILTER (WHERE period = 'previous'
                                              AND sell_price IS NOT NULL)
                         AS previous_rate
                FROM windows
                GROUP BY store_id, cat_id
            )
            SELECT store_id, cat_id, previous_rate, recent_rate,
                   recent_rate / previous_rate AS trend
            FROM rates
        """)

        con.execute(f"""
            CREATE OR REPLACE TABLE forecasts_v3 AS
            SELECT f.*,
                   f.ma28_weekday * t.trend AS ma28_trend,
                   f.ma28_weekday * (1 + {DAMPING} * (t.trend - 1)) AS ma28_damped
            FROM forecasts_v2 f
            JOIN trend_factors t
              ON t.store_id = f.store_id AND t.cat_id = f.cat_id
        """)

        rows = con.execute("SELECT COUNT(*) FROM forecasts_v3").fetchone()[0]
        print(f"Forecast rows: {rows:,}")

        print("\n=== 1. Trend factors (1.05 = sales 5% higher in the last 28 days) ===")
        con.sql("""
            SELECT store_id, cat_id,
                   ROUND(previous_rate, 3) AS previous_rate,
                   ROUND(recent_rate, 3) AS recent_rate,
                   ROUND(trend, 3) AS trend
            FROM trend_factors
            ORDER BY cat_id, store_id
        """).show(max_rows=40)

        print("\n=== 2. Overall accuracy (sorted by RMSE, best first) ===")
        con.sql(f"""
            SELECT model, {METRICS}
            FROM {FORECAST_LONG}
            GROUP BY model
            ORDER BY rmse
        """).show()

        print("\n=== 3. Accuracy by category ===")
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
