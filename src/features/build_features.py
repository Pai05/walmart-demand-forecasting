from datetime import timedelta
from pathlib import Path
import duckdb

PROJECT = Path(__file__).resolve().parents[2]
BRONZE = PROJECT / "data" / "bronze" / "retail_bronze.duckdb"
SILVER = PROJECT / "data" / "silver" / "retail_silver.duckdb"
GOLD = PROJECT / "data" / "gold" / "retail_gold.duckdb"
TEST_DAYS = 28
TRAIN_DAYS = 364
HISTORY_BUFFER = 400


def main():
    con = duckdb.connect(str(GOLD))
    try:
        con.execute("SET preserve_insertion_order = false")
        con.execute(f"ATTACH '{SILVER.as_posix()}' AS silver (READ_ONLY)")
        con.execute(f"ATTACH '{BRONZE.as_posix()}' AS bronze (READ_ONLY)")

        last_date = con.execute(
            "SELECT MAX(date) FROM silver.silver_sales"
        ).fetchone()[0]
        train_end = last_date - timedelta(days=TEST_DAYS)
        feature_start = train_end - timedelta(days=TRAIN_DAYS - 1)
        history_start = feature_start - timedelta(days=HISTORY_BUFFER)

        print(f"Training rows: {feature_start} to {train_end}")
        print(f"Test rows:     {train_end + timedelta(days=1)} to {last_date}")
        print("Building features (this may take a few minutes)...")

        con.execute(f"""
            CREATE OR REPLACE TABLE features AS
            WITH base AS (
                SELECT id, item_id, dept_id, cat_id, store_id, state_id,
                       date, units, sell_price
                FROM silver.silver_sales
                WHERE date >= DATE '{history_start}'
            ),
            lagged AS (
                SELECT *,
                       LAG(units, 28)  OVER w AS lag_28,
                       LAG(units, 35)  OVER w AS lag_35,
                       LAG(units, 364) OVER w AS lag_364,
                       AVG(units) OVER (PARTITION BY id ORDER BY date
                           ROWS BETWEEN 34 PRECEDING AND 28 PRECEDING)
                           AS rmean_7_lag28,
                       AVG(units) OVER (PARTITION BY id ORDER BY date
                           ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING)
                           AS rmean_28_lag28,
                       MAX(sell_price) OVER (PARTITION BY id) AS max_price
                FROM base
                WINDOW w AS (PARTITION BY id ORDER BY date)
            ),
            cal AS (
                SELECT date, event_name_1, event_type_1,
                       snap_CA, snap_TX, snap_WI,
                       LEAD(event_name_1, 1) OVER (ORDER BY date)
                           AS event_tomorrow
                FROM bronze.bronze_calendar
            ),
            first_price AS (
                SELECT id, MIN(date) AS first_price_date
                FROM silver.silver_sales
                WHERE sell_price IS NOT NULL
                GROUP BY id
            )
            SELECT l.id, l.item_id, l.dept_id, l.cat_id, l.store_id,
                   l.state_id, l.date,
                   CAST(l.units AS FLOAT) AS units,
                   dayofweek(l.date) AS wday,
                   day(l.date) AS mday,
                   month(l.date) AS month,
                   c.event_name_1, c.event_type_1, c.event_tomorrow,
                   CASE l.state_id
                       WHEN 'CA' THEN c.snap_CA
                       WHEN 'TX' THEN c.snap_TX
                       ELSE c.snap_WI
                   END AS snap,
                   l.sell_price,
                   l.sell_price / l.max_price AS price_ratio,
                   date_diff('day', f.first_price_date, l.date)
                       AS product_age_days,
                   l.lag_28, l.lag_35, l.lag_364,
                   l.rmean_7_lag28, l.rmean_28_lag28,
                   l.date > DATE '{train_end}' AS is_test
            FROM lagged l
            JOIN cal c ON c.date = l.date
            JOIN first_price f ON f.id = l.id
            WHERE l.date >= DATE '{feature_start}'
              AND l.sell_price IS NOT NULL
              AND NOT (MONTH(l.date) = 12 AND DAY(l.date) = 25)
        """)

        print("\n=== 1. Rows in each set ===")
        con.sql("""
            SELECT CASE WHEN is_test THEN 'test' ELSE 'train' END AS dataset,
                   COUNT(*) AS rows,
                   MIN(date) AS first_date,
                   MAX(date) AS last_date
            FROM features
            GROUP BY dataset
            ORDER BY dataset DESC
        """).show()

        print("\n=== 2. Missing values in the lag features ===")
        con.sql("""
            SELECT COUNT(*) FILTER (WHERE lag_28 IS NULL) AS null_lag_28,
                   COUNT(*) FILTER (WHERE lag_364 IS NULL) AS null_lag_364,
                   COUNT(*) FILTER (WHERE rmean_28_lag28 IS NULL)
                       AS null_rmean_28
            FROM features
        """).show()

        print("\n=== 3. Example: one product, last 7 training days ===")
        con.sql(f"""
            SELECT date, wday, snap, event_tomorrow, units,
                   lag_28, ROUND(rmean_28_lag28, 2) AS rmean_28,
                   lag_364, ROUND(price_ratio, 2) AS price_ratio
            FROM features
            WHERE id = 'FOODS_3_090_CA_3_evaluation'
              AND date > DATE '{train_end}' - INTERVAL 7 DAY
              AND date <= DATE '{train_end}'
            ORDER BY date
        """).show()
    finally:
        con.close()


if __name__ == "__main__":
    main()
