from datetime import timedelta
from pathlib import Path
import duckdb

PROJECT = Path(__file__).resolve().parents[2]
BRONZE = PROJECT / "data" / "bronze" / "retail_bronze.duckdb"
SILVER = PROJECT / "data" / "silver" / "retail_silver.duckdb"
GOLD = PROJECT / "data" / "gold" / "retail_gold.duckdb"
TEST_DAYS = 28
HORIZON = 28
TRAIN_ORIGINS = 12


def main():
    con = duckdb.connect(str(GOLD))
    try:
        con.execute("SET preserve_insertion_order = false")
        con.execute(f"ATTACH '{SILVER.as_posix()}' AS silver (READ_ONLY)")
        con.execute(f"ATTACH '{BRONZE.as_posix()}' AS bronze (READ_ONLY)")

        last_date = con.execute(
            "SELECT MAX(date) FROM silver.silver_sales"
        ).fetchone()[0]
        test_origin = last_date - timedelta(days=TEST_DAYS)
        valid_origin = test_origin - timedelta(days=HORIZON)
        origins = [(test_origin, "test"), (valid_origin, "valid")]
        origins += [
            (valid_origin - timedelta(days=HORIZON * k), "train")
            for k in range(1, TRAIN_ORIGINS + 1)
        ]
        for d, label in origins:
            print(f"  origin {d}  ({label})")

        values = ", ".join(f"(DATE '{d}', '{label}')" for d, label in origins)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE origins AS
            SELECT * FROM (VALUES {values}) v(origin, dataset)
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE targets AS
            SELECT o.origin, o.dataset, CAST(h.range AS INTEGER) AS horizon,
                   CAST(o.origin + CAST(h.range AS INTEGER) AS DATE) AS target_date
            FROM origins o
            CROSS JOIN range(1, {HORIZON + 1}) h
        """)

        print("Computing levels as of each origin...")
        con.execute("""
            CREATE OR REPLACE TEMP TABLE item_window AS
            SELECT o.origin, s.id, s.date, s.units
            FROM origins o
            JOIN silver.silver_sales s
              ON s.date BETWEEN o.origin - 27 AND o.origin
        """)
        con.execute("""
            CREATE OR REPLACE TEMP TABLE levels AS
            SELECT origin, id,
                   AVG(units) AS level_28,
                   AVG(units) FILTER (WHERE date > origin - 7) AS level_7
            FROM item_window
            GROUP BY origin, id
        """)
        con.execute("""
            CREATE OR REPLACE TEMP TABLE wdays AS
            SELECT origin, id, dayofweek(date) AS dow,
                   AVG(units) AS wday_avg_4
            FROM item_window
            GROUP BY origin, id, dow
        """)
        con.execute("""
            CREATE OR REPLACE TEMP TABLE pooled AS
            WITH daily AS (
                SELECT o.origin, s.store_id, s.cat_id, s.date,
                       SUM(s.units) AS units
                FROM origins o
                JOIN silver.silver_sales s
                  ON s.date BETWEEN o.origin - 83 AND o.origin
                GROUP BY o.origin, s.store_id, s.cat_id, s.date
            ),
            by_dow AS (
                SELECT origin, store_id, cat_id, dayofweek(date) AS dow,
                       AVG(units) AS dow_avg
                FROM daily
                GROUP BY origin, store_id, cat_id, dow
            )
            SELECT origin, store_id, cat_id, dow,
                   dow_avg / AVG(dow_avg) OVER
                       (PARTITION BY origin, store_id, cat_id) AS pooled_mult
            FROM by_dow
        """)

        print("Assembling the feature table...")
        con.execute("""
            CREATE OR REPLACE TABLE origin_features AS
            WITH cal AS (
                SELECT date, event_name_1, event_type_1,
                       snap_CA, snap_TX, snap_WI,
                       LEAD(event_name_1, 1) OVER (ORDER BY date) AS event_tomorrow
                FROM bronze.bronze_calendar
            ),
            first_price AS (
                SELECT id, MIN(date) AS first_price_date
                FROM silver.silver_sales
                WHERE sell_price IS NOT NULL
                GROUP BY id
            ),
            max_price AS (
                SELECT id, MAX(sell_price) AS max_price
                FROM silver.silver_sales
                GROUP BY id
            ),
            target_rows AS (
                SELECT t.origin, t.dataset, t.horizon, s.*
                FROM targets t
                JOIN silver.silver_sales s ON s.date = t.target_date
            ),
            last_year AS (
                SELECT id, date, units
                FROM silver.silver_sales
                WHERE date IN (SELECT target_date - 364 FROM targets)
            )
            SELECT r.dataset, r.origin, r.horizon,
                   r.id, r.item_id, r.dept_id, r.cat_id, r.store_id,
                   r.state_id, r.date,
                   CAST(r.units AS FLOAT) AS units,
                   dayofweek(r.date) AS wday,
                   day(r.date) AS mday,
                   month(r.date) AS month,
                   c.event_name_1, c.event_type_1, c.event_tomorrow,
                   CASE r.state_id
                       WHEN 'CA' THEN c.snap_CA
                       WHEN 'TX' THEN c.snap_TX
                       ELSE c.snap_WI
                   END AS snap,
                   r.sell_price,
                   r.sell_price / mp.max_price AS price_ratio,
                   date_diff('day', fp.first_price_date, r.date)
                       AS product_age_days,
                   lv.level_28, lv.level_7, wd.wday_avg_4, pm.pooled_mult,
                   lv.level_28 * pm.pooled_mult AS baseline_fc,
                   ly.units AS lag_364
            FROM target_rows r
            JOIN cal c ON c.date = r.date
            JOIN first_price fp ON fp.id = r.id
            JOIN max_price mp ON mp.id = r.id
            JOIN levels lv ON lv.origin = r.origin AND lv.id = r.id
            JOIN wdays wd ON wd.origin = r.origin AND wd.id = r.id
                         AND wd.dow = dayofweek(r.date)
            JOIN pooled pm ON pm.origin = r.origin
                          AND pm.store_id = r.store_id
                          AND pm.cat_id = r.cat_id
                          AND pm.dow = dayofweek(r.date)
            LEFT JOIN last_year ly ON ly.id = r.id AND ly.date = r.date - 364
            WHERE r.sell_price IS NOT NULL
              AND NOT (MONTH(r.date) = 12 AND DAY(r.date) = 25)
        """)

        print("\n=== 1. Rows per dataset ===")
        con.sql("""
            SELECT dataset, COUNT(DISTINCT origin) AS origins,
                   COUNT(*) AS rows, MIN(date) AS first_target,
                   MAX(date) AS last_target
            FROM origin_features
            GROUP BY dataset
            ORDER BY MIN(date)
        """).show()

        print("\n=== 2. Check: baseline_fc should equal ma28_weekday on the test set ===")
        con.sql("""
            SELECT COUNT(*) AS matched_rows,
                   ROUND(MAX(ABS(o.baseline_fc - f.ma28_weekday)), 6)
                       AS max_difference
            FROM origin_features o
            JOIN forecasts_v2 f ON f.id = o.id AND f.date = o.date
            WHERE o.dataset = 'test'
        """).show()

        print("\n=== 3. Example: one product, first week of the test ===")
        con.sql("""
            SELECT date, horizon, wday,
                   ROUND(level_28, 2) AS level_28,
                   ROUND(pooled_mult, 3) AS pooled_mult,
                   ROUND(baseline_fc, 2) AS baseline_fc,
                   units
            FROM origin_features
            WHERE dataset = 'test' AND id = 'FOODS_3_090_CA_3_evaluation'
              AND horizon <= 7
            ORDER BY horizon
        """).show()
    finally:
        con.close()


if __name__ == "__main__":
    main()
