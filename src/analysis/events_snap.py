from pathlib import Path
import duckdb

PROJECT = Path(__file__).resolve().parents[2]
SILVER = PROJECT / "data" / "silver" / "retail_silver.duckdb"
BRONZE = PROJECT / "data" / "bronze" / "retail_bronze.duckdb"

SNAP_QUERY = """
    WITH tagged AS (
        SELECT s.state_id, s.cat_id, s.units, s.sell_price,
               CASE s.state_id
                   WHEN 'CA' THEN c.snap_CA
                   WHEN 'TX' THEN c.snap_TX
                   ELSE c.snap_WI
               END AS snap
        FROM silver_sales s
        JOIN bronze.bronze_calendar c ON c.date = s.date
        WHERE NOT (MONTH(s.date) = 12 AND DAY(s.date) = 25)
    ),
    rates AS (
        SELECT state_id, cat_id,
               SUM(units) FILTER (WHERE snap = 1)
                 / COUNT(*) FILTER (WHERE snap = 1 AND sell_price IS NOT NULL)
                 AS snap_rate,
               SUM(units) FILTER (WHERE snap = 0)
                 / COUNT(*) FILTER (WHERE snap = 0 AND sell_price IS NOT NULL)
                 AS normal_rate
        FROM tagged
        GROUP BY state_id, cat_id
    )
    SELECT state_id, cat_id,
           ROUND(normal_rate, 3) AS normal_day,
           ROUND(snap_rate, 3) AS snap_day,
           ROUND(100 * (snap_rate / normal_rate - 1), 1) AS lift_pct
    FROM rates
    ORDER BY cat_id, state_id
"""

EVENT_BASE = """
    WITH daily AS (
        SELECT date,
               SUM(units) / COUNT(*) FILTER (WHERE sell_price IS NOT NULL) AS rate
        FROM silver_sales
        WHERE NOT (MONTH(date) = 12 AND DAY(date) = 25)
        GROUP BY date
    ),
    tagged AS (
        SELECT d.date, d.rate, c.event_name_1, c.event_type_1,
               YEAR(d.date) AS yr, dayofweek(d.date) AS dow
        FROM daily d
        JOIN bronze.bronze_calendar c ON c.date = d.date
    ),
    baseline AS (
        SELECT yr, dow, AVG(rate) AS base_rate
        FROM tagged
        WHERE event_name_1 IS NULL
        GROUP BY yr, dow
    ),
    lifts AS (
        SELECT t.event_type_1, t.event_name_1,
               100 * (t.rate / b.base_rate - 1) AS lift
        FROM tagged t
        JOIN baseline b USING (yr, dow)
        WHERE t.event_name_1 IS NOT NULL
    )
"""

EVENT_TYPE_QUERY = EVENT_BASE + """
    SELECT event_type_1 AS event_type,
           COUNT(*) AS event_days,
           ROUND(AVG(lift), 1) AS avg_lift_pct
    FROM lifts
    GROUP BY event_type
    ORDER BY avg_lift_pct
"""

EVENT_NAME_QUERY = EVENT_BASE + """
    SELECT event_type_1 AS event_type,
           event_name_1 AS event,
           COUNT(*) AS times,
           ROUND(AVG(lift), 1) AS avg_lift_pct
    FROM lifts
    GROUP BY event_type, event
    ORDER BY avg_lift_pct
"""


def main():
    con = duckdb.connect(str(SILVER), read_only=True)
    try:
        con.execute(f"ATTACH '{BRONZE.as_posix()}' AS bronze (READ_ONLY)")

        print("\n=== 1. SNAP effect: units per active product per day ===")
        con.sql(SNAP_QUERY).show(max_rows=20)

        print("\n=== 2. Event effect by type (vs same weekday, same year) ===")
        con.sql(EVENT_TYPE_QUERY).show()

        print("\n=== 3. Event effect by event, lowest to highest ===")
        con.sql(EVENT_NAME_QUERY).show(max_rows=40)
    finally:
        con.close()


if __name__ == "__main__":
    main()
