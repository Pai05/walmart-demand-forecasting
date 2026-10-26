from pathlib import Path
import duckdb

PROJECT = Path(__file__).resolve().parents[2]
SILVER = PROJECT / "data" / "silver" / "retail_silver.duckdb"

BASE = """
    SELECT YEAR(date) AS year,
           COUNT(DISTINCT date) AS days,
           ROUND(SUM(units) / COUNT(DISTINCT date), 0) AS units_per_day,
           ROUND(COUNT(*) FILTER (WHERE sell_price IS NOT NULL)
                 / COUNT(DISTINCT date), 0) AS active_items_per_day,
           ROUND(SUM(units)
                 / COUNT(*) FILTER (WHERE sell_price IS NOT NULL), 3)
                 AS units_per_active_item
    FROM silver_sales
    WHERE NOT (MONTH(date) = 12 AND DAY(date) = 25)
      {extra_filter}
    GROUP BY year
    ORDER BY year
"""

tests = {
    "A. All available days (Christmas excluded)": "",
    "B. Fair comparison: February to April only, every year": "AND MONTH(date) BETWEEN 2 AND 4",
}


def main():
    con = duckdb.connect(str(SILVER), read_only=True)
    try:
        for title, extra in tests.items():
            print(f"\n=== {title} ===")
            con.sql(BASE.format(extra_filter=extra)).show()
    finally:
        con.close()


if __name__ == "__main__":
    main()
