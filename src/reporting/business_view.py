from pathlib import Path
import duckdb

PROJECT = Path(__file__).resolve().parents[2]
GOLD = PROJECT / "data" / "gold" / "retail_gold.duckdb"
MODELS = ["seasonal_naive", "moving_avg_28", "ma28_weekday", "lgbm_v2"]

LEVELS = [
    ("Product x store x day", ["id", "date"]),
    ("Product x store, 28-day total", ["id"]),
    ("Department x store x day", ["dept_id", "store_id", "date"]),
    ("Category x store x day", ["cat_id", "store_id", "date"]),
    ("Store x day", ["store_id", "date"]),
    ("State x day", ["state_id", "date"]),
    ("All stores x day", ["date"]),
    ("All stores, 28-day total", []),
]


def main():
    con = duckdb.connect(str(GOLD))
    try:
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE long AS
            SELECT model, id, date, cat_id, store_id, actual, forecast,
                   split_part(id, '_', 1) || '_' || split_part(id, '_', 2)
                       AS dept_id,
                   left(store_id, 2) AS state_id
            FROM (UNPIVOT forecasts_v6
                  ON {", ".join(MODELS)}
                  INTO NAME model VALUE forecast)
        """)

        print("=== 1. WAPE (%) at each business level - lower is better ===\n")
        print(f"  {'Level':<32}" + "".join(f"{m:>16}" for m in MODELS))
        for name, keys in LEVELS:
            group = ", ".join(["model"] + keys)
            rows = con.execute(f"""
                WITH agg AS (
                    SELECT {group},
                           SUM(forecast) AS f,
                           SUM(actual) AS a
                    FROM long
                    GROUP BY {group}
                )
                SELECT model, 100 * SUM(ABS(f - a)) / SUM(a) AS wape
                FROM agg
                GROUP BY model
            """).fetchall()
            wape = dict(rows)
            print(f"  {name:<32}"
                  + "".join(f"{wape[m]:>16.1f}" for m in MODELS))

        print("\n=== 2. Overall bias (%) - the same at every level ===\n")
        con.sql("""
            SELECT model,
                   ROUND(100 * (SUM(forecast) / SUM(actual) - 1), 1) AS bias_pct
            FROM long
            GROUP BY model
            ORDER BY model
        """).show()

        print("\n=== 3. 28-day store totals: LightGBM v2 vs actual ===")
        con.sql("""
            SELECT store_id,
                   ROUND(SUM(actual)) AS actual_units,
                   ROUND(SUM(lgbm_v2)) AS forecast_units,
                   ROUND(SUM(lgbm_v2) - SUM(actual)) AS difference,
                   ROUND(100 * (SUM(lgbm_v2) / SUM(actual) - 1), 1)
                       AS error_pct
            FROM forecasts_v6
            GROUP BY store_id
            ORDER BY store_id
        """).show()
    finally:
        con.close()


if __name__ == "__main__":
    main()
