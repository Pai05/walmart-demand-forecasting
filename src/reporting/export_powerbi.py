from pathlib import Path
import duckdb

PROJECT = Path(__file__).resolve().parents[2]
GOLD = PROJECT / "data" / "gold" / "retail_gold.duckdb"
SILVER = PROJECT / "data" / "silver" / "retail_silver.duckdb"
OUT = PROJECT / "data" / "powerbi"

EXPORTS = {
    "forecast_daily": """
        SELECT store_id,
               left(store_id, 2) AS state_id,
               cat_id,
               split_part(id, '_', 1) || '_' || split_part(id, '_', 2) AS dept_id,
               date,
               SUM(actual) AS actual,
               ROUND(SUM(lgbm_v2), 2) AS lgbm_v2,
               ROUND(SUM(ma28_weekday), 2) AS ma28_weekday,
               ROUND(SUM(seasonal_naive), 2) AS seasonal_naive
        FROM forecasts_v6
        GROUP BY ALL
        ORDER BY store_id, dept_id, date
    """,
    "sales_history": """
        SELECT store_id, state_id, cat_id, date,
               SUM(units) AS units,
               ROUND(SUM(units * sell_price), 2) AS revenue
        FROM silver.silver_sales
        GROUP BY ALL
        ORDER BY store_id, cat_id, date
    """,
    "model_metrics": """
        SELECT model,
               ROUND(AVG(ABS(forecast - actual)), 3) AS mae,
               ROUND(SQRT(AVG(POW(forecast - actual, 2))), 3) AS rmse,
               ROUND(100 * (SUM(forecast) / SUM(actual) - 1), 1) AS bias_pct
        FROM (UNPIVOT forecasts_v6
              ON zero, naive_last_day, seasonal_naive, moving_avg_28,
                 ma28_weekday, lgbm, lgbm_v2
              INTO NAME model VALUE forecast)
        GROUP BY model
        ORDER BY rmse
    """,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(GOLD), read_only=True)
    try:
        con.execute(f"ATTACH '{SILVER.as_posix()}' AS silver (READ_ONLY)")
        for name, sql in EXPORTS.items():
            path = OUT / f"{name}.csv"
            con.execute(f"COPY ({sql}) TO '{path.as_posix()}' (HEADER, DELIMITER ',')")
            rows = con.execute(f"SELECT COUNT(*) FROM ({sql})").fetchone()[0]
            print(f"  {name}.csv: {rows:,} rows")
    finally:
        con.close()


if __name__ == "__main__":
    main()
