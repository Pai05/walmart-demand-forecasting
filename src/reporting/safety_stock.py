import sys
from pathlib import Path
import duckdb
import lightgbm as lgb
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src" / "models"))
from train_lgbm_v2 import CATEGORICAL, NUMERIC, FEATURES

GOLD = PROJECT / "data" / "gold" / "retail_gold.duckdb"
MODEL_PATH = PROJECT / "models" / "lgbm_v2.txt"
SERVICE_LEVELS = [0.5, 0.8, 0.9, 0.95, 0.99]
LEVEL_LIST = "[" + ", ".join(str(s) for s in SERVICE_LEVELS) + "]"
DEPT = "split_part(id, '_', 1) || '_' || split_part(id, '_', 2)"
STORE = "split_part(id, '_', 4) || '_' || split_part(id, '_', 5)"


def main():
    con = duckdb.connect(str(GOLD))
    try:
        print("Predicting the validation month with the saved model...")
        codes = ",\n".join(
            f"CAST(DENSE_RANK() OVER (ORDER BY COALESCE({c}, 'none')) - 1 "
            f"AS INTEGER) AS {c}_code"
            for c in CATEGORICAL
        )
        numbers = ",\n".join(f"CAST({c} AS FLOAT) AS {c}" for c in NUMERIC)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE model_data AS
            SELECT dataset, id, date, units, {codes}, {numbers}
            FROM origin_features
        """)
        cols = ", ".join(FEATURES)
        valid = con.execute(
            f"SELECT id, date, units, {cols} FROM model_data "
            f"WHERE dataset = 'valid'"
        ).df()
        model = lgb.Booster(model_file=str(MODEL_PATH))
        valid["forecast"] = np.clip(model.predict(valid[FEATURES]), 0, None)

        con.register("valid_preds", valid[["id", "date", "units", "forecast"]])
        con.execute("""
            CREATE OR REPLACE TABLE lgbm_v2_valid_forecasts AS
            SELECT id, CAST(date AS DATE) AS date,
                   units AS actual, forecast
            FROM valid_preds
        """)

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE valid_agg AS
            SELECT {DEPT} AS dept_id, {STORE} AS store_id, date,
                   SUM(actual) AS a, SUM(forecast) AS f
            FROM lgbm_v2_valid_forecasts
            GROUP BY ALL
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE test_agg AS
            SELECT {DEPT} AS dept_id, store_id, date,
                   SUM(actual) AS a, SUM(lgbm_v2) AS f
            FROM forecasts_v6
            GROUP BY ALL
        """)

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE buffers AS
            WITH q AS (
                SELECT dept_id,
                       {LEVEL_LIST} AS levels,
                       quantile_cont(a / f - 1, {LEVEL_LIST}) AS buffers
                FROM valid_agg
                WHERE f > 0
                GROUP BY dept_id
            )
            SELECT dept_id,
                   unnest(levels) AS service_level,
                   unnest(buffers) AS buffer
            FROM q
            UNION ALL
            SELECT DISTINCT dept_id, 0.0 AS service_level, 0.0 AS buffer
            FROM valid_agg
        """)

        print("\n=== 1. Buffer per department (learned on validation) ===")
        con.sql("""
            PIVOT (
                SELECT dept_id,
                       CASE WHEN service_level = 0 THEN 'none'
                            ELSE CAST(CAST(service_level * 100 AS INTEGER)
                                      AS VARCHAR) || '%' END AS target,
                       ROUND(100 * buffer, 1) AS buffer_pct
                FROM buffers
                WHERE service_level IN (0.5, 0.9, 0.95, 0.99)
            )
            ON target IN ('50%', '90%', '95%', '99%')
            USING FIRST(buffer_pct)
            ORDER BY dept_id
        """).show()

        for label, table in [("VALIDATION (sanity check)", "valid_agg"),
                             ("TEST (the real result)", "test_agg")]:
            print(f"\n=== Trade-off on {label} ===")
            con.sql(f"""
                SELECT CASE WHEN b.service_level = 0 THEN 'no buffer'
                            ELSE CAST(CAST(b.service_level * 100 AS INTEGER)
                                      AS VARCHAR) || '%' END AS target,
                       ROUND(100 * AVG(CASE WHEN t.f * (1 + b.buffer) >= t.a
                                            THEN 1 ELSE 0 END), 1)
                           AS achieved_pct,
                       ROUND(100 * (SUM(t.f * (1 + b.buffer)) / SUM(t.f) - 1), 1)
                           AS extra_stock_pct,
                       ROUND(SUM(GREATEST(t.a - t.f * (1 + b.buffer), 0)))
                           AS units_short,
                       ROUND(SUM(GREATEST(t.f * (1 + b.buffer) - t.a, 0)))
                           AS units_leftover
                FROM {table} t
                JOIN buffers b ON b.dept_id = t.dept_id
                GROUP BY b.service_level
                ORDER BY b.service_level
            """).show()
    finally:
        con.close()


if __name__ == "__main__":
    main()
