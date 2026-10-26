from pathlib import Path
import duckdb
import lightgbm as lgb
import numpy as np
from train_lgbm_v2 import CATEGORICAL, NUMERIC, FEATURES

PROJECT = Path(__file__).resolve().parents[2]
GOLD = PROJECT / "data" / "gold" / "retail_gold.duckdb"
MODEL_PATH = PROJECT / "models" / "lgbm_v2.txt"
WEIGHTS = [round(0.1 * i, 1) for i in range(11)]

METRICS = """
    ROUND(AVG(ABS(forecast - actual)), 3) AS mae,
    ROUND(SQRT(AVG(POW(forecast - actual, 2))), 3) AS rmse,
    ROUND(100.0 * (SUM(forecast) / SUM(actual) - 1), 1) AS bias_pct
"""
FORECAST_LONG = """
    (UNPIVOT forecasts_v6
     ON ma28_weekday, lgbm_v2, ensemble
     INTO NAME model VALUE forecast)
"""


def score(actual, forecast):
    error = forecast - actual
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    bias = 100 * (forecast.sum() / actual.sum() - 1)
    return mae, rmse, bias


def weight_table(actual, lgbm, baseline):
    results = []
    print(f"  {'w':>4}  {'MAE':>7}  {'RMSE':>7}  {'bias%':>7}")
    for w in WEIGHTS:
        mae, rmse, bias = score(actual, w * lgbm + (1 - w) * baseline)
        results.append((w, rmse))
        print(f"  {w:>4}  {mae:7.4f}  {rmse:7.4f}  {bias:7.2f}")
    return min(results, key=lambda r: r[1])[0]


def main():
    con = duckdb.connect(str(GOLD))
    try:
        codes = ",\n".join(
            f"CAST(DENSE_RANK() OVER (ORDER BY COALESCE({c}, 'none')) - 1 "
            f"AS INTEGER) AS {c}_code"
            for c in CATEGORICAL
        )
        numbers = ",\n".join(f"CAST({c} AS FLOAT) AS {c}" for c in NUMERIC)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE model_data AS
            SELECT dataset, units, {codes}, {numbers}
            FROM origin_features
        """)
        cols = ", ".join(FEATURES)
        valid = con.execute(
            f"SELECT units, {cols} FROM model_data WHERE dataset = 'valid'"
        ).df()

        model = lgb.Booster(model_file=str(MODEL_PATH))
        valid_actual = valid["units"].to_numpy()
        valid_lgbm = np.clip(model.predict(valid[FEATURES]), 0, None)
        valid_base = valid["baseline_fc"].to_numpy()

        print("=== 1. Choosing the weight on VALIDATION (Mar 28 - Apr 24) ===")
        best_w = weight_table(valid_actual, valid_lgbm, valid_base)
        print(f"\n  Chosen weight: w = {best_w}  "
              f"({best_w:.0%} LightGBM, {1 - best_w:.0%} baseline)")

        test = con.execute(
            "SELECT actual, lgbm_v2, ma28_weekday FROM forecasts_v5"
        ).df()
        print("\n=== 2. FOR LEARNING ONLY: every weight on the TEST set ===")
        print("  (Never choose from this table - that would be leakage.)")
        weight_table(test["actual"].to_numpy(), test["lgbm_v2"].to_numpy(),
                     test["ma28_weekday"].to_numpy())

        con.execute(f"""
            CREATE OR REPLACE TABLE forecasts_v6 AS
            SELECT *,
                   {best_w} * lgbm_v2 + {round(1 - best_w, 1)} * ma28_weekday
                       AS ensemble
            FROM forecasts_v5
        """)

        print("\n=== 3. Final test result with the chosen weight ===")
        con.sql(f"""
            SELECT model, {METRICS} FROM {FORECAST_LONG}
            GROUP BY model ORDER BY rmse
        """).show()

        print("\n=== 4. By category ===")
        con.sql(f"""
            SELECT cat_id, model, {METRICS} FROM {FORECAST_LONG}
            GROUP BY cat_id, model ORDER BY cat_id, rmse
        """).show()

        print("\n=== 5. Example rows to check by hand ===")
        con.sql("""
            SELECT date, ROUND(lgbm_v2, 2) AS lgbm_v2,
                   ROUND(ma28_weekday, 2) AS baseline,
                   ROUND(ensemble, 2) AS ensemble, actual
            FROM forecasts_v6
            WHERE id = 'FOODS_3_090_CA_3_evaluation'
            ORDER BY date
            LIMIT 3
        """).show()
    finally:
        con.close()


if __name__ == "__main__":
    main()
