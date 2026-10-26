from pathlib import Path
import duckdb
import lightgbm as lgb
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
GOLD = PROJECT / "data" / "gold" / "retail_gold.duckdb"
MODEL_PATH = PROJECT / "models" / "lgbm_v2.txt"

CATEGORICAL = ["item_id", "dept_id", "cat_id", "store_id", "state_id",
               "event_name_1", "event_type_1", "event_tomorrow"]
NUMERIC = ["horizon", "wday", "mday", "month", "snap", "sell_price",
           "price_ratio", "product_age_days", "level_28", "level_7",
           "wday_avg_4", "pooled_mult", "baseline_fc", "lag_364"]
CAT_CODES = [f"{c}_code" for c in CATEGORICAL]
FEATURES = CAT_CODES + NUMERIC

PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.08,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "seed": 42,
    "verbose": -1,
}

METRICS = """
    ROUND(AVG(ABS(forecast - actual)), 3) AS mae,
    ROUND(SQRT(AVG(POW(forecast - actual, 2))), 3) AS rmse,
    ROUND(100.0 * (SUM(forecast) / SUM(actual) - 1), 1) AS bias_pct
"""
FORECAST_LONG = """
    (UNPIVOT forecasts_v5
     ON moving_avg_28, ma28_weekday, lgbm, lgbm_v2
     INTO NAME model VALUE forecast)
"""


def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(GOLD))
    try:
        print("Encoding categories...")
        codes = ",\n".join(
            f"CAST(DENSE_RANK() OVER (ORDER BY COALESCE({c}, 'none')) - 1 "
            f"AS INTEGER) AS {c}_code"
            for c in CATEGORICAL
        )
        numbers = ",\n".join(f"CAST({c} AS FLOAT) AS {c}" for c in NUMERIC)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE model_data AS
            SELECT dataset, id, date, units,
                   {codes},
                   {numbers}
            FROM origin_features
        """)

        cols = ", ".join(FEATURES)
        print("Loading data into memory...")
        train = con.execute(
            f"SELECT units, {cols} FROM model_data WHERE dataset = 'train'"
        ).df()
        valid = con.execute(
            f"SELECT units, {cols} FROM model_data WHERE dataset = 'valid'"
        ).df()
        test = con.execute(
            f"SELECT id, date, {cols} FROM model_data WHERE dataset = 'test'"
        ).df()
        print(f"  train: {len(train):,}  valid: {len(valid):,}  test: {len(test):,}")

        dtrain = lgb.Dataset(train[FEATURES], label=train["units"],
                             categorical_feature=CAT_CODES)
        dvalid = lgb.Dataset(valid[FEATURES], label=valid["units"],
                             categorical_feature=CAT_CODES, reference=dtrain)

        print("\nTraining...")
        model = lgb.train(
            PARAMS, dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid], valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )
        model.save_model(str(MODEL_PATH))
        print(f"Best number of trees: {model.best_iteration}")

        test["lgbm_v2"] = np.clip(
            model.predict(test[FEATURES], num_iteration=model.best_iteration),
            0, None,
        )
        con.register("predictions", test[["id", "date", "lgbm_v2"]])
        con.execute("""
            CREATE OR REPLACE TABLE forecasts_v5 AS
            SELECT f.*, p.lgbm_v2
            FROM forecasts_v4 f
            JOIN predictions p
              ON p.id = f.id AND CAST(p.date AS DATE) = f.date
        """)
        rows = con.execute("SELECT COUNT(*) FROM forecasts_v5").fetchone()[0]
        print(f"Scored rows: {rows:,}")

        print("\n=== 1. Test accuracy (sorted by RMSE, best first) ===")
        con.sql(f"""
            SELECT model, {METRICS} FROM {FORECAST_LONG}
            GROUP BY model ORDER BY rmse
        """).show()

        print("\n=== 2. By category ===")
        con.sql(f"""
            SELECT cat_id, model, {METRICS} FROM {FORECAST_LONG}
            GROUP BY cat_id, model ORDER BY cat_id, rmse
        """).show(max_rows=20)

        print("\n=== 3. Feature importance (share of total gain) ===")
        gains = model.feature_importance(importance_type="gain")
        total = gains.sum()
        for name, gain in sorted(zip(FEATURES, gains), key=lambda x: -x[1]):
            print(f"  {name:<22} {100 * gain / total:5.1f}%")
    finally:
        con.close()


if __name__ == "__main__":
    main()
