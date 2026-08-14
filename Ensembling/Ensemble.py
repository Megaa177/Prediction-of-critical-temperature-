"""
XGBoost, CatBoost, AdaBoost trained and evaluated SEPARATELY
(no voting/stacking combination) for superconductor critical-temp prediction.
Split: 70% train / 15% val / 15% test.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
DATA_PATH = "train.csv"        # <-- point this at your combined dataset
TARGET_COL = "critical_temp"   # <-- rename if needed

df = pd.read_csv(DATA_PATH)

non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"Dropping non-numeric columns (encode these separately if needed): {non_numeric_cols}")
    df = df.drop(columns=non_numeric_cols)

df = df.dropna(subset=[TARGET_COL])
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ---------------------------------------------------------------------------
# 2. 70 / 15 / 15 split
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------------------
# 3. Models — each one standalone, nothing combined
# ---------------------------------------------------------------------------
models = {
    "XGBoost": XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "CatBoost": CatBoostRegressor(
        iterations=300, depth=6, learning_rate=0.05,
        random_state=RANDOM_STATE, verbose=False
    ),
    "AdaBoost": AdaBoostRegressor(
        n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE
    ),
}

# ---------------------------------------------------------------------------
# 4. Train + validate each model independently
# ---------------------------------------------------------------------------
def evaluate(model, X_tr, y_tr, X_ev, y_ev):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_ev)
    rmse = np.sqrt(mean_squared_error(y_ev, preds))
    mae = mean_absolute_error(y_ev, preds)
    r2 = r2_score(y_ev, preds)
    return rmse, mae, r2

results_val = []
for name, model in models.items():
    rmse, mae, r2 = evaluate(model, X_train, y_train, X_val, y_val)
    results_val.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
    print(f"[VAL] {name:12s} RMSE={rmse:8.3f}  MAE={mae:8.3f}  R2={r2:6.3f}")

val_df = pd.DataFrame(results_val).sort_values("RMSE")
print("\n=== Validation leaderboard ===")
print(val_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Final test-set evaluation (train on train+val, evaluate once on test)
# ---------------------------------------------------------------------------
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

results_test = []
for name, model in models.items():
    rmse, mae, r2 = evaluate(model, X_trainval, y_trainval, X_test, y_test)
    results_test.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})

test_df = pd.DataFrame(results_test).sort_values("RMSE")
print("\n=== Final test-set leaderboard ===")
print(test_df.to_string(index=False))

test_df.to_csv("separate_models_results.csv", index=False)