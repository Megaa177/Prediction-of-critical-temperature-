"""
XGBoost, CatBoost, AdaBoost trained and evaluated separately
(no voting/stacking combination) for superconductor critical-temperature prediction.

Split:
    70% Train
    15% Validation
    15% Test

Reports:
    RMSE
    MAE
    R2

For each model:
    1. Train + Validation results
    2. Final Test results

The dataset path is automatically taken from the same folder
as this Python file.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TARGET_COL = "critical_temp"


# ============================================================
# 1. LOAD DATA
# ============================================================

# Automatically find train.csv in the SAME folder as Ensemble.py
DATA_PATH = Path(__file__).resolve().parent / "train.csv"

print("Looking for dataset at:")
print(DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nERROR: train.csv was not found here:\n{DATA_PATH}\n\n"
        "Make sure train.csv is inside the same folder as Ensemble.py."
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. CHECK TARGET COLUMN
# ============================================================

if TARGET_COL not in df.columns:
    raise ValueError(
        f"\nERROR: Target column '{TARGET_COL}' was not found.\n"
        f"Available columns are:\n{df.columns.tolist()}"
    )


# ============================================================
# 3. REMOVE NON-NUMERIC COLUMNS
# ============================================================

non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

if non_numeric_cols:
    print("\nDropping non-numeric columns:")
    print(non_numeric_cols)

    df = df.drop(columns=non_numeric_cols)


# ============================================================
# 4. REMOVE ROWS WITH MISSING TARGET
# ============================================================

df = df.dropna(subset=[TARGET_COL])


# ============================================================
# 5. FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print("\nNumber of features:", X.shape[1])
print("Number of samples:", X.shape[0])


# ============================================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First split:
# 70% Train
# 30% Temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE
)


# Second split:
# 15% Validation
# 15% Test

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE
)


print("\n============================================================")
print("DATA SPLIT")
print("============================================================")

print("Training set:   ", X_train.shape)
print("Validation set: ", X_val.shape)
print("Test set:       ", X_test.shape)


# ============================================================
# 7. DEFINE MODELS
# ============================================================

models = {

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "CatBoost": CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbose=False
    ),

    "AdaBoost": AdaBoostRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=RANDOM_STATE
    )
}


# ============================================================
# 8. METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return rmse, mae, r2


# ============================================================
# 9. TRAIN EACH MODEL
# ============================================================

validation_results = []


print("\n\n============================================================")
print("TRAINING + VALIDATION RESULTS")
print("============================================================")


for name, model in models.items():

    print(f"\nTraining {name}...")

    # --------------------------------------------------------
    # Train ONLY on training set
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predictions on TRAIN
    # --------------------------------------------------------

    train_predictions = model.predict(
        X_train
    )


    # --------------------------------------------------------
    # Predictions on VALIDATION
    # --------------------------------------------------------

    val_predictions = model.predict(
        X_val
    )


    # --------------------------------------------------------
    # TRAIN metrics
    # --------------------------------------------------------

    train_rmse, train_mae, train_r2 = calculate_metrics(
        y_train,
        train_predictions
    )


    # --------------------------------------------------------
    # VALIDATION metrics
    # --------------------------------------------------------

    val_rmse, val_mae, val_r2 = calculate_metrics(
        y_val,
        val_predictions
    )


    # Save results
    validation_results.append({

        "Model": name,

        "Train_RMSE": train_rmse,
        "Train_MAE": train_mae,
        "Train_R2": train_r2,

        "Val_RMSE": val_rmse,
        "Val_MAE": val_mae,
        "Val_R2": val_r2
    })


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "-" * 60)

    print(name)

    print("-" * 60)

    print(
        f"TRAIN      -> "
        f"RMSE = {train_rmse:.3f}   "
        f"MAE = {train_mae:.3f}   "
        f"R2 = {train_r2:.3f}"
    )

    print(
        f"VALIDATION -> "
        f"RMSE = {val_rmse:.3f}   "
        f"MAE = {val_mae:.3f}   "
        f"R2 = {val_r2:.3f}"
    )


# ============================================================
# 10. VALIDATION LEADERBOARD
# ============================================================

validation_df = pd.DataFrame(
    validation_results
)

validation_df = validation_df.sort_values(
    by="Val_RMSE"
)


print("\n\n============================================================")
print("VALIDATION LEADERBOARD")
print("============================================================")

print(
    validation_df.to_string(
        index=False
    )
)


# ============================================================
# 11. COMBINE TRAIN + VALIDATION
# ============================================================

X_trainval = pd.concat(
    [X_train, X_val],
    axis=0
)

y_trainval = pd.concat(
    [y_train, y_val],
    axis=0
)


print("\n\n============================================================")
print("FINAL TEST EVALUATION")
print("============================================================")

print(
    "\nEach model will now be retrained using "
    "TRAIN + VALIDATION data (85%)."
)

print(
    "The TEST set (15%) remains completely untouched "
    "until final evaluation."
)


# ============================================================
# 12. FINAL TEST RESULTS
# ============================================================

test_results = []


for name, model in models.items():

    print(f"\nRetraining {name} on Train + Validation...")


    # --------------------------------------------------------
    # Train using 85% of the data
    # --------------------------------------------------------

    model.fit(
        X_trainval,
        y_trainval
    )


    # --------------------------------------------------------
    # Predictions on TRAIN + VALIDATION
    # --------------------------------------------------------

    trainval_predictions = model.predict(
        X_trainval
    )


    # --------------------------------------------------------
    # Predictions on TEST
    # --------------------------------------------------------

    test_predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # TRAIN + VALIDATION metrics
    # --------------------------------------------------------

    trainval_rmse, trainval_mae, trainval_r2 = calculate_metrics(
        y_trainval,
        trainval_predictions
    )


    # --------------------------------------------------------
    # TEST metrics
    # --------------------------------------------------------

    test_rmse, test_mae, test_r2 = calculate_metrics(
        y_test,
        test_predictions
    )


    # Save results
    test_results.append({

        "Model": name,

        "TrainVal_RMSE": trainval_rmse,
        "TrainVal_MAE": trainval_mae,
        "TrainVal_R2": trainval_r2,

        "Test_RMSE": test_rmse,
        "Test_MAE": test_mae,
        "Test_R2": test_r2
    })


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "-" * 60)

    print(name)

    print("-" * 60)

    print(
        f"TRAIN+VAL -> "
        f"RMSE = {trainval_rmse:.3f}   "
        f"MAE = {trainval_mae:.3f}   "
        f"R2 = {trainval_r2:.3f}"
    )

    print(
        f"TEST      -> "
        f"RMSE = {test_rmse:.3f}   "
        f"MAE = {test_mae:.3f}   "
        f"R2 = {test_r2:.3f}"
    )


# ============================================================
# 13. FINAL TEST LEADERBOARD
# ============================================================

test_df = pd.DataFrame(
    test_results
)

test_df = test_df.sort_values(
    by="Test_RMSE"
)


print("\n\n============================================================")
print("FINAL TEST LEADERBOARD")
print("============================================================")

print(
    test_df.to_string(
        index=False
    )
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

validation_output = (
    Path(__file__).resolve().parent
    / "separate_models_train_val_results.csv"
)

test_output = (
    Path(__file__).resolve().parent
    / "separate_models_test_results.csv"
)


validation_df.to_csv(
    validation_output,
    index=False
)


test_df.to_csv(
    test_output,
    index=False
)


print("\n\n============================================================")
print("RESULT FILES SAVED")
print("============================================================")

print("Validation results:")
print(validation_output)

print("\nTest results:")
print(test_output)

print("\nDone!")