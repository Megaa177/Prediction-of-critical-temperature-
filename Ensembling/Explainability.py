"""
Ensemble Model Explainability & Interpretability (XGBoost, CatBoost, AdaBoost)
for Superconductor Critical Temperature Prediction.

This script provides comprehensive model explainability without modifying existing code:
    1. Built-in Feature Importance (XGBoost, CatBoost, AdaBoost)
    2. Permutation Feature Importance (Model-Agnostic evaluation on Test Set)
    3. SHAP (SHapley Additive exPlanations) Analysis & Summary Plots
    4. Partial Dependence Analysis for Top Driver Features
    5. Local / Instance-Level Prediction Decomposition (High vs Low Critical Temp)

Outputs:
    - Saved plots (PNG) & CSV summaries in the same folder.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})

# ============================================================
# SETTINGS & DATA LOADING
# ============================================================

RANDOM_STATE = 42
TARGET_COL = "critical_temp"
FOLDER_PATH = Path(__file__).resolve().parent
DATA_PATH = FOLDER_PATH / "train.csv"

print("============================================================")
print("ENSEMBLE MODEL EXPLAINABILITY & INTERPRETABILITY")
print("============================================================")
print(f"Loading dataset from: {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nERROR: train.csv not found at {DATA_PATH}.\n"
        "Ensure train.csv is in the Ensembling folder."
    )

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# Clean non-numeric columns and missing target values
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    df = df.drop(columns=non_numeric_cols)

df = df.dropna(subset=[TARGET_COL])

X = df.drop(columns=[TARGET_COL]).astype(float)
y = df[TARGET_COL].astype(float)

feature_names = X.columns.tolist()
print(f"Features count: {len(feature_names)}")

# 70% Train, 15% Validation, 15% Test split (matching Ensemble.py)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
)

# Retrain set (85%) matching final Ensemble evaluation
X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0)

print(f"Train + Validation set size: {X_trainval.shape}")
print(f"Test set size:                {X_test.shape}")

# ============================================================
# TRAIN MODELS
# ============================================================

print("\n" + "="*60)
print("TRAINING ENSEMBLE MODELS FOR EXPLAINABILITY")
print("="*60)

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

trained_models = {}
for name, model in models.items():
    print(f"Fitting {name}...")
    model.fit(X_trainval, y_trainval)
    trained_models[name] = model

# ============================================================
# 1. BUILT-IN FEATURE IMPORTANCE ANALYSIS
# ============================================================

print("\n" + "="*60)
print("1. BUILT-IN FEATURE IMPORTANCES")
print("="*60)

importance_dict = {}
for name, model in trained_models.items():
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        # Normalize to percentage
        imps = (imps / np.sum(imps)) * 100
        importance_dict[name] = imps

imp_df = pd.DataFrame(importance_dict, index=feature_names)
imp_df["Average_Importance"] = imp_df.mean(axis=1)
imp_df = imp_df.sort_values(by="Average_Importance", ascending=False)

print("\nTop 10 Most Important Features Across Ensembles (Built-in Importances):")
print(imp_df.head(10).round(3).to_string())

# Save built-in importance table
builtin_csv_path = FOLDER_PATH / "explainability_builtin_feature_importances.csv"
imp_df.to_csv(builtin_csv_path)
print(f"\nSaved built-in feature importances to: {builtin_csv_path}")

# Plot Top 15 Built-in Feature Importances
plt.figure(figsize=(12, 8))
top15 = imp_df.head(15).drop(columns=["Average_Importance"])
top15.plot(kind="barh", figsize=(12, 7))
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances Across Ensemble Models (%)", fontsize=14, fontweight='bold')
plt.xlabel("Normalized Importance (%)")
plt.ylabel("Feature")
plt.legend(title="Model")
plt.tight_layout()
builtin_plot_path = FOLDER_PATH / "explainability_feature_importances.png"
plt.savefig(builtin_plot_path, dpi=300)
plt.close()
print(f"Saved feature importance plot to: {builtin_plot_path}")

# ============================================================
# 2. PERMUTATION FEATURE IMPORTANCE (ON TEST SET)
# ============================================================

print("\n" + "="*60)
print("2. PERMUTATION FEATURE IMPORTANCE (MODEL-AGNOSTIC ON TEST SET)")
print("="*60)

perm_importance_dict = {}
for name, model in trained_models.items():
    print(f"Computing Permutation Importance for {name} on Test set...")
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    perm_importance_dict[f"{name}_Permutation_Mean"] = result.importances_mean

perm_df = pd.DataFrame(perm_importance_dict, index=feature_names)
perm_df["Overall_Permutation_Mean"] = perm_df.mean(axis=1)
perm_df = perm_df.sort_values(by="Overall_Permutation_Mean", ascending=False)

print("\nTop 10 Features by Test Permutation Importance (Drop in R² / MSE Impact):")
print(perm_df.head(10).round(4).to_string())

perm_csv_path = FOLDER_PATH / "explainability_permutation_importances.csv"
perm_df.to_csv(perm_csv_path)
print(f"\nSaved permutation importances to: {perm_csv_path}")

# ============================================================
# 3. SHAP (SHapley Additive exPlanations) ANALYSIS
# ============================================================

print("\n" + "="*60)
print("3. SHAP (SHapley Additive exPlanations) ANALYSIS")
print("="*60)

try:
    import shap

    # Sample test set for faster SHAP calculation if dataset is large
    sample_size = min(500, len(X_test))
    X_shap = X_test.sample(sample_size, random_state=RANDOM_STATE)

    print(f"Calculating SHAP values for XGBoost on {sample_size} test samples...")
    xgb_explainer = shap.TreeExplainer(trained_models["XGBoost"])
    xgb_shap_values = xgb_explainer(X_shap)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(xgb_shap_values, X_shap, show=False)
    plt.title("XGBoost SHAP Summary Plot (Feature Impact on Critical Temperature)", fontsize=13, fontweight='bold')
    shap_plot_path = FOLDER_PATH / "explainability_shap_summary_xgb.png"
    plt.savefig(shap_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved XGBoost SHAP summary plot to: {shap_plot_path}")

    # Bar plot for top SHAP features
    plt.figure(figsize=(10, 6))
    shap.plots.bar(xgb_shap_values, show=False, max_display=15)
    plt.title("XGBoost Top 15 Features by Mean |SHAP Value|", fontsize=13, fontweight='bold')
    shap_bar_path = FOLDER_PATH / "explainability_shap_bar_xgb.png"
    plt.savefig(shap_bar_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved XGBoost SHAP bar plot to: {shap_bar_path}")

    print(f"Calculating SHAP values for CatBoost on {sample_size} test samples...")
    cb_explainer = shap.TreeExplainer(trained_models["CatBoost"])
    cb_shap_values = cb_explainer(X_shap)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(cb_shap_values, X_shap, show=False)
    plt.title("CatBoost SHAP Summary Plot (Feature Impact on Critical Temperature)", fontsize=13, fontweight='bold')
    cb_shap_plot_path = FOLDER_PATH / "explainability_shap_summary_catboost.png"
    plt.savefig(cb_shap_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved CatBoost SHAP summary plot to: {cb_shap_plot_path}")

except ImportError:
    print("Notice: 'shap' package is not installed. To generate SHAP waterfall/summary plots, run: pip install shap")
except Exception as e:
    print(f"Warning: SHAP evaluation encountered an issue: {e}")

# ============================================================
# 4. PARTIAL DEPENDENCE PLOTS (PDP)
# ============================================================

print("\n" + "="*60)
print("4. PARTIAL DEPENDENCE ANALYSIS FOR TOP FEATURES")
print("="*60)

top_features = imp_df.head(4).index.tolist()
print(f"Generating Partial Dependence Plots for top 4 driver features: {top_features}")

fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(
    trained_models["XGBoost"],
    X_trainval,
    top_features,
    grid_resolution=30,
    ax=ax
)
plt.suptitle("Partial Dependence Plots (XGBoost) for Top Driver Features", fontsize=14, fontweight='bold')
plt.tight_layout()
pdp_plot_path = FOLDER_PATH / "explainability_partial_dependence.png"
plt.savefig(pdp_plot_path, dpi=300)
plt.close()
print(f"Saved Partial Dependence plot to: {pdp_plot_path}")

# ============================================================
# 5. LOCAL INSTANCE-LEVEL EXPLANATIONS
# ============================================================

print("\n" + "="*60)
print("5. LOCAL INSTANCE-LEVEL SAMPLE EXPLANATIONS")
print("="*60)

# Evaluate predictions on test set
best_model = trained_models["XGBoost"]
y_test_pred = best_model.predict(X_test)

high_tc_idx = np.argmax(y_test_pred)
low_tc_idx = np.argmin(y_test_pred)

print("\nSample 1: Highest Predicted Critical Temperature Superconductor")
print(f"  True Tc:      {y_test.iloc[high_tc_idx]:.2f} K")
print(f"  Predicted Tc: {y_test_pred[high_tc_idx]:.2f} K")
print("  Top Key Feature Values:")
for feat in top_features:
    print(f"    - {feat}: {X_test.iloc[high_tc_idx][feat]:.4f}")

print("\nSample 2: Lowest Predicted Critical Temperature Superconductor")
print(f"  True Tc:      {y_test.iloc[low_tc_idx]:.2f} K")
print(f"  Predicted Tc: {y_test_pred[low_tc_idx]:.2f} K")
print("  Top Key Feature Values:")
for feat in top_features:
    print(f"    - {feat}: {X_test.iloc[low_tc_idx][feat]:.4f}")

# Overall Summary File
summary_text = f"""EXPLAINABILITY REPORT SUMMARY
============================================================
Target Variable: critical_temp (Superconductor Critical Temp in Kelvin)
Models Analyzed: XGBoost, CatBoost, AdaBoost

Top 5 Driver Features across all Ensembles:
{imp_df.head(5)[['Average_Importance']].to_string()}

Generated Explainability Artifacts:
1. {builtin_csv_path.name}
2. {builtin_plot_path.name}
3. {perm_csv_path.name}
4. {pdp_plot_path.name}
============================================================
"""

summary_txt_path = FOLDER_PATH / "explainability_summary_report.txt"
with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print("\n" + "="*60)
print("EXPLAINABILITY ANALYSIS COMPLETE")
print("============================================================")
print(f"Summary report written to: {summary_txt_path}")
