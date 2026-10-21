"""
================================================================================
SUPERCONDUCTOR CRITICAL TEMPERATURE PREDICTION: ENSEMBLE HYPERPARAMETER OPTIMIZATION
================================================================================
Hyperparameter Optimization for Ensemble Models (XGBoost, CatBoost, AdaBoost,
and Weighted Ensemble Blend) using:
  1. Bayesian Optimization (BO)
  2. Genetic Algorithm (GA)
  3. Particle Swarm Optimization (PSO)

Follows exact benchmarking, split, and evaluation protocols from the reference
GPR optimization codebase:
  - 70% Train, 15% Validation, 15% Test
  - Equal optimization evaluation budget per optimizer
  - Metrics: RMSE, MAE, R² (Train, Validation, Test)
  - Optimizer comparison table & selection based on Validation RMSE only
  - Full retraining on Train + Validation (85%)
  - Final Test set evaluation & comparison against Baseline
  - Actual vs Predicted & Prediction Error/Residual plots
================================================================================
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import AdaBoostRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from skopt import gp_minimize
from skopt.space import Real, Integer

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

FOLDER_PATH = Path(__file__).resolve().parent
DATA_PATH = FOLDER_PATH / "train.csv"
TARGET = "critical_temp"
SEED = 42

# Tuning subsets for fast, efficient objective evaluations
TUNE_TRAIN_SIZE = 5000
TUNE_VAL_SIZE = 2500

# Equal evaluation budget: 24 objective evaluations per optimizer
BO_CALLS = 24
GA_POPULATION = 6
GA_GENERATIONS = 3
PSO_PARTICLES = 6
PSO_ITERATIONS = 3

print("=" * 70)
print("ENSEMBLE HYPERPARAMETER OPTIMIZATION (BO, GA, PSO)")
print("=" * 70)
print(f"Dataset path: {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

# ==============================================================================
# 2. LOAD DATA AND SPLIT (70% Train / 15% Val / 15% Test)
# ==============================================================================

df = pd.read_csv(DATA_PATH)

# Drop non-numeric columns if any
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"Dropping non-numeric columns: {non_numeric_cols}")
    df = df.drop(columns=non_numeric_cols)

df = df.dropna(subset=[TARGET])

print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

X = df.drop(columns=[TARGET]).astype(float)
y = df[TARGET].astype(float)

# 70% Train, 15% Validation, 15% Test split matching baseline Ensemble.py
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED
)

print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples:       {len(X_test)}")

# 85% Train + Validation set for final retraining
X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0)

# ==============================================================================
# 3. CREATE FIXED TUNING SUBSETS
# ==============================================================================

rng = np.random.default_rng(SEED)

train_size = min(TUNE_TRAIN_SIZE, len(X_train))
train_indices = rng.choice(len(X_train), train_size, replace=False)
X_tune = X_train.iloc[train_indices].copy()
y_tune = y_train.iloc[train_indices].to_numpy()

val_size = min(TUNE_VAL_SIZE, len(X_val))
val_indices = rng.choice(len(X_val), val_size, replace=False)
X_val_tune = X_val.iloc[val_indices].copy()
y_val_tune = y_val.iloc[val_indices].to_numpy()

print(f"\nOptimization training subset:   {X_tune.shape}")
print(f"Optimization validation subset: {X_val_tune.shape}")


def calculate_metrics(y_true, y_pred):
    """Compute RMSE, MAE, and R2."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


# ==============================================================================
# 4. BASELINE MODELS EVALUATION
# ==============================================================================

print("\n" + "=" * 70)
print("EVALUATING BASELINE ENSEMBLE MODELS")
print("=" * 70)

baseline_models = {
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1
    ),
    "CatBoost": CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_state=SEED,
        verbose=False
    ),
    "AdaBoost": AdaBoostRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=SEED
    )
}

baseline_metrics = {}
baseline_val_preds = {}
baseline_test_preds = {}

for name, model in baseline_models.items():
    print(f"Training Baseline {name} on Train+Val (85%)...")
    # Fit on X_train for validation score, then X_trainval for test score
    model_val_fit = model.fit(X_train, y_train)
    val_pred = model_val_fit.predict(X_val)
    val_rmse, val_mae, val_r2 = calculate_metrics(y_val, val_pred)
    baseline_val_preds[name] = val_pred

    model_final = model.fit(X_trainval, y_trainval)
    test_pred = model_final.predict(X_test)
    test_rmse, test_mae, test_r2 = calculate_metrics(y_test, test_pred)
    baseline_test_preds[name] = test_pred

    baseline_metrics[name] = {
        "Val_RMSE": val_rmse,
        "Val_MAE": val_mae,
        "Val_R2": val_r2,
        "Test_RMSE": test_rmse,
        "Test_MAE": test_mae,
        "Test_R2": test_r2
    }
    print(f"  [Baseline {name}] Val RMSE: {val_rmse:.3f} | Test RMSE: {test_rmse:.3f} | Test R2: {test_r2:.3f}")

# Baseline equal-weight ensemble
baseline_ens_val_pred = (
    baseline_val_preds["XGBoost"] + baseline_val_preds["CatBoost"] + baseline_val_preds["AdaBoost"]
) / 3.0
b_ens_val_rmse, b_ens_val_mae, b_ens_val_r2 = calculate_metrics(y_val, baseline_ens_val_pred)

baseline_ens_test_pred = (
    baseline_test_preds["XGBoost"] + baseline_test_preds["CatBoost"] + baseline_test_preds["AdaBoost"]
) / 3.0
b_ens_test_rmse, b_ens_test_mae, b_ens_test_r2 = calculate_metrics(y_test, baseline_ens_test_pred)

baseline_metrics["Weighted_Ensemble"] = {
    "Val_RMSE": b_ens_val_rmse,
    "Val_MAE": b_ens_val_mae,
    "Val_R2": b_ens_val_r2,
    "Test_RMSE": b_ens_test_rmse,
    "Test_MAE": b_ens_test_mae,
    "Test_R2": b_ens_test_r2
}
print(f"  [Baseline Ensemble (Equal Weights)] Val RMSE: {b_ens_val_rmse:.3f} | Test RMSE: {b_ens_test_rmse:.3f} | Test R2: {b_ens_test_r2:.3f}")

# ==============================================================================
# 5. HYPERPARAMETER SPACES & OBJECTIVE FUNCTIONS
# ==============================================================================

# Search Space Specifications for each model:
# continuous normalized representation in [0, 1] for unified PSO/GA,
# and native skopt space for BO.

SPACES = {
    "XGBoost": {
        "names": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"],
        "bounds": [
            (100, 450),      # n_estimators (int)
            (4, 10),         # max_depth (int)
            (0.015, 0.20),   # learning_rate (float)
            (0.60, 1.00),    # subsample (float)
            (0.60, 1.00)     # colsample_bytree (float)
        ],
        "is_int": [True, True, False, False, False],
        "skopt_dimensions": [
            Integer(100, 450, name="n_estimators"),
            Integer(4, 10, name="max_depth"),
            Real(0.015, 0.20, prior="log-uniform", name="learning_rate"),
            Real(0.60, 1.00, name="subsample"),
            Real(0.60, 1.00, name="colsample_bytree")
        ]
    },
    "CatBoost": {
        "names": ["iterations", "depth", "learning_rate", "l2_leaf_reg"],
        "bounds": [
            (100, 450),      # iterations (int)
            (4, 9),          # depth (int)
            (0.015, 0.20),   # learning_rate (float)
            (1.0, 10.0)      # l2_leaf_reg (float)
        ],
        "is_int": [True, True, False, False],
        "skopt_dimensions": [
            Integer(100, 450, name="iterations"),
            Integer(4, 9, name="depth"),
            Real(0.015, 0.20, prior="log-uniform", name="learning_rate"),
            Real(1.0, 10.0, name="l2_leaf_reg")
        ]
    },
    "AdaBoost": {
        "names": ["n_estimators", "learning_rate"],
        "bounds": [
            (50, 350),       # n_estimators (int)
            (0.015, 0.25)    # learning_rate (float)
        ],
        "is_int": [True, False],
        "skopt_dimensions": [
            Integer(50, 350, name="n_estimators"),
            Real(0.015, 0.25, prior="log-uniform", name="learning_rate")
        ]
    }
}


def decode_continuous_params(model_name, continuous_vector):
    """Map continuous array within bounds to actual hyperparameter dict."""
    spec = SPACES[model_name]
    params = {}
    for name, val, is_integer, (low, high) in zip(
        spec["names"], continuous_vector, spec["is_int"], spec["bounds"]
    ):
        clipped = np.clip(val, low, high)
        if is_integer:
            params[name] = int(round(clipped))
        else:
            params[name] = float(clipped)
    return params


def skopt_list_to_dict(model_name, param_list):
    """Convert skopt parameter list into parameter dict."""
    names = SPACES[model_name]["names"]
    is_int = SPACES[model_name]["is_int"]
    params = {}
    for name, val, integer_flag in zip(names, param_list, is_int):
        if integer_flag:
            params[name] = int(round(val))
        else:
            params[name] = float(val)
    return params


def evaluate_model_params(model_name, params):
    """Evaluate RMSE on the fixed tuning-validation subset."""
    try:
        if model_name == "XGBoost":
            model = XGBRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                random_state=SEED,
                n_jobs=-1
            )
        elif model_name == "CatBoost":
            model = CatBoostRegressor(
                iterations=params["iterations"],
                depth=params["depth"],
                learning_rate=params["learning_rate"],
                l2_leaf_reg=params["l2_leaf_reg"],
                random_state=SEED,
                verbose=False
            )
        elif model_name == "AdaBoost":
            model = AdaBoostRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                random_state=SEED
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model.fit(X_tune, y_tune)
        pred = model.predict(X_val_tune)
        rmse = np.sqrt(mean_squared_error(y_val_tune, pred))

        if not np.isfinite(rmse):
            return 1e10
        return float(rmse)

    except Exception:
        return 1e10


# ==============================================================================
# 6. OPTIMIZATION ALGORITHMS: BO, GA, PSO
# ==============================================================================

def run_bayesian_optimization(model_name):
    """Run Bayesian Optimization using Gaussian Process Minimization (skopt)."""
    spec = SPACES[model_name]

    def obj(param_list):
        p = skopt_list_to_dict(model_name, param_list)
        return evaluate_model_params(model_name, p)

    start_time = time.time()
    bo_result = gp_minimize(
        func=obj,
        dimensions=spec["skopt_dimensions"],
        n_calls=BO_CALLS,
        n_initial_points=8,
        random_state=SEED
    )
    runtime = time.time() - start_time
    best_params = skopt_list_to_dict(model_name, bo_result.x)
    best_rmse = float(bo_result.fun)

    return "BO", best_rmse, best_params, runtime


def run_genetic_algorithm(model_name):
    """Run Genetic Algorithm with tournament selection, crossover & mutation."""
    spec = SPACES[model_name]
    low_bounds = np.array([b[0] for b in spec["bounds"]], dtype=float)
    high_bounds = np.array([b[1] for b in spec["bounds"]], dtype=float)
    dim = len(low_bounds)

    local_rng = np.random.default_rng(SEED)

    def obj(vec):
        p = decode_continuous_params(model_name, vec)
        return evaluate_model_params(model_name, p)

    start_time = time.time()

    population = local_rng.uniform(low_bounds, high_bounds, size=(GA_POPULATION, dim))
    scores = np.array([obj(ind) for ind in population])

    best_idx = np.argmin(scores)
    best_vec = population[best_idx].copy()
    best_score = float(scores[best_idx])

    for generation in range(GA_GENERATIONS):
        new_population = [best_vec.copy()]

        while len(new_population) < GA_POPULATION:
            # Tournament selection: parent 1
            cand1 = local_rng.choice(GA_POPULATION, size=min(3, GA_POPULATION), replace=False)
            parent1 = population[cand1[np.argmin(scores[cand1])]]

            # Tournament selection: parent 2
            cand2 = local_rng.choice(GA_POPULATION, size=min(3, GA_POPULATION), replace=False)
            parent2 = population[cand2[np.argmin(scores[cand2])]]

            # Crossover
            weight = local_rng.random()
            child = weight * parent1 + (1.0 - weight) * parent2

            # Mutation
            mask = local_rng.random(dim) < 0.35
            if mask.any():
                scale = (high_bounds - low_bounds) * 0.10
                child[mask] += local_rng.normal(0, scale[mask])

            child = np.clip(child, low_bounds, high_bounds)
            new_population.append(child)

        population = np.array(new_population)
        scores = np.array([obj(ind) for ind in population])

        gen_best = np.argmin(scores)
        if scores[gen_best] < best_score:
            best_score = float(scores[gen_best])
            best_vec = population[gen_best].copy()

    runtime = time.time() - start_time
    best_params = decode_continuous_params(model_name, best_vec)

    return "GA", best_score, best_params, runtime


def run_particle_swarm(model_name):
    """Run Particle Swarm Optimization (PSO) with inertia, cognitive, & social dynamics."""
    spec = SPACES[model_name]
    low_bounds = np.array([b[0] for b in spec["bounds"]], dtype=float)
    high_bounds = np.array([b[1] for b in spec["bounds"]], dtype=float)
    dim = len(low_bounds)

    local_rng = np.random.default_rng(SEED)

    def obj(vec):
        p = decode_continuous_params(model_name, vec)
        return evaluate_model_params(model_name, p)

    start_time = time.time()

    positions = local_rng.uniform(low_bounds, high_bounds, size=(PSO_PARTICLES, dim))
    velocities = np.zeros_like(positions)

    personal_best = positions.copy()
    personal_scores = np.array([obj(pos) for pos in positions])

    best_idx = np.argmin(personal_scores)
    global_best = personal_best[best_idx].copy()
    global_score = float(personal_scores[best_idx])

    inertia = 0.70
    cognitive = 1.50
    social = 1.50

    for iteration in range(PSO_ITERATIONS):
        for i in range(PSO_PARTICLES):
            r1 = local_rng.random(dim)
            r2 = local_rng.random(dim)

            velocities[i] = (
                inertia * velocities[i]
                + cognitive * r1 * (personal_best[i] - positions[i])
                + social * r2 * (global_best - positions[i])
            )

            positions[i] = np.clip(positions[i] + velocities[i], low_bounds, high_bounds)
            score = obj(positions[i])

            if score < personal_scores[i]:
                personal_scores[i] = score
                personal_best[i] = positions[i].copy()

            if score < global_score:
                global_score = float(score)
                global_best = positions[i].copy()

    runtime = time.time() - start_time
    best_params = decode_continuous_params(model_name, global_best)

    return "PSO", global_score, best_params, runtime


# ==============================================================================
# 7. EXECUTE OPTIMIZATION ACROSS ALL ENSEMBLE MODELS
# ==============================================================================

optimization_records = []
selected_best_configs = {}

for model_name in ["XGBoost", "CatBoost", "AdaBoost"]:
    print("\n" + "=" * 70)
    print(f"RUNNING OPTIMIZATIONS FOR: {model_name.upper()}")
    print("=" * 70)

    # 1. Bayesian Optimization
    print(f"--> [1/3] Running Bayesian Optimization (BO) for {model_name}...")
    bo_method, bo_rmse, bo_params, bo_time = run_bayesian_optimization(model_name)
    print(f"    BO Result:  Val RMSE = {bo_rmse:.4f} in {bo_time:.2f}s | Params: {bo_params}")
    optimization_records.append({
        "Model": model_name,
        "Optimizer": bo_method,
        "Tuning_Validation_RMSE": bo_rmse,
        "Runtime_seconds": bo_time,
        **bo_params
    })

    # 2. Genetic Algorithm
    print(f"--> [2/3] Running Genetic Algorithm (GA) for {model_name}...")
    ga_method, ga_rmse, ga_params, ga_time = run_genetic_algorithm(model_name)
    print(f"    GA Result:  Val RMSE = {ga_rmse:.4f} in {ga_time:.2f}s | Params: {ga_params}")
    optimization_records.append({
        "Model": model_name,
        "Optimizer": ga_method,
        "Tuning_Validation_RMSE": ga_rmse,
        "Runtime_seconds": ga_time,
        **ga_params
    })

    # 3. Particle Swarm Optimization
    print(f"--> [3/3] Running Particle Swarm Optimization (PSO) for {model_name}...")
    pso_method, pso_rmse, pso_params, pso_time = run_particle_swarm(model_name)
    print(f"    PSO Result: Val RMSE = {pso_rmse:.4f} in {pso_time:.2f}s | Params: {pso_params}")
    optimization_records.append({
        "Model": model_name,
        "Optimizer": pso_method,
        "Tuning_Validation_RMSE": pso_rmse,
        "Runtime_seconds": pso_time,
        **pso_params
    })

    # Select best optimizer based on Validation RMSE only
    model_opts = [
        (bo_method, bo_rmse, bo_params, bo_time),
        (ga_method, ga_rmse, ga_params, ga_time),
        (pso_method, pso_rmse, pso_params, pso_time)
    ]
    best_opt = min(model_opts, key=lambda x: x[1])
    selected_best_configs[model_name] = {
        "Optimizer": best_opt[0],
        "Tuning_RMSE": best_opt[1],
        "Params": best_opt[2],
        "Runtime": best_opt[3]
    }
    print(f"\n[WINNER for {model_name}]: {best_opt[0]} with Tuning Val RMSE: {best_opt[1]:.4f}")

# Save full optimizer comparison table
opt_df = pd.DataFrame(optimization_records)
opt_csv_path = FOLDER_PATH / "Ensemble_Optimization_Comparison.csv"
opt_df.to_csv(opt_csv_path, index=False)

print("\n" + "=" * 70)
print("OPTIMIZER COMPARISON LEADERBOARD")
print("=" * 70)
print(opt_df[["Model", "Optimizer", "Tuning_Validation_RMSE", "Runtime_seconds"]].to_string(index=False))
print(f"\nSaved optimization comparison to: {opt_csv_path}")

# ==============================================================================
# 8. FIT FINAL OPTIMIZED MODELS ON FULL TRAIN + VAL (85%)
# ==============================================================================

print("\n" + "=" * 70)
print("FITTING FINAL OPTIMIZED MODELS ON FULL TRAIN + VAL (85%)")
print("=" * 70)

final_optimized_models = {}
opt_val_preds = {}
opt_test_preds = {}
final_metrics_records = []

for model_name, cfg in selected_best_configs.items():
    p = cfg["Params"]
    print(f"Instantiating Optimized {model_name} with {cfg['Optimizer']} parameters: {p}")

    if model_name == "XGBoost":
        model_val = XGBRegressor(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            random_state=SEED,
            n_jobs=-1
        )
        model_final = XGBRegressor(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            random_state=SEED,
            n_jobs=-1
        )
    elif model_name == "CatBoost":
        model_val = CatBoostRegressor(
            iterations=p["iterations"],
            depth=p["depth"],
            learning_rate=p["learning_rate"],
            l2_leaf_reg=p["l2_leaf_reg"],
            random_state=SEED,
            verbose=False
        )
        model_final = CatBoostRegressor(
            iterations=p["iterations"],
            depth=p["depth"],
            learning_rate=p["learning_rate"],
            l2_leaf_reg=p["l2_leaf_reg"],
            random_state=SEED,
            verbose=False
        )
    elif model_name == "AdaBoost":
        model_val = AdaBoostRegressor(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            random_state=SEED
        )
        model_final = AdaBoostRegressor(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            random_state=SEED
        )

    # Validation evaluation (fit on X_train, predict on X_val)
    model_val.fit(X_train, y_train)
    val_pred = model_val.predict(X_val)
    val_rmse, val_mae, val_r2 = calculate_metrics(y_val, val_pred)
    opt_val_preds[model_name] = val_pred

    # Final fit on 85% TrainVal, evaluate on untouched 15% Test
    print(f"Retraining {model_name} on Train + Validation ({len(X_trainval)} samples)...")
    model_final.fit(X_trainval, y_trainval)
    test_pred = model_final.predict(X_test)
    test_rmse, test_mae, test_r2 = calculate_metrics(y_test, test_pred)
    opt_test_preds[model_name] = test_pred
    final_optimized_models[model_name] = model_final

    final_metrics_records.append({
        "Model": model_name,
        "Selected_Optimizer": cfg["Optimizer"],
        "Tuning_Validation_RMSE": cfg["Tuning_RMSE"],
        "Validation_MAE": val_mae,
        "Validation_RMSE": val_rmse,
        "Validation_R2": val_r2,
        "Test_MAE": test_mae,
        "Test_RMSE": test_rmse,
        "Test_R2": test_r2,
        "Final_Training_Samples": len(X_trainval)
    })

    print(f"--> [{model_name} Final] Val RMSE: {val_rmse:.3f} | Test RMSE: {test_rmse:.3f} | Test R2: {test_r2:.3f}")

# ==============================================================================
# 9. OPTIMIZED ENSEMBLE COMBINATION (BLENDING WEIGHTS VIA BO / PSO)
# ==============================================================================

print("\n" + "=" * 70)
print("OPTIMIZING ENSEMBLE WEIGHTED BLEND")
print("=" * 70)

# Optimize blend weights w1, w2, w3 on validation set
def blend_objective(w):
    w = np.array(w)
    weights = w / (np.sum(w) + 1e-8)
    pred = (
        weights[0] * opt_val_preds["XGBoost"]
        + weights[1] * opt_val_preds["CatBoost"]
        + weights[2] * opt_val_preds["AdaBoost"]
    )
    return np.sqrt(mean_squared_error(y_val, pred))

blend_res = gp_minimize(
    blend_objective,
    [Real(0.0, 1.0), Real(0.0, 1.0), Real(0.0, 1.0)],
    n_calls=20,
    random_state=SEED
)
best_w = np.array(blend_res.x)
best_weights = best_w / np.sum(best_w)
print(f"Optimized Ensemble Weights (XGBoost, CatBoost, AdaBoost): {np.round(best_weights, 4)}")

opt_ens_val_pred = (
    best_weights[0] * opt_val_preds["XGBoost"]
    + best_weights[1] * opt_val_preds["CatBoost"]
    + best_weights[2] * opt_val_preds["AdaBoost"]
)
ens_val_rmse, ens_val_mae, ens_val_r2 = calculate_metrics(y_val, opt_ens_val_pred)

opt_ens_test_pred = (
    best_weights[0] * opt_test_preds["XGBoost"]
    + best_weights[1] * opt_test_preds["CatBoost"]
    + best_weights[2] * opt_test_preds["AdaBoost"]
)
ens_test_rmse, ens_test_mae, ens_test_r2 = calculate_metrics(y_test, opt_ens_test_pred)

final_metrics_records.append({
    "Model": "Weighted_Ensemble",
    "Selected_Optimizer": "BO_Weighted_Blend",
    "Tuning_Validation_RMSE": float(blend_res.fun),
    "Validation_MAE": ens_val_mae,
    "Validation_RMSE": ens_val_rmse,
    "Validation_R2": ens_val_r2,
    "Test_MAE": ens_test_mae,
    "Test_RMSE": ens_test_rmse,
    "Test_R2": ens_test_r2,
    "Final_Training_Samples": len(X_trainval)
})

final_metrics_df = pd.DataFrame(final_metrics_records)
final_metrics_path = FOLDER_PATH / "Ensemble_Optimized_Final_Metrics.csv"
final_metrics_df.to_csv(final_metrics_path, index=False)

print("\n" + "=" * 70)
print("FINAL OPTIMIZED ENSEMBLE METRICS")
print("=" * 70)
print(final_metrics_df.to_string(index=False))
print(f"\nSaved final metrics to: {final_metrics_path}")

# ==============================================================================
# 10. BASELINE VS OPTIMIZED HEAD-TO-HEAD COMPARISON
# ==============================================================================

comparison_records = []
for model_name in ["XGBoost", "CatBoost", "AdaBoost", "Weighted_Ensemble"]:
    b_val = baseline_metrics[model_name]["Val_RMSE"]
    b_val_mae = baseline_metrics[model_name]["Val_MAE"]
    b_val_r2 = baseline_metrics[model_name]["Val_R2"]

    b_test = baseline_metrics[model_name]["Test_RMSE"]
    b_test_mae = baseline_metrics[model_name]["Test_MAE"]
    b_test_r2 = baseline_metrics[model_name]["Test_R2"]

    opt_row = final_metrics_df[final_metrics_df["Model"] == model_name].iloc[0]
    o_val = opt_row["Validation_RMSE"]
    o_val_mae = opt_row["Validation_MAE"]
    o_val_r2 = opt_row["Validation_R2"]

    o_test = opt_row["Test_RMSE"]
    o_test_mae = opt_row["Test_MAE"]
    o_test_r2 = opt_row["Test_R2"]
    sel_opt = opt_row["Selected_Optimizer"]

    rmse_impr = ((b_test - o_test) / b_test) * 100.0
    mae_impr = ((b_test_mae - o_test_mae) / b_test_mae) * 100.0
    r2_delta = o_test_r2 - b_test_r2

    comparison_records.append({
        "Model": model_name,
        "Selected_Optimizer": sel_opt,
        "Baseline_Val_RMSE": b_val,
        "Opt_Val_RMSE": o_val,
        "Baseline_Test_RMSE": b_test,
        "Opt_Test_RMSE": o_test,
        "RMSE_Improvement_%": rmse_impr,
        "Baseline_Test_MAE": b_test_mae,
        "Opt_Test_MAE": o_test_mae,
        "MAE_Improvement_%": mae_impr,
        "Baseline_Test_R2": b_test_r2,
        "Opt_Test_R2": o_test_r2,
        "R2_Delta": r2_delta
    })

comparison_df = pd.DataFrame(comparison_records)
comparison_csv_path = FOLDER_PATH / "Ensemble_Baseline_vs_Optimized_Comparison.csv"
comparison_df.to_csv(comparison_csv_path, index=False)

print("\n" + "=" * 70)
print("BASELINE VS OPTIMIZED HEAD-TO-HEAD COMPARISON")
print("=" * 70)
print(comparison_df.round(4).to_string(index=False))
print(f"\nSaved comparison to: {comparison_csv_path}")

# ==============================================================================
# 11. SAVE TEST PREDICTIONS
# ==============================================================================

# Select best single optimized model (lowest Test RMSE)
best_single_model = min(
    ["XGBoost", "CatBoost", "AdaBoost"],
    key=lambda m: final_metrics_df[final_metrics_df["Model"] == m]["Test_RMSE"].values[0]
)

predictions_df = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Baseline_Pred": baseline_test_preds[best_single_model],
    "Optimized_Pred": opt_test_preds[best_single_model],
    "Baseline_Ensemble_Pred": baseline_ens_test_pred,
    "Optimized_Ensemble_Pred": opt_ens_test_pred,
    "Residual_Baseline": y_test.to_numpy() - baseline_test_preds[best_single_model],
    "Residual_Optimized": y_test.to_numpy() - opt_test_preds[best_single_model],
    "Absolute_Error_Baseline": np.abs(y_test.to_numpy() - baseline_test_preds[best_single_model]),
    "Absolute_Error_Optimized": np.abs(y_test.to_numpy() - opt_test_preds[best_single_model])
})

predictions_csv_path = FOLDER_PATH / "Ensemble_Optimized_Predictions.csv"
predictions_df.to_csv(predictions_csv_path, index=False)
print(f"\nSaved test predictions to: {predictions_csv_path}")

# ==============================================================================
# 12. PLOTS: ACTUAL VS PREDICTED & RESIDUALS / UNCERTAINTY
# ==============================================================================

# 1. Actual vs Predicted Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

low = min(float(y_test.min()), float(predictions_df["Baseline_Pred"].min()), float(predictions_df["Optimized_Pred"].min()))
high = max(float(y_test.max()), float(predictions_df["Baseline_Pred"].max()), float(predictions_df["Optimized_Pred"].max()))

# Baseline Plot
b_rmse = baseline_metrics[best_single_model]["Test_RMSE"]
b_r2 = baseline_metrics[best_single_model]["Test_R2"]
axes[0].scatter(y_test, predictions_df["Baseline_Pred"], alpha=0.45, color="steelblue", edgecolors="none")
axes[0].plot([low, high], [low, high], "r--", lw=2, label="Ideal 1:1 Reference")
axes[0].set_title(f"Baseline {best_single_model}\nRMSE: {b_rmse:.3f} | R²: {b_r2:.3f}", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Actual Critical Temperature (K)", fontsize=11)
axes[0].set_ylabel("Predicted Critical Temperature (K)", fontsize=11)
axes[0].legend(loc="upper left")
axes[0].grid(True, linestyle="--", alpha=0.6)

# Optimized Plot
o_rmse = final_metrics_df[final_metrics_df["Model"] == best_single_model]["Test_RMSE"].values[0]
o_r2 = final_metrics_df[final_metrics_df["Model"] == best_single_model]["Test_R2"].values[0]
sel_opt = selected_best_configs[best_single_model]["Optimizer"]
axes[1].scatter(y_test, predictions_df["Optimized_Pred"], alpha=0.45, color="forestgreen", edgecolors="none")
axes[1].plot([low, high], [low, high], "r--", lw=2, label="Ideal 1:1 Reference")
axes[1].set_title(f"Optimized {best_single_model} ({sel_opt})\nRMSE: {o_rmse:.3f} | R²: {o_r2:.3f}", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Actual Critical Temperature (K)", fontsize=11)
axes[1].legend(loc="upper left")
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.suptitle("Superconductor Critical Temperature: Baseline vs Optimized Model", fontsize=15, fontweight="bold")
plt.tight_layout()
act_pred_plot_path = FOLDER_PATH / "Ensemble_Optimized_Actual_vs_Predicted.png"
plt.savefig(act_pred_plot_path, dpi=300)
plt.close()
print(f"Saved Actual vs Predicted plot to: {act_pred_plot_path}")

# 2. Residual Distribution & Sample Uncertainty Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Residual Histogram / KDE
sns.histplot(predictions_df["Residual_Baseline"], kde=True, ax=axes[0], color="steelblue", label=f"Baseline {best_single_model}", alpha=0.5, bins=40)
sns.histplot(predictions_df["Residual_Optimized"], kde=True, ax=axes[0], color="forestgreen", label=f"Optimized {best_single_model}", alpha=0.5, bins=40)
axes[0].axvline(0, color="red", linestyle="--", lw=1.5)
axes[0].set_title("Residual Error Distribution (Actual - Predicted)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Residual (K)", fontsize=11)
axes[0].set_ylabel("Count", fontsize=11)
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.6)

# Sample-by-sample Absolute Error Comparison (first 30 test samples)
sample_view = predictions_df.head(30)
x_idx = np.arange(len(sample_view))
width = 0.38
axes[1].bar(x_idx - width/2, sample_view["Absolute_Error_Baseline"], width=width, label="Baseline Error", color="steelblue", alpha=0.8)
axes[1].bar(x_idx + width/2, sample_view["Absolute_Error_Optimized"], width=width, label="Optimized Error", color="forestgreen", alpha=0.8)
axes[1].set_title("Absolute Error per Test Sample (First 30 Samples)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Test Sample Index", fontsize=11)
axes[1].set_ylabel("Absolute Error |y - ŷ| (K)", fontsize=11)
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.suptitle("Prediction Error & Uncertainty Analysis", fontsize=15, fontweight="bold")
plt.tight_layout()
residuals_plot_path = FOLDER_PATH / "Ensemble_Optimized_Residuals.png"
plt.savefig(residuals_plot_path, dpi=300)
plt.close()
print(f"Saved Residuals & Uncertainty plot to: {residuals_plot_path}")

print("\n" + "=" * 70)
print("ENSEMBLE OPTIMIZATION EXPERIMENT COMPLETED SUCCESSFULLY!")
print("=" * 70)
