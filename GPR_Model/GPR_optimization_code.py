
# ==========================================================
# SUPERCONDUCTOR CRITICAL TEMPERATURE PREDICTION
# GPR HYPERPARAMETER OPTIMIZATION USING BO, GA AND PSO
# ==========================================================

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from skopt import gp_minimize
from skopt.space import Real

warnings.filterwarnings("ignore")

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

DATA_PATH = (
    r"C:\Users\Rupa\Downloads\material_modelling_project"
    r"\main_dataset_superconductor\train.csv"
)

TARGET = "critical_temp"
SEED = 42

# Small subset for computationally expensive GPR optimization
TUNE_TRAIN_SIZE = 600
TUNE_VAL_SIZE = 2000

# Equal budget: 24 objective evaluations per optimizer
BO_CALLS = 24
GA_POPULATION = 6
GA_GENERATIONS = 3
PSO_PARTICLES = 6
PSO_ITERATIONS = 3

# Final fitting subset, matching baseline's maximum size
FINAL_TRAIN_SIZE = 5000

# ==========================================================
# 2. LOAD DATA AND SPLIT
# ==========================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("GPR HYPERPARAMETER OPTIMIZATION")
print("=" * 60)
print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

if df.isnull().sum().sum() > 0:
    raise ValueError(
        "Dataset contains missing values. Apply the same "
        "missing-value handling used by your baseline first."
    )

X = df.drop(TARGET, axis=1)
y = df[TARGET]

# Same 70/15/15 split as the baseline
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

# Fit scaling on training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# 3. CREATE FIXED TUNING SUBSETS
# ==========================================================

rng = np.random.default_rng(SEED)

train_size = min(TUNE_TRAIN_SIZE, len(X_train_scaled))
train_indices = rng.choice(
    len(X_train_scaled), train_size, replace=False
)

X_tune = X_train_scaled[train_indices]
y_tune = y_train.iloc[train_indices].to_numpy()

val_size = min(TUNE_VAL_SIZE, len(X_val_scaled))
val_indices = rng.choice(
    len(X_val_scaled), val_size, replace=False
)

X_val_tune = X_val_scaled[val_indices]
y_val_tune = y_val.iloc[val_indices].to_numpy()

print("\nOptimization training subset:", X_tune.shape)
print("Optimization validation subset:", X_val_tune.shape)

# ==========================================================
# 4. HYPERPARAMETER SEARCH SPACE
# ==========================================================

# Search in log10 space
# [constant_value, length_scale, noise_level, alpha]

LOWER = np.array([-2.0, np.log10(0.05), -3.0, -10.0])
UPPER = np.array([ 2.0, np.log10(20.0),  1.0,  -2.0])

evaluation_count = 0


def decode_params(params):
    """Convert logarithmic coordinates into actual values."""
    return {
        "constant_value": float(10 ** params[0]),
        "length_scale": float(10 ** params[1]),
        "noise_level": float(10 ** params[2]),
        "alpha": float(10 ** params[3])
    }


def objective(params):
    """Minimize RMSE on the fixed tuning-validation subset."""
    global evaluation_count
    evaluation_count += 1

    p = decode_params(params)

    kernel = (
        ConstantKernel(
            p["constant_value"],
            constant_value_bounds="fixed"
        )
        * RBF(
            length_scale=p["length_scale"],
            length_scale_bounds="fixed"
        )
        + WhiteKernel(
            noise_level=p["noise_level"],
            noise_level_bounds="fixed"
        )
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=p["alpha"],
        normalize_y=True,
        optimizer=None,
        random_state=SEED
    )

    try:
        model.fit(X_tune, y_tune)
        pred = model.predict(X_val_tune)

        rmse = np.sqrt(mean_squared_error(y_val_tune, pred))

        if not np.isfinite(rmse):
            return 1e10

        return float(rmse)

    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return 1e10


# ==========================================================
# 5. BAYESIAN OPTIMIZATION (BO)
# ==========================================================

print("\n" + "=" * 60)
print("BAYESIAN OPTIMIZATION")
print("=" * 60)

start = time.time()

bo_result = gp_minimize(
    func=objective,
    dimensions=[
        Real(-2.0, 2.0, name="log_constant"),
        Real(np.log10(0.05), np.log10(20.0),
             name="log_length_scale"),
        Real(-3.0, 1.0, name="log_noise"),
        Real(-10.0, -2.0, name="log_alpha")
    ],
    n_calls=BO_CALLS,
    n_initial_points=8,
    random_state=SEED
)

bo_time = time.time() - start
bo_params = np.array(bo_result.x)
bo_rmse = float(bo_result.fun)

print("Best BO RMSE:", round(bo_rmse, 4))
print("Best BO parameters:", decode_params(bo_params))
print("BO runtime:", round(bo_time, 2), "seconds")

# ==========================================================
# 6. GENETIC ALGORITHM (GA)
# ==========================================================

def genetic_algorithm():
    local_rng = np.random.default_rng(SEED)

    population = local_rng.uniform(
        LOWER, UPPER,
        size=(GA_POPULATION, len(LOWER))
    )

    scores = np.array([objective(ind) for ind in population])

    best_idx = np.argmin(scores)
    best_params = population[best_idx].copy()
    best_score = float(scores[best_idx])

    for generation in range(GA_GENERATIONS):

        new_population = [best_params.copy()]

        while len(new_population) < GA_POPULATION:

            # Tournament selection: parent 1
            candidates = local_rng.choice(
                GA_POPULATION, size=3, replace=False
            )
            parent1 = population[
                candidates[np.argmin(scores[candidates])]
            ]

            # Tournament selection: parent 2
            candidates = local_rng.choice(
                GA_POPULATION, size=3, replace=False
            )
            parent2 = population[
                candidates[np.argmin(scores[candidates])]
            ]

            # Crossover
            weight = local_rng.random()
            child = weight * parent1 + (1 - weight) * parent2

            # Mutation
            mask = local_rng.random(len(LOWER)) < 0.30

            if mask.any():
                child[mask] += local_rng.normal(
                    0, 0.15, size=mask.sum()
                )

            child = np.clip(child, LOWER, UPPER)
            new_population.append(child)

        population = np.array(new_population)
        scores = np.array([objective(ind) for ind in population])

        generation_best = np.argmin(scores)

        if scores[generation_best] < best_score:
            best_score = float(scores[generation_best])
            best_params = population[generation_best].copy()

        print(
            f"GA Generation {generation + 1}: "
            f"Best RMSE = {best_score:.4f}"
        )

    return best_params, best_score


print("\n" + "=" * 60)
print("GENETIC ALGORITHM")
print("=" * 60)

start = time.time()

ga_params, ga_rmse = genetic_algorithm()

ga_time = time.time() - start

print("Best GA RMSE:", round(ga_rmse, 4))
print("Best GA parameters:", decode_params(ga_params))
print("GA runtime:", round(ga_time, 2), "seconds")

# ==========================================================
# 7. PARTICLE SWARM OPTIMIZATION (PSO)
# ==========================================================

def particle_swarm():
    local_rng = np.random.default_rng(SEED)

    positions = local_rng.uniform(
        LOWER, UPPER,
        size=(PSO_PARTICLES, len(LOWER))
    )

    velocities = np.zeros_like(positions)

    personal_best = positions.copy()
    personal_scores = np.array([
        objective(position) for position in positions
    ])

    idx = np.argmin(personal_scores)
    global_best = personal_best[idx].copy()
    global_score = float(personal_scores[idx])

    inertia = 0.7
    cognitive = 1.5
    social = 1.5

    for iteration in range(PSO_ITERATIONS):

        for i in range(PSO_PARTICLES):

            r1 = local_rng.random(len(LOWER))
            r2 = local_rng.random(len(LOWER))

            velocities[i] = (
                inertia * velocities[i]
                + cognitive * r1
                * (personal_best[i] - positions[i])
                + social * r2
                * (global_best - positions[i])
            )

            positions[i] = np.clip(
                positions[i] + velocities[i],
                LOWER, UPPER
            )

            score = objective(positions[i])

            if score < personal_scores[i]:
                personal_scores[i] = score
                personal_best[i] = positions[i].copy()

            if score < global_score:
                global_score = float(score)
                global_best = positions[i].copy()

        print(
            f"PSO Iteration {iteration + 1}: "
            f"Best RMSE = {global_score:.4f}"
        )

    return global_best, global_score


print("\n" + "=" * 60)
print("PARTICLE SWARM OPTIMIZATION")
print("=" * 60)

start = time.time()

pso_params, pso_rmse = particle_swarm()

pso_time = time.time() - start

print("Best PSO RMSE:", round(pso_rmse, 4))
print("Best PSO parameters:", decode_params(pso_params))
print("PSO runtime:", round(pso_time, 2), "seconds")

# ==========================================================
# 8. COMPARE ALL THREE OPTIMIZERS
# ==========================================================

comparison = pd.DataFrame([
    {
        "Model": "GPR",
        "Method": "BO",
        "Tuning_Validation_RMSE": bo_rmse,
        "Runtime_seconds": bo_time,
        **decode_params(bo_params)
    },
    {
        "Model": "GPR",
        "Method": "GA",
        "Tuning_Validation_RMSE": ga_rmse,
        "Runtime_seconds": ga_time,
        **decode_params(ga_params)
    },
    {
        "Model": "GPR",
        "Method": "PSO",
        "Tuning_Validation_RMSE": pso_rmse,
        "Runtime_seconds": pso_time,
        **decode_params(pso_params)
    }
])

comparison.to_csv(
    "GPR_Optimization_Comparison.csv", index=False
)

print("\nOPTIMIZER COMPARISON")
print(comparison.to_string(index=False))

# Select using validation RMSE only, never test RMSE
results = [
    ("BO", bo_rmse, bo_params),
    ("GA", ga_rmse, ga_params),
    ("PSO", pso_rmse, pso_params)
]

best_method, best_tuning_rmse, best_params = min(
    results, key=lambda item: item[1]
)

best_values = decode_params(best_params)

print("\nSelected optimizer:", best_method)
print("Selected parameters:", best_values)

# ==========================================================
# 9. FIT THE SELECTED OPTIMIZED MODEL
# ==========================================================

# Use a deterministic random subset for the final fit.
# Exact GPR can be expensive at 5,000 training samples.
final_size = min(FINAL_TRAIN_SIZE, len(X_train_scaled))

final_rng = np.random.default_rng(SEED)
final_indices = final_rng.choice(
    len(X_train_scaled), final_size, replace=False
)

X_train_final = X_train_scaled[final_indices]
y_train_final = y_train.iloc[final_indices].to_numpy()

final_kernel = (
    ConstantKernel(
        best_values["constant_value"],
        constant_value_bounds="fixed"
    )
    * RBF(
        length_scale=best_values["length_scale"],
        length_scale_bounds="fixed"
    )
    + WhiteKernel(
        noise_level=best_values["noise_level"],
        noise_level_bounds="fixed"
    )
)

gpr_optimized = GaussianProcessRegressor(
    kernel=final_kernel,
    alpha=best_values["alpha"],
    normalize_y=True,
    optimizer=None,
    random_state=SEED
)

print("\nFitting final optimized GPR...")
print("Final training samples:", final_size)

gpr_optimized.fit(X_train_final, y_train_final)

# ==========================================================
# 10. VALIDATION EVALUATION
# ==========================================================

y_val_pred, y_val_std = gpr_optimized.predict(
    X_val_scaled, return_std=True
)

val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)

print("\nFULL VALIDATION RESULTS")
print("MAE :", round(val_mae, 4))
print("RMSE:", round(val_rmse, 4))
print("R²  :", round(val_r2, 4))

# ==========================================================
# 11. FINAL TEST EVALUATION
# ==========================================================

y_test_pred, y_test_std = gpr_optimized.predict(
    X_test_scaled, return_std=True
)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\nFINAL TEST RESULTS")
print("Selected optimizer:", best_method)
print("MAE :", round(test_mae, 4))
print("RMSE:", round(test_rmse, 4))
print("R²  :", round(test_r2, 4))

# ==========================================================
# 12. SAVE FINAL RESULTS
# ==========================================================

final_metrics = pd.DataFrame([{
    "Model": "GPR",
    "Selected_Optimizer": best_method,
    "Tuning_Validation_RMSE": best_tuning_rmse,
    "Validation_MAE": val_mae,
    "Validation_RMSE": val_rmse,
    "Validation_R2": val_r2,
    "Test_MAE": test_mae,
    "Test_RMSE": test_rmse,
    "Test_R2": test_r2,
    "Final_Training_Samples": final_size
}])

final_metrics.to_csv(
    "GPR_Optimized_Final_Metrics.csv", index=False
)

predictions = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_test_pred,
    "Std_Deviation": y_test_std
})

predictions.to_csv(
    "GPR_Optimized_Predictions.csv", index=False
)

print("\nSaved:")
print("GPR_Optimization_Comparison.csv")
print("GPR_Optimized_Final_Metrics.csv")
print("GPR_Optimized_Predictions.csv")

# ==========================================================
# 13. ACTUAL VS PREDICTED PLOT
# ==========================================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_test_pred, alpha=0.5)

low = min(float(y_test.min()), float(y_test_pred.min()))
high = max(float(y_test.max()), float(y_test_pred.max()))

plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual Critical Temperature")
plt.ylabel("Predicted Critical Temperature")
plt.title(f"Optimized GPR: Actual vs Predicted ({best_method})")
plt.tight_layout()
plt.savefig("GPR_Optimized_Actual_vs_Predicted.png", dpi=300)
plt.show()

# ==========================================================
# 14. PREDICTION UNCERTAINTY
# ==========================================================

uncertainty = predictions.head(20)

plt.figure(figsize=(12, 6))

plt.errorbar(
    np.arange(len(uncertainty)),
    uncertainty["Predicted"],
    yerr=uncertainty["Std_Deviation"],
    fmt="o",
    capsize=3
)

plt.xlabel("Test Sample Index")
plt.ylabel("Predicted Critical Temperature")
plt.title("Optimized GPR Prediction Uncertainty")
plt.tight_layout()
plt.savefig("GPR_Optimized_Uncertainty.png", dpi=300)
plt.show()

print("\nGPR optimization experiment completed.")
