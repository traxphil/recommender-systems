# Task 6 LightGBM gap model sweep around the stored setting (4 earlier gaps as rows, rows-from-end input, 255 leaves, 600
# trees, lr 0.02, min 100 per leaf; validation MAE 80.41 days): more earlier gaps, bigger trees, more / slower trees;
# the best setting replaces the stored model + task_6_predictions.csv only if its out-of-fold MAE is lower
# (old file -> output/before_task6_sweep/)
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np

import assignment1
import lib.config as model_configs
from lib.models.gap_regressor import load_gap_model, store_gap_model

task = assignment1.load_task6()
mae_days = lambda result: np.abs(result.valid_predictions - task.validation_gaps).mean() / 86400
best_mae = stored_mae = mae_days(load_gap_model("task6", "lightgbm_gap_model"))
print(f"RESULT task6 stored: {stored_mae:.3f} days", flush=True)
base_config = model_configs.TASK6_LIGHTGBM_GAP_MODEL
settings = {
    "6 earlier gaps": dict(n_earlier_gaps_as_training_rows=6),
    "8 earlier gaps": dict(n_earlier_gaps_as_training_rows=8),
    "511 leaves": dict(lightgbm_num_leaves=511),
    "1000 trees": dict(lightgbm_n_estimators=1000),
    "min 50 per leaf": dict(lightgbm_min_child_samples=50),
    "min 200 per leaf": dict(lightgbm_min_child_samples=200),
}
results = {}
for label, changes in settings.items():
    start = time.perf_counter()
    model_configs.TASK6_LIGHTGBM_GAP_MODEL = replace(base_config, **changes)
    result = assignment1.train_task6_lightgbm_gap_model(task)
    results[label] = (mae_days(result), changes, result)
    print(f"RESULT task6 {label}: {results[label][0]:.3f} days ({(time.perf_counter() - start) / 60:.1f} min)", flush=True)

# the two best changes together (if they touch different settings)
ranked = sorted(results, key=lambda label: results[label][0])
first, second = ranked[0], ranked[1]
if set(results[first][1]) != set(results[second][1]):
    start = time.perf_counter()
    model_configs.TASK6_LIGHTGBM_GAP_MODEL = replace(base_config, **results[first][1], **results[second][1])
    result = assignment1.train_task6_lightgbm_gap_model(task)
    label = f"{first} + {second}"
    results[label] = (mae_days(result), {**results[first][1], **results[second][1]}, result)
    print(f"RESULT task6 {label}: {results[label][0]:.3f} days ({(time.perf_counter() - start) / 60:.1f} min)", flush=True)

best_label = min(results, key=lambda label: results[label][0])
best_value, best_changes, best_result = results[best_label]
print(f"RESULT task6 best: {best_label} {best_value:.3f} days (stored {stored_mae:.3f}) changes {best_changes}", flush=True)
if best_value < stored_mae:
    backup_dir = assignment1.TASK6_PREDICTION_FILE.parent / "before_task6_sweep"
    backup_dir.mkdir(exist_ok=True)
    if not (backup_dir / assignment1.TASK6_PREDICTION_FILE.name).exists():
        shutil.copy2(assignment1.TASK6_PREDICTION_FILE, backup_dir / assignment1.TASK6_PREDICTION_FILE.name)
    store_gap_model(best_result, "task6", "lightgbm_gap_model")
    assignment1.write_task6_predictions(best_result.test_predictions)
    print(f"RESULT task6: adopted {best_label}, wrote {assignment1.TASK6_PREDICTION_FILE.name}", flush=True)
