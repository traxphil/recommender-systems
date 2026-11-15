# Task 6: the LightGBM gap model with the settings in lib/config.py (earlier gaps + "rows from end" input + bagging) ->
# stored and task_6_predictions.csv written only if its out-of-fold MAE beats the stored model (old file -> output/before_task6_bagging/)
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np

import assignment1
import lib.config as model_configs
from lib.models.gap_regressor import load_gap_model, store_gap_model

task = assignment1.load_task6()
old_mae = np.abs(load_gap_model("task6", "lightgbm_gap_model").valid_predictions - task.validation_gaps).mean()
result = assignment1.train_task6_lightgbm_gap_model(task)
new_mae = np.abs(result.valid_predictions - task.validation_gaps).mean()
print(f"RESULT task6: {model_configs.TASK6_LIGHTGBM_GAP_MODEL}", flush=True)
print(f"RESULT task6: out-of-fold MAE {new_mae / 86400:.3f} days (stored model {old_mae / 86400:.3f} days)", flush=True)
if new_mae < old_mae:
    backup_dir = assignment1.TASK6_PREDICTION_FILE.parent / "before_task6_bagging"
    backup_dir.mkdir(exist_ok=True)
    if not (backup_dir / assignment1.TASK6_PREDICTION_FILE.name).exists():
        shutil.copy2(assignment1.TASK6_PREDICTION_FILE, backup_dir / assignment1.TASK6_PREDICTION_FILE.name)
    store_gap_model(result, "task6", "lightgbm_gap_model")
    assignment1.write_task6_predictions(result.test_predictions)
    print(f"RESULT task6: wrote {assignment1.TASK6_PREDICTION_FILE.name}", flush=True)
else:
    print("RESULT task6: not better -> kept", flush=True)
