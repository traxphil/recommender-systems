# Task 1: bayesian_factorization_machine retrained without zero ratings with 30 + 380 draws -> replaces the stored one only if its validation MSE
# (ratings > 0) is lower (old file -> trained_models_backup_bfm_upgrade/); then ridge + LightGBM blends on the ratings > 0,
# task_1_predictions.csv rewritten only if the submitted method's out-of-fold MSE beats the stored blend
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import assignment1
import lib.config as model_configs
from lib.helper_functioins import mse
from lib.model_storage import load_model_result, save_model_result, trained_model_path
from lib.models.bayesian_factorization_machine import train_bayesian_factorization_machine
from lib.models.blender import store_blender

task = assignment1.load_task1()
nonzero = task.validation_data.ratings > 0
ratings = task.validation_data.ratings[nonzero]
old_score = mse(load_model_result("task1", "bayesian_factorization_machine").valid_predictions[nonzero], ratings)
config = replace(model_configs.TASK1_BAYESIAN_FACTORIZATION_MACHINE, n_burn_in_iterations=30, n_kept_samples=380)
start = time.perf_counter()
result = train_bayesian_factorization_machine(assignment1.rating_model_task(task), config)
new_score = mse(result.valid_predictions[nonzero], ratings)
print(f"RESULT task1 bayesian_factorization_machine with 30 + 380 draws: {new_score:.5f} (stored: {old_score:.5f}) ({(time.perf_counter() - start) / 60:.0f} min)", flush=True)
if new_score >= old_score:
    print("RESULT not better -> stored model and predictions kept", flush=True)
    sys.exit(0)
stored_path = trained_model_path("task1", "bayesian_factorization_machine")
backup_path = stored_path.parent.parent.parent / "trained_models_backup_bfm_upgrade" / "task1" / stored_path.name
backup_path.parent.mkdir(parents=True, exist_ok=True)
if not backup_path.exists():
    shutil.copy2(stored_path, backup_path)
save_model_result(result, "task1", "bayesian_factorization_machine")
del result

jobs_dir = Path.cwd().parent / "jobs"
for waiting_for in ("gpu_05_reblend_tasks_1_3.py",):
    while not any((jobs_dir / folder / waiting_for).exists() for folder in ("done", "failed")):
        time.sleep(60)
submitted_method = model_configs.TASK1_SUBMITTED_BLENDER.method
old_blend_score = mse(load_model_result("task1", f"{submitted_method}_blender").valid_predictions[nonzero], ratings)
blender_configs = (model_configs.TASK1_RIDGE_BLENDER, model_configs.TASK1_LIGHTGBM_BLENDER)
inputs = {name: load_model_result("task1", name) for name in blender_configs[0].blended_models}
for blender_config in blender_configs:
    blend = assignment1.train_task1_blender(task, inputs, blender_config)
    score = mse(blend.valid_predictions[nonzero], ratings)
    print(f"RESULT task1 {blender_config.method} blend: {score:.5f} (stored submitted blend: {old_blend_score:.5f})", flush=True)
    if blender_config.method == submitted_method and score < old_blend_score:
        backup_dir = assignment1.TASK1_PREDICTION_FILE.parent / "before_bfm_upgrade"
        backup_dir.mkdir(exist_ok=True)
        if not (backup_dir / assignment1.TASK1_PREDICTION_FILE.name).exists():
            shutil.copy2(assignment1.TASK1_PREDICTION_FILE, backup_dir / assignment1.TASK1_PREDICTION_FILE.name)
        store_blender(blend, "task1")
        assignment1.write_task1_predictions(blend.test_predictions)
        print(f"RESULT task1: wrote {assignment1.TASK1_PREDICTION_FILE.name}", flush=True)
