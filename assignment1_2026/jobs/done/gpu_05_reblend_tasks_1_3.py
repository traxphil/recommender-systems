# Tasks 1 / 3: after the zero-free retraining (GPU jobs before this one, the Bayesian FMs in the CPU lane) -> ridge + LightGBM
# blends on the validation rows > 0; the submitted method's predictions are written only if its out-of-fold MSE beats the
# blend stored before (current files -> output/before_zero_free_retraining/)
import shutil
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import assignment1
import lib.config as model_configs
from lib.helper_functioins import mse
from lib.model_storage import load_model_result
from lib.models.blender import store_blender

jobs_dir = Path.cwd().parent / "jobs"
waiting_for = "cpu_03_retrain_task1_bfm_without_zeros.py"
while not any((jobs_dir / folder / waiting_for).exists() for folder in ("done", "failed")):
    time.sleep(60)                                                  # the CPU lane's last Bayesian FM is not finished yet
print(f"{waiting_for} finished -> blending", flush=True)

backup_dir = assignment1.TASK1_PREDICTION_FILE.parent / "before_zero_free_retraining"
backup_dir.mkdir(exist_ok=True)
for task_number in (1, 3):
    task = getattr(assignment1, f"load_task{task_number}")()
    nonzero = task.validation_data.ratings > 0
    ratings = task.validation_data.ratings[nonzero]
    prediction_file = getattr(assignment1, f"TASK{task_number}_PREDICTION_FILE")
    if not (backup_dir / prediction_file.name).exists():
        shutil.copy2(prediction_file, backup_dir / prediction_file.name)
    submitted_method = getattr(model_configs, f"TASK{task_number}_SUBMITTED_BLENDER").method
    old_score = mse(load_model_result(f"task{task_number}", f"{submitted_method}_blender").valid_predictions[nonzero], ratings)
    blender_configs = (getattr(model_configs, f"TASK{task_number}_RIDGE_BLENDER"), getattr(model_configs, f"TASK{task_number}_LIGHTGBM_BLENDER"))
    inputs = {name: load_model_result(f"task{task_number}", name) for name in blender_configs[0].blended_models}
    for blender_config in blender_configs:
        blend = getattr(assignment1, f"train_task{task_number}_blender")(task, inputs, blender_config)
        store_blender(blend, f"task{task_number}")
        score = mse(blend.valid_predictions[nonzero], ratings)
        print(f"RESULT task {task_number} {blender_config.method:<8} blend, sub-models without zeros: {score:.5f} "
              f"(blend before, submitted {submitted_method}: {old_score:.5f})", flush=True)
        if blender_config.method == submitted_method:
            if score < old_score:
                getattr(assignment1, f"write_task{task_number}_predictions")(blend.test_predictions)
                print(f"RESULT task {task_number}: wrote {prediction_file.name}", flush=True)
            else:
                print(f"RESULT task {task_number}: not better -> {prediction_file.name} kept", flush=True)
