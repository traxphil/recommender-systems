# Task 1: every rating sub-model of the blend retrained on the ratings > 0 (TRAIN_RATING_MODELS_WITHOUT_ZERO_RATINGS) and
# stored; skipped here: the zero models (they keep the zeros) and the Bayesian FM (CPU lane job)
# the stored version from before is copied to trained_models_backup_with_zeros/ first
import shutil
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch

import assignment1
import lib.config as model_configs
from lib.helper_functioins import mse
from lib.model_storage import save_model_result, trained_model_path

assert model_configs.TRAIN_RATING_MODELS_WITHOUT_ZERO_RATINGS
SKIPPED = {"zero_model", "bayesian_factorization_machine"}
task = assignment1.load_task1()
nonzero = task.validation_data.ratings > 0
for model_name in model_configs.TASK1_RIDGE_BLENDER.blended_models:
    if model_name in SKIPPED:
        continue
    start = time.perf_counter()
    try:
        result = assignment1.TASK1_SUB_MODEL_TRAINERS[model_name](task)
    except Exception:
        traceback.print_exc()
        print(f"FAILED task1 {model_name}", flush=True)
        continue
    stored_path = trained_model_path("task1", model_name)
    backup_path = stored_path.parent.parent.parent / "trained_models_backup_with_zeros" / "task1" / stored_path.name
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if stored_path.exists() and not backup_path.exists():
        shutil.copy2(stored_path, backup_path)
    save_model_result(result, "task1", model_name)
    print(f"RESULT task1 {model_name:<36} MSE on validation rows > 0 {mse(result.valid_predictions[nonzero], task.validation_data.ratings[nonzero]):.5f}"
          f" ({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
    del result
    torch.cuda.empty_cache()
