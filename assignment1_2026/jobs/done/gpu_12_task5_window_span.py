# Task 5 pair blender + the Task 2 features (jobs/gpu_10, Task 2 0.93689 -> 0.94091): reading window (how often others read
# the item within 5 books, in time order, of the user's books) and the user's active span, on the stored sub-models
# (4 negatives per validation user); accuracy on the original pairs (the first quarter); task_5_predictions.csv rewritten
# only if better than the stored blend (old file -> output/before_task5_window/)
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np

import assignment1
import lib.config as model_configs
from lib.model_storage import load_model_result
from lib.models.pair_blender import store_pair_blender, train_pair_blender

task = assignment1.load_task5()
first_is_positive = task.validation_pairs.first_is_positive
n_original = len(first_is_positive) // model_configs.TASK5_NEGATIVES_PER_VALIDATION_USER
accuracy = lambda probabilities: float(np.mean((probabilities[:n_original] > 0.5) == first_is_positive[:n_original]))
best_accuracy = stored_accuracy = accuracy(load_model_result("task5", "pair_blender").valid_predictions)
print(f"RESULT task5 stored blend: {stored_accuracy:.5f} ({n_original} original pairs of {len(first_is_positive)})", flush=True)

sub_model_results = {name: load_model_result("task5", name) for name in model_configs.TASK5_PAIR_BLENDER.combined_models}
for label, changes in {"window": dict(use_reading_window_features=True),
                       "span + window": dict(use_user_span_features=True, use_reading_window_features=True)}.items():
    start = time.perf_counter()
    blend = train_pair_blender(task, sub_model_results, replace(model_configs.TASK5_PAIR_BLENDER, **changes))
    blend_accuracy = accuracy(blend.valid_predictions)
    print(f"RESULT task5 {label}: {blend_accuracy:.5f} ({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
    if blend_accuracy > best_accuracy:
        backup_dir = assignment1.TASK5_PREDICTION_FILE.parent / "before_task5_window"
        backup_dir.mkdir(exist_ok=True)
        if not (backup_dir / assignment1.TASK5_PREDICTION_FILE.name).exists():
            shutil.copy2(assignment1.TASK5_PREDICTION_FILE, backup_dir / assignment1.TASK5_PREDICTION_FILE.name)
        store_pair_blender(blend, "task5")
        assignment1.write_task5_predictions(blend.test_predictions)
        best_accuracy = blend_accuracy
        print(f"RESULT task5: adopted {label}, wrote {assignment1.TASK5_PREDICTION_FILE.name}", flush=True)
print(f"RESULT task5 final: {best_accuracy:.5f} (stored before: {stored_accuracy:.5f})", flush=True)
