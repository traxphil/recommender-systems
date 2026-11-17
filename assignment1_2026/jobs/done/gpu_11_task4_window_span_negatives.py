# Task 4 pair blender with the Task 2 features (jobs/gpu_10: Task 2 0.93689 -> 0.94091): reading window (how often others
# read the item within 5 books of the user's books, time order) and the user's active span; Task 4 already has the reader
# timing around t and the bigger LightGBM -> stage A: + window, + span + window on the stored sub-models; stage B: the 5
# sub-models rescored with 4 negatives per held-out interaction (trained_models_task4_four_negatives/), blended with stage
# A's best setting; adopted only if the out-of-fold accuracy on the original pairs beats the stored blend
# (old prediction file -> output/before_task4_features/, old task4 models -> trained_models_backup_task4_one_negative/)
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch

import assignment1
import lib.config as model_configs
import lib.model_storage as model_storage
from lib.model_storage import load_model_result, save_model_result, trained_model_path
from lib.models.pair_blender import store_pair_blender, train_pair_blender
from lib.pair_choice import pair_choice_accuracy

SUB_MODELS = ("popularity", "item_cosine", "ease", "time_ease", "mult_vae")
main_dir = model_storage.TRAINED_MODELS_DIR
four_negatives_dir = main_dir.parent / "trained_models_task4_four_negatives"


def original_pair_accuracy(probabilities, first_is_positive, n_original) -> float:
    return float(np.mean((probabilities[:n_original] > 0.5) == first_is_positive[:n_original]))


def adopt(blend, accuracy: float, label: str):
    global best_accuracy
    if accuracy <= best_accuracy:
        return
    backup_dir = assignment1.TASK4_PREDICTION_FILE.parent / "before_task4_features"
    backup_dir.mkdir(exist_ok=True)
    if not (backup_dir / assignment1.TASK4_PREDICTION_FILE.name).exists():
        shutil.copy2(assignment1.TASK4_PREDICTION_FILE, backup_dir / assignment1.TASK4_PREDICTION_FILE.name)
    store_pair_blender(blend, "task4")
    assignment1.write_task4_predictions(blend.test_predictions)
    best_accuracy = accuracy
    print(f"RESULT task4: adopted {label} ({accuracy:.5f}), wrote {assignment1.TASK4_PREDICTION_FILE.name}", flush=True)


# ---- stage A: stored sub-models, 1 negative per held-out interaction ---------------------------------------
task = assignment1.load_task4()
first_is_positive = task.validation_pairs.first_is_positive
n_original = len(first_is_positive)
stored_accuracy = original_pair_accuracy(load_model_result("task4", "pair_blender").valid_predictions, first_is_positive, n_original)
best_accuracy = stored_accuracy
print(f"RESULT task4 stored blend: {stored_accuracy:.5f}", flush=True)

sub_model_results = {name: load_model_result("task4", name) for name in SUB_MODELS}
base_config = model_configs.TASK4_PAIR_BLENDER
settings = {
    "window": dict(use_reading_window_features=True),
    "span + window": dict(use_user_span_features=True, use_reading_window_features=True),
}
stage_a = {}
for label, changes in settings.items():
    start = time.perf_counter()
    blend = train_pair_blender(task, sub_model_results, replace(base_config, **changes))
    accuracy = original_pair_accuracy(blend.valid_predictions, first_is_positive, n_original)
    stage_a[label] = accuracy
    print(f"RESULT task4 stage A {label}: {accuracy:.5f} ({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
    adopt(blend, accuracy, f"stage A {label}")
best_label = max(stage_a, key=stage_a.get)
best_changes = settings[best_label]
print(f"RESULT task4 stage A best: {best_label} {stage_a[best_label]:.5f}", flush=True)
del task, sub_model_results

# ---- stage B: 4 negatives per held-out interaction -----------------------------------------------------------
model_configs.TASK4_NEGATIVES_PER_VALIDATION_PAIR = 4
model_storage.TRAINED_MODELS_DIR = four_negatives_dir
task = assignment1.load_task4()
first_is_positive = task.validation_pairs.first_is_positive
print(f"{len(first_is_positive)} validation pairs ({n_original} original)", flush=True)
for name in SUB_MODELS:
    if trained_model_path("task4", name).exists():
        print(f"reusing {name}", flush=True)
        continue
    start = time.perf_counter()
    result = assignment1.TASK4_SUB_MODEL_TRAINERS[name](task)
    save_model_result(result, "task4", name)
    print(f"RESULT task4 {name:<12} on its own, original pairs "
          f"{pair_choice_accuracy(result.valid_predictions[:n_original], first_is_positive[:n_original]):.5f} "
          f"({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
    del result
    torch.cuda.empty_cache()

start = time.perf_counter()
blend = train_pair_blender(task, {name: load_model_result("task4", name) for name in SUB_MODELS}, replace(base_config, **best_changes))
accuracy = original_pair_accuracy(blend.valid_predictions, first_is_positive, n_original)
print(f"RESULT task4 stage B 4 negatives + {best_label}: original pairs {accuracy:.5f} | all pairs "
      f"{np.mean((blend.valid_predictions > 0.5) == first_is_positive):.5f} ({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
if accuracy > best_accuracy:
    store_pair_blender(blend, "task4")                              # into trained_models_task4_four_negatives/
    model_storage.TRAINED_MODELS_DIR = main_dir
    old_models, backup_models = main_dir / "task4", main_dir.parent / "trained_models_backup_task4_one_negative" / "task4"
    if not backup_models.exists():
        shutil.copytree(old_models, backup_models)
    for path in (four_negatives_dir / "task4").glob("*.pt"):
        shutil.copy2(path, old_models / path.name)
    adopt(blend, accuracy, f"stage B 4 negatives + {best_label}")   # stores the blender again in trained_models/task4
print(f"RESULT task4 final: {best_accuracy:.5f} (stored before: {stored_accuracy:.5f})", flush=True)
