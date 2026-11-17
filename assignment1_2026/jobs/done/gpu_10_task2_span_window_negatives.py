# Task 2 pair blender with (only from the training file, which has timestamps although the test file does not):
#   user span features: when the item was read by others vs. the user's active span (first / median / last day)
#   reading window features: how often others read the item within 5 books (time order) of the user's books
# quick CPU check 15 Nov (item cosine + popularity, 5 folds): 0.8772 -> + span 0.8858, + window 0.9113, + both 0.9143
# stage A (stored sub-models, 1 negative): every feature set + the bigger LightGBM of Tasks 4 / 5 (63 leaves, 800 trees)
# stage B: the 4 sub-models rescored with 4 negatives per held-out interaction (trained_models_task2_four_negatives/),
# blended with stage A's best setting, scored on the original pairs (the first quarter, identical to stage A)
# the best setting replaces the stored blend only if its out-of-fold accuracy on the original pairs is higher
# (old prediction file -> output/before_task2_features/, old task2 models -> trained_models_backup_task2_one_negative/)
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

SUB_MODELS = ("popularity", "item_cosine", "ease", "mult_vae")
BIGGER_LIGHTGBM = dict(lightgbm_n_estimators=800, lightgbm_learning_rate=0.03, lightgbm_num_leaves=63, lightgbm_min_child_samples=100)
main_dir = model_storage.TRAINED_MODELS_DIR
four_negatives_dir = main_dir.parent / "trained_models_task2_four_negatives"


def original_pair_accuracy(probabilities, first_is_positive, n_original) -> float:
    return float(np.mean((probabilities[:n_original] > 0.5) == first_is_positive[:n_original]))


def adopt(blend, accuracy: float, label: str):
    global best_accuracy
    if accuracy <= best_accuracy:
        return
    backup_dir = assignment1.TASK2_PREDICTION_FILE.parent / "before_task2_features"
    backup_dir.mkdir(exist_ok=True)
    if not (backup_dir / assignment1.TASK2_PREDICTION_FILE.name).exists():
        shutil.copy2(assignment1.TASK2_PREDICTION_FILE, backup_dir / assignment1.TASK2_PREDICTION_FILE.name)
    store_pair_blender(blend, "task2")
    assignment1.write_task2_predictions(blend.test_predictions)
    best_accuracy = accuracy
    print(f"RESULT task2: adopted {label} ({accuracy:.5f}), wrote {assignment1.TASK2_PREDICTION_FILE.name}", flush=True)


# ---- stage A: stored sub-models, 1 negative per held-out interaction ---------------------------------------
task = assignment1.load_task2()
first_is_positive = task.validation_pairs.first_is_positive
n_original = len(first_is_positive)
stored_accuracy = original_pair_accuracy(load_model_result("task2", "pair_blender").valid_predictions, first_is_positive, n_original)
best_accuracy = stored_accuracy
print(f"RESULT task2 stored blend: {stored_accuracy:.5f}", flush=True)

sub_model_results = {name: load_model_result("task2", name) for name in SUB_MODELS}
base_config = model_configs.TASK2_PAIR_BLENDER
settings = {
    "span": dict(use_user_span_features=True),
    "window": dict(use_reading_window_features=True),
    "span + window": dict(use_user_span_features=True, use_reading_window_features=True),
    "span + window + bigger LightGBM": dict(use_user_span_features=True, use_reading_window_features=True, **BIGGER_LIGHTGBM),
}
stage_a = {}
for label, changes in settings.items():
    start = time.perf_counter()
    blend = train_pair_blender(task, sub_model_results, replace(base_config, **changes))
    accuracy = original_pair_accuracy(blend.valid_predictions, first_is_positive, n_original)
    stage_a[label] = accuracy
    print(f"RESULT task2 stage A {label}: {accuracy:.5f} ({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
    adopt(blend, accuracy, f"stage A {label}")
best_label = max(stage_a, key=stage_a.get)
best_changes = settings[best_label]
print(f"RESULT task2 stage A best: {best_label} {stage_a[best_label]:.5f}", flush=True)
del task, sub_model_results

# ---- stage B: 4 negatives per held-out interaction -----------------------------------------------------------
model_configs.TASK2_NEGATIVES_PER_VALIDATION_PAIR = 4
model_storage.TRAINED_MODELS_DIR = four_negatives_dir
task = assignment1.load_task2()
first_is_positive = task.validation_pairs.first_is_positive
print(f"{len(first_is_positive)} validation pairs ({n_original} original)", flush=True)
for name in SUB_MODELS:
    if trained_model_path("task2", name).exists():
        print(f"reusing {name}", flush=True)
        continue
    start = time.perf_counter()
    result = assignment1.TASK2_SUB_MODEL_TRAINERS[name](task)
    save_model_result(result, "task2", name)
    print(f"RESULT task2 {name:<12} on its own, original pairs "
          f"{pair_choice_accuracy(result.valid_predictions[:n_original], first_is_positive[:n_original]):.5f} "
          f"({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
    del result
    torch.cuda.empty_cache()

start = time.perf_counter()
blend = train_pair_blender(task, {name: load_model_result("task2", name) for name in SUB_MODELS}, replace(base_config, **best_changes))
accuracy = original_pair_accuracy(blend.valid_predictions, first_is_positive, n_original)
print(f"RESULT task2 stage B 4 negatives + {best_label}: original pairs {accuracy:.5f} | all pairs "
      f"{np.mean((blend.valid_predictions > 0.5) == first_is_positive):.5f} ({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
if accuracy > best_accuracy:
    store_pair_blender(blend, "task2")                              # into trained_models_task2_four_negatives/
    model_storage.TRAINED_MODELS_DIR = main_dir
    old_models, backup_models = main_dir / "task2", main_dir.parent / "trained_models_backup_task2_one_negative" / "task2"
    if not backup_models.exists():
        shutil.copytree(old_models, backup_models)
    for path in (four_negatives_dir / "task2").glob("*.pt"):
        shutil.copy2(path, old_models / path.name)
    adopt(blend, accuracy, f"stage B 4 negatives + {best_label}")   # stores the blender again in trained_models/task2
print(f"RESULT task2 final: {best_accuracy:.5f} (stored before: {stored_accuracy:.5f})", flush=True)
