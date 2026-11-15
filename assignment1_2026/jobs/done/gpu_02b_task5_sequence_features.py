# Task 5: pair blend of the stored sub-models with vs. without the last-books sequence features (use_sequence_features);
# the better one is stored and written if it beats the stored blend (current file -> output/before_sequence_features/)
import shutil
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np

import assignment1
import lib.config as model_configs
from lib.model_storage import load_model_result
from lib.models.pair_blender import store_pair_blender, train_pair_blender

task = assignment1.load_task5()
config = model_configs.TASK5_PAIR_BLENDER
accuracy = lambda blend: np.mean((blend.valid_predictions > 0.5) == task.validation_pairs.first_is_positive)
stored_accuracy = accuracy(load_model_result("task5", "pair_blender"))
inputs = {name: load_model_result("task5", name) for name in config.combined_models}
results = {}
for label, blender_config in (("without sequence features", replace(config, use_sequence_features=False)),
                              ("with sequence features", replace(config, use_sequence_features=True))):
    results[label] = train_pair_blender(task, inputs, blender_config)
    print(f"RESULT task5 blend of {len(inputs)} sub-models {label}: {accuracy(results[label]):.5f} (stored blend {stored_accuracy:.5f})", flush=True)
best_label = max(results, key=lambda label: accuracy(results[label]))
if accuracy(results[best_label]) > stored_accuracy:
    backup_dir = assignment1.TASK5_PREDICTION_FILE.parent / "before_sequence_features"
    backup_dir.mkdir(exist_ok=True)
    if not (backup_dir / assignment1.TASK5_PREDICTION_FILE.name).exists():
        shutil.copy2(assignment1.TASK5_PREDICTION_FILE, backup_dir / assignment1.TASK5_PREDICTION_FILE.name)
    store_pair_blender(results[best_label], "task5")
    assignment1.write_task5_predictions(results[best_label].test_predictions)
    print(f"RESULT task5: wrote {assignment1.TASK5_PREDICTION_FILE.name} ({best_label})", flush=True)
else:
    print("RESULT task5: no improvement over the stored blend -> kept", flush=True)
