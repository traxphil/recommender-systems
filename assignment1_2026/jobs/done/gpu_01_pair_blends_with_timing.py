# Tasks 4 / 5: pair blends with the reader-timing features (lib/config.py) from the stored sub-models -> stored, predictions
# written; plus a small LightGBM grid for Tasks 2 / 4 / 5 (reported only, nothing written for it)
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

backup_dir = assignment1.TASK1_PREDICTION_FILE.parent / "before_pair_timing"
backup_dir.mkdir(exist_ok=True)
for task_number in (4, 5):
    prediction_file = getattr(assignment1, f"TASK{task_number}_PREDICTION_FILE")
    if not (backup_dir / prediction_file.name).exists():
        shutil.copy2(prediction_file, backup_dir / prediction_file.name)

GRID = [(31, 200, 400, 0.05), (63, 200, 400, 0.05), (31, 100, 400, 0.05), (15, 200, 400, 0.05), (31, 200, 800, 0.03), (63, 100, 800, 0.03)]
for task_number in (5, 4, 2):
    task = getattr(assignment1, f"load_task{task_number}")()
    config = getattr(model_configs, f"TASK{task_number}_PAIR_BLENDER")
    inputs = {name: load_model_result(f"task{task_number}", name) for name in config.combined_models}
    accuracy = lambda blend: np.mean((blend.valid_predictions > 0.5) == task.validation_pairs.first_is_positive)
    if task_number in (4, 5):
        blend = train_pair_blender(task, inputs, config)
        store_pair_blender(blend, f"task{task_number}")
        getattr(assignment1, f"write_task{task_number}_predictions")(blend.test_predictions)
        print(f"RESULT task {task_number}: blend with reader timing {accuracy(blend):.5f} -> wrote {getattr(assignment1, f'TASK{task_number}_PREDICTION_FILE').name}", flush=True)
    for leaves, min_child, trees, rate in GRID:
        grid_config = replace(config, lightgbm_num_leaves=leaves, lightgbm_min_child_samples=min_child, lightgbm_n_estimators=trees,
                              lightgbm_learning_rate=rate)
        print(f"GRID task {task_number}: leaves {leaves} min_child {min_child} trees {trees} rate {rate} -> "
              f"{accuracy(train_pair_blender(task, inputs, grid_config)):.5f}", flush=True)
