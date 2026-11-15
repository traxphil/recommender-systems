# check (nothing stored, no prediction file written): Task 5 / Task 4 pair blends from the stored sub-models,
# as now vs. + the reader-timing features (use_reader_timing_features) -> out-of-fold accuracy on the validation pairs
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np

import assignment1
import lib.config as model_configs
from lib.model_storage import load_model_result
from lib.models.pair_blender import train_pair_blender

for task_number in (5, 4):
    task = getattr(assignment1, f"load_task{task_number}")()
    config = getattr(model_configs, f"TASK{task_number}_PAIR_BLENDER")
    inputs = {name: load_model_result(f"task{task_number}", name) for name in config.combined_models}
    print(f"===== task {task_number}: {len(task.validation_pairs.user_rows)} validation pairs, models {list(config.combined_models)}", flush=True)
    for label, blender_config in (("as now", config), ("+ reader timing", replace(config, use_reader_timing_features=True))):
        blend = train_pair_blender(task, inputs, blender_config)
        accuracy = np.mean((blend.valid_predictions > 0.5) == task.validation_pairs.first_is_positive)
        print(f"RESULT task {task_number} {label:<16} out-of-fold accuracy {accuracy:.5f}", flush=True)
