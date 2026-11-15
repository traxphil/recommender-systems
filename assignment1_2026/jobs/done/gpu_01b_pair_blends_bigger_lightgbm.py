# Tasks 4 / 5: pair blends with reader timing + the bigger LightGBM (lib/config.py, 15 Nov grid) -> stored, predictions written
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import assignment1
import lib.config as model_configs
from lib.model_storage import load_model_result
from lib.models.pair_blender import store_pair_blender, train_pair_blender
for task_number in (5, 4):
    task = getattr(assignment1, f"load_task{task_number}")()
    config = getattr(model_configs, f"TASK{task_number}_PAIR_BLENDER")
    blend = train_pair_blender(task, {name: load_model_result(f"task{task_number}", name) for name in config.combined_models}, config)
    store_pair_blender(blend, f"task{task_number}")
    getattr(assignment1, f"write_task{task_number}_predictions")(blend.test_predictions)
    accuracy = np.mean((blend.valid_predictions > 0.5) == task.validation_pairs.first_is_positive)
    print(f"RESULT task {task_number}: {config.lightgbm_num_leaves} leaves / {config.lightgbm_n_estimators} trees -> {accuracy:.5f}, "
          f"wrote {getattr(assignment1, f'TASK{task_number}_PREDICTION_FILE').name}", flush=True)
