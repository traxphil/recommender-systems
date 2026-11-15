# Task 5: two more SASRec runs (seed 1; seed 2 with a 100-book history) -> stored; then the pair blend of all 9 sub-models
# (reader timing, bigger LightGBM); task_5_predictions.csv is rewritten only if the out-of-fold accuracy beats the stored blend
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
from lib.model_storage import load_model_result, save_model_result
from lib.models.pair_blender import store_pair_blender, train_pair_blender
from lib.pair_choice import pair_choice_accuracy

task = assignment1.load_task5()
for model_name, trainer in (("sasrec_seed_1", assignment1.train_task5_sasrec_seed_1),
                            ("sasrec_long_history", assignment1.train_task5_sasrec_long_history)):
    start = time.perf_counter()
    result = trainer(task)
    save_model_result(result, "task5", model_name)
    print(f"RESULT task5 {model_name}: validation accuracy on its own "
          f"{pair_choice_accuracy(result.valid_predictions, task.validation_pairs.first_is_positive):.5f} ({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
    del result
    torch.cuda.empty_cache()

config = model_configs.TASK5_PAIR_BLENDER
accuracy = lambda blend: np.mean((blend.valid_predictions > 0.5) == task.validation_pairs.first_is_positive)
old_accuracy = accuracy(load_model_result("task5", "pair_blender"))
inputs = {name: load_model_result("task5", name) for name in config.combined_models}
blend = train_pair_blender(task, inputs, config)
print(f"RESULT task5 blend of {len(inputs)} sub-models: {accuracy(blend):.5f} (stored blend: {old_accuracy:.5f})", flush=True)
for left_out in ("sasrec_seed_1", "sasrec_long_history"):
    names = tuple(name for name in config.combined_models if name != left_out)
    partial = train_pair_blender(task, {name: inputs[name] for name in names}, replace(config, combined_models=names))
    print(f"RESULT task5 blend without {left_out}: {accuracy(partial):.5f}", flush=True)
if accuracy(blend) > old_accuracy:
    backup_dir = assignment1.TASK5_PREDICTION_FILE.parent / "before_sasrec_variants"
    backup_dir.mkdir(exist_ok=True)
    if not (backup_dir / assignment1.TASK5_PREDICTION_FILE.name).exists():
        shutil.copy2(assignment1.TASK5_PREDICTION_FILE, backup_dir / assignment1.TASK5_PREDICTION_FILE.name)
    store_pair_blender(blend, "task5")
    assignment1.write_task5_predictions(blend.test_predictions)
    print(f"RESULT task5: wrote {assignment1.TASK5_PREDICTION_FILE.name}", flush=True)
else:
    print("RESULT task5: not better -> prediction file and stored blend kept", flush=True)
