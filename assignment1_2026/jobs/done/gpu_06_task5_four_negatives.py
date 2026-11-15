# Task 5 with 4 negatives per validation user (TASK5_NEGATIVES_PER_VALIDATION_USER): every sub-model retrained in
# trained_models_task5_four_negatives/ (a model already there is reused), so the blender learns from 4x the pairs; scored on
# the original pairs (the first quarter, identical to before) -> if better than the stored blend: predictions written
# (old file -> output/before_four_negatives/) and the models copied into trained_models/task5 (old ones -> backup folder)
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch

import lib.config as model_configs
import lib.model_storage as model_storage
import assignment1
from lib.model_storage import load_model_result, save_model_result, trained_model_path
from lib.models.pair_blender import store_pair_blender, train_pair_blender
from lib.pair_choice import pair_choice_accuracy

main_dir = model_storage.TRAINED_MODELS_DIR
four_negatives_dir = main_dir.parent / "trained_models_task5_four_negatives"
model_configs.TASK5_NEGATIVES_PER_VALIDATION_USER = 1
original_task = assignment1.load_task5()
n_original = len(original_task.validation_pairs.user_rows)
stored_accuracy = np.mean((load_model_result("task5", "pair_blender").valid_predictions > 0.5) == original_task.validation_pairs.first_is_positive)
print(f"stored blend (1 negative per user): {stored_accuracy:.5f}", flush=True)
del original_task

model_configs.TASK5_NEGATIVES_PER_VALIDATION_USER = 4
model_storage.TRAINED_MODELS_DIR = four_negatives_dir
task = assignment1.load_task5()
first_is_positive = task.validation_pairs.first_is_positive
print(f"{len(first_is_positive)} validation pairs ({n_original} original)", flush=True)

trainers = {
    "popularity": assignment1.train_task5_popularity_model,
    "item_cosine": assignment1.train_task5_item_cosine_model,
    "transition": assignment1.train_task5_transition_model,
    "ease + time_ease": lambda task: dict(zip(("ease", "time_ease"), assignment1.train_task5_ease_models(task))),
    "mult_vae": assignment1.train_task5_mult_vae,
    "sasrec": assignment1.train_task5_sasrec,
    "sasrec_long_history": assignment1.train_task5_sasrec_long_history,
    "sasrec_longer_history": assignment1.train_task5_sasrec_longer_history,
}
for label, trainer in trainers.items():
    names = label.split(" + ")
    if all(trained_model_path("task5", name).exists() for name in names):
        print(f"reusing {label}", flush=True)
        continue
    start = time.perf_counter()
    result = trainer(task)
    results = result if isinstance(result, dict) else {label: result}
    for name, model_result in results.items():
        save_model_result(model_result, "task5", name)
        print(f"RESULT task5 {name:<22} on its own, original pairs {pair_choice_accuracy(model_result.valid_predictions[:n_original], first_is_positive[:n_original]):.5f}"
              f" ({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
    del result, results
    torch.cuda.empty_cache()

config = model_configs.TASK5_PAIR_BLENDER
base_models = ("popularity", "item_cosine", "transition", "ease", "time_ease", "mult_vae", "sasrec", "sasrec_long_history")
blends = {}
for models in (base_models + ("sasrec_longer_history",), base_models):
    blend = train_pair_blender(task, {name: load_model_result("task5", name) for name in models}, replace(config, combined_models=models))
    accuracy = np.mean((blend.valid_predictions[:n_original] > 0.5) == first_is_positive[:n_original])
    blends[models] = (accuracy, blend)
    print(f"RESULT task5 blend, 4 negatives, {len(models)} sub-models (sequence features {config.use_sequence_features}): "
          f"original pairs {accuracy:.5f} | all pairs {np.mean((blend.valid_predictions > 0.5) == first_is_positive):.5f}", flush=True)

best_models = max(blends, key=lambda models: blends[models][0])
best_accuracy, best_blend = blends[best_models]
if best_accuracy > stored_accuracy:
    store_pair_blender(best_blend, "task5")                         # into trained_models_task5_four_negatives/
    backup_dir = assignment1.TASK5_PREDICTION_FILE.parent / "before_four_negatives"
    backup_dir.mkdir(exist_ok=True)
    if not (backup_dir / assignment1.TASK5_PREDICTION_FILE.name).exists():
        shutil.copy2(assignment1.TASK5_PREDICTION_FILE, backup_dir / assignment1.TASK5_PREDICTION_FILE.name)
    assignment1.write_task5_predictions(best_blend.test_predictions)
    model_backup = main_dir.parent / "trained_models_backup_task5_one_negative"
    if not model_backup.exists():
        shutil.copytree(main_dir / "task5", model_backup)
    for model_file in (four_negatives_dir / "task5").glob("*.pt"):
        shutil.copy2(model_file, main_dir / "task5" / model_file.name)
    print(f"RESULT task5: {best_accuracy:.5f} > {stored_accuracy:.5f} -> wrote {assignment1.TASK5_PREDICTION_FILE.name}, models copied to "
          f"trained_models/task5 (old ones in {model_backup.name}); best blend: {best_models}", flush=True)
else:
    print(f"RESULT task5: {best_accuracy:.5f} not better than {stored_accuracy:.5f} -> nothing written", flush=True)
