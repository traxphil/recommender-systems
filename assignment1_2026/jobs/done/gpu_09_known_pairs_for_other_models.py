# Tasks 1 / 3: the models that can use every known pair (the 0-star ratings left out of training + the validation / test
# pairs, ratings never used) retrained with it: SVD++ / timeSVD++ / integrated model (N(u) = known pairs) and the rating
# transformer (known pairs in the history, rating hidden); the improved ones replace the stored ones only if the submitted
# blend improves with them (old files -> trained_models_backup_known_pairs/, old predictions -> output/before_known_pairs/);
# waits for cpu_06 (Bayesian FM variants) before it re-blends, so both never write the same files at once
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch

import assignment1
import lib.config as model_configs
import lib.model_storage as model_storage
from lib.helper_functioins import mse
from lib.model_storage import load_model_result, save_model_result, trained_model_path
from lib.models.blender import store_blender

# task -> sub-model -> (its config in lib/config.py, the flag that switches the known pairs on)
VARIANTS = {
    "task1": {"svdpp": ("TASK1_SVDPP", "use_test_pairs_as_implicit_feedback"),
              "integrated_model": ("TASK1_INTEGRATED_MODEL", "use_test_pairs_as_implicit_feedback"),
              "rating_transformer": ("TASK1_RATING_TRANSFORMER", "use_test_pairs_in_history")},
    "task3": {"svdpp": ("TASK3_SVDPP", "use_test_pairs_as_implicit_feedback"),
              "time_svdpp": ("TASK3_TIME_SVDPP", "use_test_pairs_as_implicit_feedback"),
              "integrated_model": ("TASK3_INTEGRATED_MODEL", "use_test_pairs_as_implicit_feedback"),
              "rating_transformer": ("TASK3_RATING_TRANSFORMER", "use_test_pairs_in_history")},
}
main_dir = model_storage.TRAINED_MODELS_DIR
candidates_dir = main_dir.parent / "trained_models_known_pairs_candidates"
jobs_dir = Path.cwd().parent / "jobs"


def nonzero_mse(task, predictions) -> float:
    nonzero = task.validation_data.ratings > 0
    return mse(predictions[nonzero], task.validation_data.ratings[nonzero])


def write_atomically(result, task_name: str, model_name: str):
    target = trained_model_path(task_name, model_name)
    temporary = target.with_suffix(".pt.tmp")
    torch.save(result._asdict(), temporary)
    os.replace(temporary, target)


# 1) every variant trained, kept in the candidates folder; the ones better on their own are blend candidates
tasks, improved = {}, {task_name: {} for task_name in VARIANTS}
for task_name, variants in VARIANTS.items():
    task = tasks[task_name] = assignment1.load_task1() if task_name == "task1" else assignment1.load_task3()
    trainers = assignment1.TASK1_SUB_MODEL_TRAINERS if task_name == "task1" else assignment1.TASK3_SUB_MODEL_TRAINERS
    for model_name, (config_name, flag) in variants.items():
        original_config = getattr(model_configs, config_name)
        setattr(model_configs, config_name, replace(original_config, **{flag: True}))
        start = time.perf_counter()
        result = trainers[model_name](task)
        setattr(model_configs, config_name, original_config)
        if isinstance(result.model, torch.nn.Module):
            result.model.to("cpu")                                  # several results wait for the blend -> off the GPU
        model_storage.TRAINED_MODELS_DIR = candidates_dir
        save_model_result(result, task_name, f"{model_name}_known_pairs")
        model_storage.TRAINED_MODELS_DIR = main_dir
        new_score = nonzero_mse(task, result.valid_predictions)
        old_score = nonzero_mse(task, load_model_result(task_name, model_name).valid_predictions)
        print(f"RESULT {task_name} {model_name} with known pairs: {new_score:.5f} (stored: {old_score:.5f}) "
              f"({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
        if new_score < old_score:
            improved[task_name][model_name] = result
        del result
        torch.cuda.empty_cache()

# 2) after cpu_06: blends with the improved variants; adopted only if the submitted blend improves
while not any((jobs_dir / folder / "cpu_06_bayesian_fm_variants.py").exists() for folder in ("done", "failed")):
    time.sleep(60)
for task_name, task in tasks.items():
    if not improved[task_name]:
        print(f"RESULT {task_name}: no variant better on its own -> nothing changed", flush=True)
        continue
    submitted_config = model_configs.TASK1_SUBMITTED_BLENDER if task_name == "task1" else model_configs.TASK3_SUBMITTED_BLENDER
    blender_configs = ((model_configs.TASK1_RIDGE_BLENDER, model_configs.TASK1_LIGHTGBM_BLENDER) if task_name == "task1"
                       else (model_configs.TASK3_RIDGE_BLENDER, model_configs.TASK3_LIGHTGBM_BLENDER))
    train_blender = assignment1.train_task1_blender if task_name == "task1" else assignment1.train_task3_blender
    prediction_file = assignment1.TASK1_PREDICTION_FILE if task_name == "task1" else assignment1.TASK3_PREDICTION_FILE
    write_predictions = assignment1.write_task1_predictions if task_name == "task1" else assignment1.write_task3_predictions
    old_blend_score = nonzero_mse(task, load_model_result(task_name, f"{submitted_config.method}_blender").valid_predictions)
    inputs = {name: improved[task_name].get(name) or load_model_result(task_name, name) for name in blender_configs[0].blended_models}
    for blender_config in blender_configs:
        blend = train_blender(task, inputs, blender_config)
        score = nonzero_mse(task, blend.valid_predictions)
        print(f"RESULT {task_name} {blender_config.method} blend with {sorted(improved[task_name])} known pairs: {score:.5f} "
              f"(stored submitted blend: {old_blend_score:.5f})", flush=True)
        if blender_config.method == submitted_config.method and score < old_blend_score:
            backup_dir = main_dir.parent / "trained_models_backup_known_pairs" / task_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            for model_name, result in improved[task_name].items():
                if not (backup_dir / f"{model_name}.pt").exists():
                    shutil.copy2(trained_model_path(task_name, model_name), backup_dir / f"{model_name}.pt")
                write_atomically(result, task_name, model_name)
            prediction_backup = prediction_file.parent / "before_known_pairs"
            prediction_backup.mkdir(exist_ok=True)
            if not (prediction_backup / prediction_file.name).exists():
                shutil.copy2(prediction_file, prediction_backup / prediction_file.name)
            store_blender(blend, task_name)
            write_predictions(blend.test_predictions)
            print(f"RESULT {task_name}: adopted {sorted(improved[task_name])} with known pairs, wrote {prediction_file.name}", flush=True)
