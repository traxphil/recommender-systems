# Tasks 1 / 3 Bayesian factorization machine: second independent Gibbs chains (n_chains = 2: seeds 0 and 1 averaged ->
# less sampling noise; a longer single chain made Task 1 worse) + Task 3 at rank 128 with the stored features (periods,
# known pairs), all side by side (myFM: one core per fit):
#   task1 seed_1          : TASK1 config, seed 1      -> averaged with the stored seed-0 model = n_chains 2
#   task3 seed_1          : TASK3 config, seed 1      -> averaged with the stored seed-0 model = n_chains 2
#   task3 rank_128_seed_0 : TASK3 config, rank 128    -> on its own (n_chains 1)
#   task3 rank_128_seed_1 : TASK3 config, rank 128, seed 1 -> averaged with rank_128_seed_0 = rank 128, n_chains 2
# every candidate replaces the stored model only if its validation MSE (ratings > 0) is lower, then the task is re-blended;
# the prediction file is rewritten only if the submitted blend improves (old files -> trained_models_backup_bfm_chains/,
# output/before_bfm_chains/); candidates kept in trained_models_bfm_candidates/
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np

import assignment1
import lib.config as model_configs
import lib.model_storage as model_storage
from lib.custom_types import ModelResult
from lib.helper_functioins import mse
from lib.model_storage import load_model_result, save_model_result, trained_model_path
from lib.models.bayesian_factorization_machine import train_bayesian_factorization_machine
from lib.models.blender import store_blender

STORED_NAME = {"task1": "bayesian_factorization_machine", "task3": "time_bayesian_factorization_machine"}
BASE_CONFIG = {"task1": model_configs.TASK1_BAYESIAN_FACTORIZATION_MACHINE, "task3": model_configs.TASK3_TIME_BAYESIAN_FACTORIZATION_MACHINE}
VARIANTS = {
    ("task1", "seed_1"): dict(seed=1),
    ("task3", "seed_1"): dict(seed=1),
    ("task3", "rank_128_seed_0"): dict(rank=128),
    ("task3", "rank_128_seed_1"): dict(rank=128, seed=1),
}


def load_task(task_name: str):
    return assignment1.load_task1() if task_name == "task1" else assignment1.load_task3()


def train_variant(task_name: str, variant: str):
    task = load_task(task_name)
    start = time.perf_counter()
    result = train_bayesian_factorization_machine(assignment1.rating_model_task(task), replace(BASE_CONFIG[task_name], **VARIANTS[(task_name, variant)]))
    return task_name, variant, result, (time.perf_counter() - start) / 60


def validation_mse(task, result) -> float:
    nonzero = task.validation_data.ratings > 0
    return mse(result.valid_predictions[nonzero], task.validation_data.ratings[nonzero])


def averaged(first: ModelResult, second: ModelResult, config) -> ModelResult:
    return ModelResult(config, [], (first.valid_predictions + second.valid_predictions) / 2,
                       (first.test_predictions + second.test_predictions) / 2)


def reblend(task_name: str, task):
    nonzero = task.validation_data.ratings > 0
    ratings = task.validation_data.ratings[nonzero]
    submitted_config = model_configs.TASK1_SUBMITTED_BLENDER if task_name == "task1" else model_configs.TASK3_SUBMITTED_BLENDER
    blender_configs = ((model_configs.TASK1_RIDGE_BLENDER, model_configs.TASK1_LIGHTGBM_BLENDER) if task_name == "task1"
                       else (model_configs.TASK3_RIDGE_BLENDER, model_configs.TASK3_LIGHTGBM_BLENDER))
    train_blender = assignment1.train_task1_blender if task_name == "task1" else assignment1.train_task3_blender
    prediction_file = assignment1.TASK1_PREDICTION_FILE if task_name == "task1" else assignment1.TASK3_PREDICTION_FILE
    write_predictions = assignment1.write_task1_predictions if task_name == "task1" else assignment1.write_task3_predictions
    old_blend_score = mse(load_model_result(task_name, f"{submitted_config.method}_blender").valid_predictions[nonzero], ratings)
    inputs = {name: load_model_result(task_name, name) for name in blender_configs[0].blended_models}
    for blender_config in blender_configs:
        blend = train_blender(task, inputs, blender_config)
        score = mse(blend.valid_predictions[nonzero], ratings)
        print(f"RESULT {task_name} {blender_config.method} blend: {score:.5f} (stored submitted blend: {old_blend_score:.5f})", flush=True)
        if blender_config.method == submitted_config.method and score < old_blend_score:
            backup_dir = prediction_file.parent / "before_bfm_chains"
            backup_dir.mkdir(exist_ok=True)
            if not (backup_dir / prediction_file.name).exists():
                shutil.copy2(prediction_file, backup_dir / prediction_file.name)
            store_blender(blend, task_name)
            write_predictions(blend.test_predictions)
            print(f"RESULT {task_name}: wrote {prediction_file.name}", flush=True)


def adopt_if_better(task_name: str, label: str, result: ModelResult):
    task = load_task(task_name)
    stored_name = STORED_NAME[task_name]
    old_score = validation_mse(task, load_model_result(task_name, stored_name))
    new_score = validation_mse(task, result)
    print(f"RESULT {task_name} {label}: {new_score:.5f} (stored {stored_name}: {old_score:.5f})", flush=True)
    if new_score >= old_score:
        print(f"RESULT {task_name} {label}: not better -> stored model kept", flush=True)
        return
    stored_path = trained_model_path(task_name, stored_name)
    backup_path = stored_path.parent.parent.parent / "trained_models_backup_bfm_chains" / task_name / stored_path.name
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(stored_path, backup_path)
    save_model_result(result, task_name, stored_name)
    print(f"RESULT {task_name} {label}: adopted as {stored_name} (config {result.model})", flush=True)
    reblend(task_name, task)


if __name__ == "__main__":
    print(f"{os.cpu_count()} CPU threads; variants: {list(VARIANTS)}", flush=True)
    main_dir = model_storage.TRAINED_MODELS_DIR
    candidates_dir = main_dir.parent / "trained_models_bfm_candidates"
    # the stored seed-0 models (= the configs in lib/config.py), kept for the averages even if a candidate replaces them
    seed_0 = {task_name: load_model_result(task_name, STORED_NAME[task_name]) for task_name in STORED_NAME}
    for task_name, result in seed_0.items():
        assert result.model == BASE_CONFIG[task_name], (task_name, result.model)
    finished = {}
    with ProcessPoolExecutor(max_workers=len(VARIANTS)) as pool:
        futures = [pool.submit(train_variant, task_name, variant) for task_name, variant in VARIANTS]
        for future in as_completed(futures):
            task_name, variant, result, minutes = future.result()
            print(f"finished {task_name} {variant} ({minutes:.0f} min)", flush=True)
            finished[(task_name, variant)] = result
            model_storage.TRAINED_MODELS_DIR = candidates_dir
            save_model_result(result, task_name, f"{STORED_NAME[task_name]}_{variant}")
            model_storage.TRAINED_MODELS_DIR = main_dir
            if variant == "seed_1":
                adopt_if_better(task_name, "2 chains (seeds 0 + 1)",
                                averaged(seed_0[task_name], result, replace(BASE_CONFIG[task_name], n_chains=2)))
            elif variant.startswith("rank_128"):
                if variant == "rank_128_seed_0":
                    adopt_if_better(task_name, "rank 128", result)
                if ("task3", "rank_128_seed_0") in finished and ("task3", "rank_128_seed_1") in finished:
                    adopt_if_better(task_name, "rank 128, 2 chains",
                                    averaged(finished[("task3", "rank_128_seed_0")], finished[("task3", "rank_128_seed_1")],
                                             replace(BASE_CONFIG["task3"], rank=128, n_chains=2)))
