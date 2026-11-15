# (CPU lane: starts right after cpu_05 instead of waiting for the slow longer-history SASRec of gpu_06)
# Bayesian factorization machine variants, trained side by side (myFM uses one CPU core per fit):
#   task1 known_pairs        : TASK1 BFM, 30 + 380 draws, N(u) / N(i) = every known pair (0-star ratings, validation + test pairs)
#   task3 rank_128           : TASK3 time BFM with rank 128
#   task3 periods_known_pairs: TASK3 time BFM + user periods of 365 / 1000 days + every known pair as implicit feedback
# quick CPU check 15 Nov (rank 8, 5 + 20 draws, MSE on the validation ratings > 0): Task 3 0.70796 -> periods 365 / 1000
# 0.70329 -> + known pairs 0.70204 (user day 0.70857, 30 days 0.70733: no gain); Task 1 0.57080 -> known pairs 0.56913
# every finished variant: kept in trained_models_bfm_candidates/; if its validation MSE (ratings > 0) beats the stored model,
# it replaces it (old file -> trained_models_backup_bfm_upgrade/) and the task is re-blended; the prediction file is
# rewritten only if the submitted blend's out-of-fold MSE improves (old file -> output/before_bfm_variants/)
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import assignment1
import lib.config as model_configs
import lib.model_storage as model_storage
from lib.helper_functioins import mse
from lib.model_storage import load_model_result, save_model_result, trained_model_path
from lib.models.bayesian_factorization_machine import train_bayesian_factorization_machine
from lib.models.blender import store_blender

STORED_NAME = {"task1": "bayesian_factorization_machine", "task3": "time_bayesian_factorization_machine"}
VARIANTS = {
    ("task1", "known_pairs"): (model_configs.TASK1_BAYESIAN_FACTORIZATION_MACHINE,
                               dict(n_burn_in_iterations=30, n_kept_samples=380, use_test_pairs_as_implicit_feedback=True)),
    ("task3", "rank_128"): (model_configs.TASK3_TIME_BAYESIAN_FACTORIZATION_MACHINE, dict(rank=128)),
    ("task3", "periods_known_pairs"): (model_configs.TASK3_TIME_BAYESIAN_FACTORIZATION_MACHINE,
                                       dict(user_period_days=(365, 1000), use_test_pairs_as_implicit_feedback=True)),
}


def load_task(task_name: str):
    return assignment1.load_task1() if task_name == "task1" else assignment1.load_task3()


def train_variant(task_name: str, variant: str):
    base_config, changes = VARIANTS[(task_name, variant)]
    task = load_task(task_name)
    start = time.perf_counter()
    result = train_bayesian_factorization_machine(assignment1.rating_model_task(task), replace(base_config, **changes))
    return task_name, variant, result, (time.perf_counter() - start) / 60


def validation_mse(task, result) -> float:
    nonzero = task.validation_data.ratings > 0
    return mse(result.valid_predictions[nonzero], task.validation_data.ratings[nonzero])


def wait_for(job_name: str):
    jobs_dir = Path.cwd().parent / "jobs"
    while not any((jobs_dir / folder / job_name).exists() for folder in ("done", "failed")):
        time.sleep(60)


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
            backup_dir = prediction_file.parent / "before_bfm_variants"
            backup_dir.mkdir(exist_ok=True)
            if not (backup_dir / prediction_file.name).exists():
                shutil.copy2(prediction_file, backup_dir / prediction_file.name)
            store_blender(blend, task_name)
            write_predictions(blend.test_predictions)
            print(f"RESULT {task_name}: wrote {prediction_file.name}", flush=True)


def adopt_if_better(task_name: str, variant: str, result):
    task = load_task(task_name)
    if task_name == "task1":
        wait_for("cpu_05_task1_bfm_more_draws.py")              # it may replace the Task 1 model and re-blend first
    stored_name = STORED_NAME[task_name]
    old_score = validation_mse(task, load_model_result(task_name, stored_name))
    new_score = validation_mse(task, result)
    print(f"RESULT {task_name} {variant}: {new_score:.5f} (stored {stored_name}: {old_score:.5f})", flush=True)
    if new_score >= old_score:
        print(f"RESULT {task_name} {variant}: not better -> stored model kept", flush=True)
        return
    stored_path = trained_model_path(task_name, stored_name)
    backup_path = stored_path.parent.parent.parent / "trained_models_backup_bfm_upgrade" / task_name / stored_path.name
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(stored_path, backup_path)
    save_model_result(result, task_name, stored_name)
    print(f"RESULT {task_name} {variant}: adopted as {stored_name} (config {result.model})", flush=True)
    reblend(task_name, task)


if __name__ == "__main__":
    print(f"{os.cpu_count()} CPU threads; variants: {list(VARIANTS)}", flush=True)
    main_dir = model_storage.TRAINED_MODELS_DIR
    candidates_dir = main_dir.parent / "trained_models_bfm_candidates"
    with ProcessPoolExecutor(max_workers=len(VARIANTS)) as pool:
        futures = [pool.submit(train_variant, task_name, variant) for task_name, variant in VARIANTS]
        for future in as_completed(futures):
            task_name, variant, result, minutes = future.result()
            print(f"finished {task_name} {variant} ({minutes:.0f} min)", flush=True)
            model_storage.TRAINED_MODELS_DIR = candidates_dir
            save_model_result(result, task_name, f"{STORED_NAME[task_name]}_{variant}")
            model_storage.TRAINED_MODELS_DIR = main_dir
            adopt_if_better(task_name, variant, result)
