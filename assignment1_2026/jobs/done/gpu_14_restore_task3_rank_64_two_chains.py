# cpu_08 stored the Task 3 rank-128 two-chain Bayesian FM (0.67461, a hair better than the rank-64 two-chain model 0.67464),
# but the blend with it was worse (0.66738 vs 0.66712), so task_3_predictions.csv and the stored ridge blender still come
# from the rank-64 two-chain model (= lib/config.py). This puts the rank-64 two-chain model back (= average of the backed-up
# seed-0 model and the seed-1 candidate) and checks that the blend reproduces the stored blend; the rank-128 model goes to
# trained_models_bfm_candidates/ as time_bayesian_factorization_machine_rank_128_2_chains
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np

import assignment1
import lib.config as model_configs
import lib.model_storage as model_storage
from lib.custom_types import ModelResult
from lib.helper_functioins import mse
from lib.model_storage import load_model_result, save_model_result

main_dir = model_storage.TRAINED_MODELS_DIR
config = model_configs.TASK3_TIME_BAYESIAN_FACTORIZATION_MACHINE


def load_from(directory, model_name):
    model_storage.TRAINED_MODELS_DIR = directory
    result = load_model_result("task3", model_name)
    model_storage.TRAINED_MODELS_DIR = main_dir
    return result


def save_to(directory, result, model_name):
    model_storage.TRAINED_MODELS_DIR = directory
    save_model_result(result, "task3", model_name)
    model_storage.TRAINED_MODELS_DIR = main_dir


seed_0 = load_from(main_dir.parent / "trained_models_backup_bfm_chains", "time_bayesian_factorization_machine")
seed_1 = load_from(main_dir.parent / "trained_models_bfm_candidates", "time_bayesian_factorization_machine_seed_1")
assert seed_0.model == replace(config, n_chains=1), seed_0.model
assert seed_1.model == replace(config, seed=1, n_chains=1), seed_1.model
two_chains = ModelResult(config, [], (seed_0.valid_predictions + seed_1.valid_predictions) / 2,
                         (seed_0.test_predictions + seed_1.test_predictions) / 2)

task = assignment1.load_task3()
nonzero = task.validation_data.ratings > 0
score = lambda predictions: mse(predictions[nonzero], task.validation_data.ratings[nonzero])
print(f"RESULT rank 64 two chains: {score(two_chains.valid_predictions):.5f} (expected 0.67464)", flush=True)

rank_128 = load_model_result("task3", "time_bayesian_factorization_machine")
print(f"stored now: rank {rank_128.model.rank}, {score(rank_128.valid_predictions):.5f}", flush=True)
if rank_128.model.rank == 128:
    save_to(main_dir.parent / "trained_models_bfm_candidates", rank_128, "time_bayesian_factorization_machine_rank_128_2_chains")
save_model_result(two_chains, "task3", "time_bayesian_factorization_machine")

stored_blend = load_model_result("task3", "ridge_blender")
inputs = {name: load_model_result("task3", name) for name in model_configs.TASK3_RIDGE_BLENDER.blended_models}
blend = assignment1.train_task3_blender(task, inputs, model_configs.TASK3_RIDGE_BLENDER)
print(f"RESULT ridge blend from the stored models: {score(blend.valid_predictions):.5f} (stored blend {score(stored_blend.valid_predictions):.5f}); "
      f"test predictions identical: {np.allclose(blend.test_predictions, stored_blend.test_predictions)}", flush=True)
