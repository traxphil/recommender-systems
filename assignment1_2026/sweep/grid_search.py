"""
Grid search over the hyperparameters of one sub-model at a time (used by the task notebooks in this folder).

    run_grid_search("task1", "svdpp", {"user_bias_regularization": [1.0, 2.0, 3.0], "n_factors": 500})

- a list with several values is swept, a single value (or a one-element list) stays constant in every run,
  every parameter that is not mentioned keeps its value from lib/config.py
- every combination: fit on the training split, score on the validation split
  (no refit on 100 %, nothing is stored in trained_models/ -> the submitted models stay untouched)
- one JSON file per sweep: output/<task>/<model>_<start time>.json, rewritten after every run
  (best result + its full parameter set, the swept and constant values, the score of every run)
- rerun_finalists_on_other_splits=True: the best n_finalists combinations are also trained on the validation splits
  with seeds 1 and 2; the winner is then the combination with the best mean over the three splits
"""

import gc
import io
import itertools
import json
import os
import sys
import time
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Paths: this file lives in assignment1_2026/sweep, the model code in assignment1_2026/submission
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

SWEEP_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = SWEEP_DIR.parent / "submission"
OUTPUT_DIR = SWEEP_DIR / "output"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

import assignment1                                             # noqa: E402  (needs the path above)
import lib.config as model_configs                             # noqa: E402  (looked up at call time -> edits are picked up by autoreload)
from lib.config import DEVICE, SPLIT_SEED, VALIDATION_FRACTION  # noqa: E402
from lib.data_loading import load_rating_prediction_task       # noqa: E402
from lib.helper_functioins import mse                          # noqa: E402
from lib.models.bias_model import BiasModel                    # noqa: E402
from lib.models.factorization_machine import train_factorization_machine  # noqa: E402
from lib.models.item_knn import ItemKnnModel                   # noqa: E402


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Validation predictions of one model with one config (fit on the training split only)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

@dataclass
class ValidationRun:
    predictions: np.ndarray                 # one prediction per validation row
    best_epoch: int | None                  # early-stopping epoch of the PyTorch models, None for closed-form / kNN


def best_epoch_of(epoch_history: list[dict]) -> int:
    return min(epoch_history, key=lambda epoch_entry: epoch_entry["valid_mse"])["epoch"]


def validate_bias_model(task, config) -> ValidationRun:
    model = BiasModel(task.training_data, task.n_user_rows, task.n_item_rows, config)
    return ValidationRun(model.predict(task.validation_data.user_rows, task.validation_data.item_rows), None)


# SVD / SVD++ (config.refit_on_train_and_valid is always False here, see run_grid_search)
def validate_latent_factor_model(task, config) -> ValidationRun:
    model_result = assignment1.train_latent_factor_model(task, **asdict(config))
    return ValidationRun(model_result.valid_predictions, best_epoch_of(model_result.history))


def validate_factorization_machine(task, config) -> ValidationRun:
    model_result = train_factorization_machine(task, config)
    return ValidationRun(model_result.valid_predictions, best_epoch_of(model_result.history))


def validate_item_knn(task, config) -> ValidationRun:
    model = ItemKnnModel(task.training_data, task.n_user_rows, task.n_item_rows, config)
    return ValidationRun(model.predict(task.validation_data.user_rows, task.validation_data.item_rows), None)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# What can be swept: per task the data, the metric and the models
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

@dataclass(frozen=True)
class SweepableModel:
    config_name: str                        # name of the model's config in lib/config.py (= every value that is not swept)
    validate: Callable                      # (task, config) -> ValidationRun


@dataclass(frozen=True)
class SweepTask:
    load: Callable                          # validation split seed -> task with that 90 / 10 split
    metric_name: str
    score: Callable                         # (task, validation predictions) -> score
    lower_is_better: bool
    models: dict                            # model name -> SweepableModel


SWEEP_TASKS = {
    "task1": SweepTask(
        load=lambda split_seed: load_rating_prediction_task(assignment1.TASK1_TRAINING_FILE, assignment1.TASK1_TEST_FILE,
                                                            VALIDATION_FRACTION, split_seed),
        metric_name="validation MSE",
        score=lambda task, predictions: mse(predictions, task.validation_data.ratings),
        lower_is_better=True,
        models={
            "bias_model": SweepableModel("TASK1_BIAS_MODEL", validate_bias_model),
            "svd": SweepableModel("TASK1_SVD", validate_latent_factor_model),
            "svdpp": SweepableModel("TASK1_SVDPP", validate_latent_factor_model),
            "factorization_machine": SweepableModel("TASK1_FACTORIZATION_MACHINE", validate_factorization_machine),
            "item_knn": SweepableModel("TASK1_ITEM_KNN", validate_item_knn),
        },
    ),
}

# every split is loaded once per kernel (loading the CSV takes a few seconds)
_loaded_tasks = {}


def task_with_split(task_name: str, split_seed: int):
    if (task_name, split_seed) not in _loaded_tasks:
        print(f"loading {task_name} with validation split seed {split_seed} ...", flush=True)
        _loaded_tasks[(task_name, split_seed)] = SWEEP_TASKS[task_name].load(split_seed)
    return _loaded_tasks[(task_name, split_seed)]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Small helpers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# {name: [several values]} -> swept, {name: value} or {name: [one value]} -> constant
def split_parameter_grid(parameter_grid: dict, base_config) -> tuple[dict, dict]:
    field_names = [config_field.name for config_field in fields(base_config)]
    unknown_names = [name for name in parameter_grid if name not in field_names]
    if unknown_names:
        raise ValueError(f"unknown parameter(s) {unknown_names}; {type(base_config).__name__} has {field_names}")
    if "refit_on_train_and_valid" in parameter_grid:
        raise ValueError("the sweep never refits (it cannot change the validation score) -> leave refit_on_train_and_valid out")

    swept_values, constant_values = {}, {}
    for name, values in parameter_grid.items():
        if not isinstance(values, list):        # a single value (also a tuple, which some configs use as one value)
            constant_values[name] = values
        elif len(values) == 1:
            constant_values[name] = values[0]
        elif len(values) > 1:
            swept_values[name] = values
        else:
            raise ValueError(f"{name}: empty list")
    return swept_values, constant_values


def describe_values(values: dict) -> str:
    return " ".join(f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}" for name, value in values.items())


def is_better(score: float, other_score: float, lower_is_better: bool) -> bool:
    return score < other_score if lower_is_better else score > other_score


# numpy numbers / paths in configs -> plain JSON values
def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


# write to a temporary file first -> the JSON file is never half-written, even if the kernel stops mid-write
# Windows refuses to replace a file another program has open at that moment (VS Code tab, virus scanner, indexer)
# -> retry for a few seconds; if it is still locked, warn and go on (the next run writes the file again)
def write_report(report: dict, output_path: Path, attempts: int = 20):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    report_text = json.dumps(report, indent=2, default=json_value)
    for _ in range(attempts):
        try:
            temporary_path.write_text(report_text, encoding="utf-8")
            os.replace(temporary_path, output_path)
            return
        except PermissionError:
            time.sleep(0.25)
    print(f"    (could not update {output_path.name}: locked by another program -> latest state in "
          f"{temporary_path.name}, retried after the next run)", flush=True)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Windows sleeps after its idle timeout even while Python computes -> ask it to stay awake while a sweep runs
# Source: Microsoft Win32 API docs, SetThreadExecutionState (ES_CONTINUOUS = 0x80000000, ES_SYSTEM_REQUIRED = 0x1)
def keep_computer_awake(enabled: bool):
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED if enabled else ES_CONTINUOUS)


# one training run -> (score, best epoch, seconds); the training's own log lines are hidden unless asked for
def train_and_score(task_name: str, sweepable_model: SweepableModel, config, split_seed: int, show_training_log: bool):
    task = task_with_split(task_name, split_seed)
    start_time = time.perf_counter()
    with nullcontext() if show_training_log else redirect_stdout(io.StringIO()):
        validation_run = sweepable_model.validate(task, config)
    score = float(SWEEP_TASKS[task_name].score(task, validation_run.predictions))
    seconds = time.perf_counter() - start_time
    best_epoch = validation_run.best_epoch
    del validation_run                      # model and predictions -> free the (GPU) memory before the next run
    free_memory()
    return score, best_epoch, seconds


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# The sweep
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def run_grid_search(task_name: str, model_name: str, parameter_grid: dict,
                    rerun_finalists_on_other_splits: bool = False, n_finalists: int = 3, other_split_seeds: tuple = (1, 2),
                    show_training_log: bool = False) -> dict:

    # 1) what to run: every combination of the swept values, on top of the config in lib/config.py
    if task_name not in SWEEP_TASKS:
        raise ValueError(f"unknown task {task_name!r}; sweepable: {list(SWEEP_TASKS)}")
    sweep_task = SWEEP_TASKS[task_name]
    if model_name not in sweep_task.models:
        raise ValueError(f"unknown model {model_name!r} for {task_name}; sweepable: {list(sweep_task.models)}")
    sweepable_model = sweep_task.models[model_name]

    configured = getattr(model_configs, sweepable_model.config_name)
    swept_values, constant_values = split_parameter_grid(parameter_grid, configured)
    # reported parameters = lib/config.py + the grid's constants (+ the swept values of a run) -> can be copied back as they are
    base_config = replace(configured, **constant_values)
    # trained parameters: the same, but never refit on 100 % (cannot change the validation score, doubles the run time)
    training_config = (replace(base_config, refit_on_train_and_valid=False)
                       if "refit_on_train_and_valid" in [config_field.name for config_field in fields(base_config)] else base_config)
    combinations = [dict(zip(swept_values, values)) for values in itertools.product(*swept_values.values())]

    n_finalist_runs = min(n_finalists, len(combinations)) * len(other_split_seeds) if rerun_finalists_on_other_splits else 0
    n_runs = len(combinations) + n_finalist_runs
    lower_is_better = sweep_task.lower_is_better

    # 2) the output file, rewritten after every run
    started_at = datetime.now()
    output_path = OUTPUT_DIR / task_name / f"{model_name}_{started_at:%Y-%m-%d_%H-%M-%S}.json"
    report = {
        "task": task_name,
        "model": model_name,
        "metric": f"{sweep_task.metric_name} ({'lower' if lower_is_better else 'higher'} is better)",
        "status": "running",
        "started": f"{started_at:%Y-%m-%d %H:%M:%S}",
        "last_update": f"{started_at:%Y-%m-%d %H:%M:%S}",
        "runs_finished": f"0 / {n_runs}",
        "device": str(DEVICE),
        "best": None,
        "swept_parameters": swept_values,
        "constant_parameters": constant_values,          # set in the grid; everything else as in lib/config.py
        "base_config": sweepable_model.config_name,
        "runs": [],                                      # every combination on validation split seed SPLIT_SEED
    }
    if rerun_finalists_on_other_splits:
        report["finalists"] = []                         # best combinations on every split + their mean

    def record_progress(n_finished: int):
        report["runs_finished"] = f"{n_finished} / {n_runs}"
        report["last_update"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
        write_report(report, output_path)

    # the best combination: score (mean over the splits for finalists), where it was measured, its full parameter set
    def best_entry(swept: dict, score: float, split_seeds: list, best_epoch=None,
                   scores_by_split_seed: dict | None = None, best_epoch_by_split_seed: dict | None = None) -> dict:
        entry = {"score": round(score, 6), "split_seeds": split_seeds}
        if scores_by_split_seed is None:
            entry["best_epoch"] = best_epoch
        else:
            entry["scores_by_split_seed"] = scores_by_split_seed
            entry["best_epoch_by_split_seed"] = best_epoch_by_split_seed
        entry["swept_values"] = swept
        entry["parameters"] = asdict(replace(base_config, **swept))
        return entry

    print(f"{task_name} / {model_name}: {len(combinations)} combination{'' if len(combinations) == 1 else 's'}"
          + (f" + {n_finalist_runs} finalist runs on split seeds {list(other_split_seeds)}" if n_finalist_runs else "")
          + f" | swept: {swept_values or '-'} | constant: {constant_values or '-'}\n-> {output_path}", flush=True)
    write_report(report, output_path)
    keep_computer_awake(True)
    n_finished = 0

    try:
        # 3) every combination on the usual validation split
        for run_number, swept in enumerate(combinations, start=1):
            print(f"[{run_number}/{len(combinations)}] {describe_values(swept) or 'config as given'} ...", flush=True)
            score, best_epoch, seconds = train_and_score(task_name, sweepable_model, replace(training_config, **swept),
                                                         SPLIT_SEED, show_training_log)
            report["runs"].append({**swept, "score": round(score, 6), "best_epoch": best_epoch, "seconds": round(seconds, 1)})

            if report["best"] is None or is_better(score, report["best"]["score"], lower_is_better):
                report["best"] = best_entry(swept, score, [SPLIT_SEED], best_epoch=best_epoch)
            n_finished += 1
            record_progress(n_finished)
            print(f"    {sweep_task.metric_name} {score:.5f}" + (f" (epoch {best_epoch})" if best_epoch is not None else "")
                  + f" | {seconds:.0f} s | best so far {report['best']['score']:.5f}", flush=True)

        # 4) optional: the best combinations again on other validation splits -> winner by the mean over all splits
        if rerun_finalists_on_other_splits:
            ranked_runs = sorted(report["runs"], key=lambda run: run["score"], reverse=not lower_is_better)
            for finalist_run in ranked_runs[:n_finalists]:
                swept = {name: finalist_run[name] for name in swept_values}
                scores_by_split_seed = {SPLIT_SEED: finalist_run["score"]}
                best_epoch_by_split_seed = {SPLIT_SEED: finalist_run["best_epoch"]}
                for split_seed in other_split_seeds:
                    print(f"[finalist, split seed {split_seed}] {describe_values(swept) or 'config as given'} ...", flush=True)
                    score, best_epoch, seconds = train_and_score(task_name, sweepable_model, replace(training_config, **swept),
                                                                 split_seed, show_training_log)
                    scores_by_split_seed[split_seed] = round(score, 6)
                    best_epoch_by_split_seed[split_seed] = best_epoch
                    n_finished += 1
                    print(f"    {sweep_task.metric_name} {score:.5f} | {seconds:.0f} s", flush=True)

                mean_score = float(np.mean(list(scores_by_split_seed.values())))
                report["finalists"].append({**swept, "mean_score": round(mean_score, 6), "scores_by_split_seed": scores_by_split_seed,
                                            "best_epoch_by_split_seed": best_epoch_by_split_seed})
                record_progress(n_finished)

            best_finalist = sorted(report["finalists"], key=lambda finalist: finalist["mean_score"], reverse=not lower_is_better)[0]
            swept = {name: best_finalist[name] for name in swept_values}
            report["best"] = best_entry(swept, best_finalist["mean_score"], list(best_finalist["scores_by_split_seed"]),
                                        scores_by_split_seed=best_finalist["scores_by_split_seed"],
                                        best_epoch_by_split_seed=best_finalist["best_epoch_by_split_seed"])

        report["status"] = "finished"

    except KeyboardInterrupt:
        report["status"] = "stopped early"
        print(f"stopped early -> the finished runs are in {output_path}", flush=True)
        raise

    except Exception as error:
        report["status"] = f"failed: {type(error).__name__}: {error}"[:300]
        print(f"run failed -> the finished runs are in {output_path}", flush=True)
        raise

    finally:
        keep_computer_awake(False)
        record_progress(n_finished)

    print(f"best: {describe_values(report['best']['swept_values']) or 'config as given'} -> "
          f"{sweep_task.metric_name} {report['best']['score']:.5f}"
          + (" (mean over split seeds " + ", ".join(map(str, report["best"]["split_seeds"])) + ")" if rerun_finalists_on_other_splits else ""),
          flush=True)
    return report


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Overview of the finished sweeps
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# one line per output file of a task (optionally only one model), oldest first
def print_sweep_overview(task_name: str, model_name: str | None = None):
    output_paths = sorted((OUTPUT_DIR / task_name).glob(f"{model_name or '*'}_*.json"))
    if not output_paths:
        print(f"no sweeps in {OUTPUT_DIR / task_name}")
        return
    for output_path in output_paths:
        report = json.loads(output_path.read_text(encoding="utf-8"))
        best = report["best"]
        best_text = (f"{best['score']:.5f} with {describe_values(best['swept_values']) or 'config as given'}"
                     if best else "no finished run")
        print(f"{output_path.name:<50} {report['status']:<14} {report['runs_finished']:>9} | best {best_text}")
