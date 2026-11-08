"""
Hyperparameter sweep for the Task 1 latent factor models (SVD and SVD++).

The model and the training code stay unchanged: every run calls
assignment1.train_latent_factor_model with one set of hyperparameters.

Files in this folder:
    sweep_queue.json           configs to run, one entry per run (edited between rounds)
    sweep_results.csv          one row per finished run: all parameters + scores
    sweep_epoch_history.csv    train / valid MSE of every epoch of every run
    sweep_log.txt              everything the training prints (epoch lines included)
"""

import csv
import ctypes
import gc
import inspect
import json
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Paths: this file lives in assignment1_2026/sweep, the model code in assignment1_2026/submission
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

SWEEP_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = SWEEP_DIR.parent / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

import assignment1                                             # noqa: E402  (needs the path above)
from lib.config import DEVICE                                  # noqa: E402
from lib.data_loading import load_rating_prediction_task       # noqa: E402

QUEUE_FILE = SWEEP_DIR / "sweep_queue.json"
RESULTS_FILE = SWEEP_DIR / "sweep_results.csv"
EPOCH_HISTORY_FILE = SWEEP_DIR / "sweep_epoch_history.csv"
LOG_FILE = SWEEP_DIR / "sweep_log.txt"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Which hyperparameters a config may set
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# keyword arguments of train_latent_factor_model a config may override
SWEEPABLE_PARAMETER_NAMES = ("use_implicit_feedback", "n_factors", "learning_rate", "user_bias_regularization",
                             "item_bias_regularization", "factor_regularization", "ratings_per_batch", "max_epochs",
                             "early_stopping_patience", "seed")

# before the bias lambda was split into a user and an item part, configs and result rows had one "bias_regularization"
# -> a config may still use it (sets both sides), old result rows are converted when the results file is read
SHARED_BIAS_REGULARIZATION_NAME = "bias_regularization"

# defaults are read from the function signature -> a config that leaves a value out gets exactly assignment1's default
_signature_parameters = inspect.signature(assignment1.train_latent_factor_model).parameters
DEFAULT_PARAMETERS = {name: _signature_parameters[name].default for name in SWEEPABLE_PARAMETER_NAMES}

RESULT_COLUMNS = ["config_id", "round", "status", "model", *SWEEPABLE_PARAMETER_NAMES, "split_seed",
                  "best_valid_mse", "best_epoch", "epochs_run", "train_mse_at_best_epoch", "bias_only_valid_mse",
                  "seconds_total", "seconds_per_epoch", "device", "finished_at", "note", "error"]
EPOCH_HISTORY_COLUMNS = ["config_id", "epoch", "train_mse", "valid_mse"]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Small file helpers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# timestamped line in the notebook output and in sweep_log.txt
def announce(message: str):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")


# append rows to a CSV, writing the header the first time
def append_csv_rows(csv_path: Path, column_names: list[str], rows: list[dict]):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=column_names, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def read_result_rows() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    with open(RESULTS_FILE, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


# results file written before the bias lambda was split: one bias_regularization column
# -> rewritten once with user_bias_regularization = item_bias_regularization = that value (same runs, same scores),
#    so new rows (written with RESULT_COLUMNS) and old rows share one header
def convert_results_file_to_split_bias_regularization():
    if not RESULTS_FILE.exists():
        return
    with open(RESULTS_FILE, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if SHARED_BIAS_REGULARIZATION_NAME not in reader.fieldnames:
            return
        result_rows = list(reader)

    for row in result_rows:
        shared_bias_regularization = row.pop(SHARED_BIAS_REGULARIZATION_NAME)
        row["user_bias_regularization"] = row["item_bias_regularization"] = shared_bias_regularization

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result_rows)
    announce(f"converted {len(result_rows)} result rows to user / item bias regularization columns")


# the queue file can be replaced while we read it -> retry a few times before giving up for this poll
def read_queue() -> dict:
    for _ in range(5):
        try:
            with open(QUEUE_FILE, encoding="utf-8") as queue_file:
                return json.load(queue_file)
        except (json.JSONDecodeError, FileNotFoundError, PermissionError):
            time.sleep(2)
    announce("could not read sweep_queue.json -> trying again at the next poll")
    return {"configs": [], "finished": False}


# Windows sleeps after its idle timeout even while Python computes -> ask it to stay awake while the sweep runs
# (the display may still turn off; the request ends with keep_computer_awake(False) or when the kernel stops)
# Source: Microsoft Win32 API docs, SetThreadExecutionState (ES_CONTINUOUS = 0x80000000, ES_SYSTEM_REQUIRED = 0x1)
def keep_computer_awake(enabled: bool):
    if sys.platform != "win32":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED if enabled else ES_CONTINUOUS)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# One run = one call of train_latent_factor_model
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# the 90 / 10 split of the training file, loaded once per split seed (seed 0 = the split assignment1 uses)
_tasks_by_split_seed = {}


def task_for_split_seed(split_seed: int):
    if split_seed not in _tasks_by_split_seed:
        announce(f"loading Task 1 data with validation split seed {split_seed} ...")
        _tasks_by_split_seed[split_seed] = load_rating_prediction_task(
            assignment1.TASK1_TRAINING_FILE, assignment1.TASK1_TEST_FILE, split_seed=split_seed)
    return _tasks_by_split_seed[split_seed]


# config from the queue -> the full set of keyword arguments for train_latent_factor_model
def parameters_of_config(config: dict) -> dict:
    parameters = dict(DEFAULT_PARAMETERS)
    if SHARED_BIAS_REGULARIZATION_NAME in config:
        parameters["user_bias_regularization"] = parameters["item_bias_regularization"] = config[SHARED_BIAS_REGULARIZATION_NAME]
    parameters.update({name: config[name] for name in SWEEPABLE_PARAMETER_NAMES if name in config})
    return parameters


def describe_parameters(parameters: dict) -> str:
    model_name = "SVD++" if parameters["use_implicit_feedback"] else "SVD"
    return (f"{model_name} k={parameters['n_factors']} lr={parameters['learning_rate']:g} "
            f"reg_factors={parameters['factor_regularization']:g} "
            f"reg_user_biases={parameters['user_bias_regularization']:g} reg_item_biases={parameters['item_bias_regularization']:g} "
            f"batch={parameters['ratings_per_batch']} seed={parameters['seed']}")


def run_one_config(config: dict) -> tuple[dict, list[dict]]:
    parameters = parameters_of_config(config)
    split_seed = int(config.get("split_seed", 0))
    task = task_for_split_seed(split_seed)

    # 1) train with early stopping on the validation split; the refit on 100 % is skipped
    #    -> it cannot change the validation score and would double the run time
    start_time = time.perf_counter()
    with open(LOG_FILE, "a", encoding="utf-8", buffering=1) as log_file, redirect_stdout(log_file):
        print(f"--- {config['config_id']}: {describe_parameters(parameters)} split_seed={split_seed}")
        model_result = assignment1.train_latent_factor_model(task, refit_on_train_and_valid=False, **parameters)
    seconds_total = time.perf_counter() - start_time

    # 2) scores from the epoch history (epoch 0 = closed-form biases only, before any factor training)
    epoch_history = model_result.history
    best_epoch_entry = min(epoch_history, key=lambda epoch_entry: epoch_entry["valid_mse"])
    epochs_run = len(epoch_history) - 1

    result_row = {
        "config_id": config["config_id"],
        "round": config.get("round", ""),
        "status": "ok",
        "model": "svdpp" if parameters["use_implicit_feedback"] else "svd",
        **parameters,
        "split_seed": split_seed,
        "best_valid_mse": f"{best_epoch_entry['valid_mse']:.6f}",
        "best_epoch": best_epoch_entry["epoch"],
        "epochs_run": epochs_run,
        "train_mse_at_best_epoch": f"{best_epoch_entry['train_mse']:.6f}",
        "bias_only_valid_mse": f"{epoch_history[0]['valid_mse']:.6f}",
        "seconds_total": f"{seconds_total:.1f}",
        "seconds_per_epoch": f"{seconds_total / max(epochs_run, 1):.2f}",
        "device": str(DEVICE),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": config.get("note", ""),
        "error": "",
    }
    epoch_rows = [{"config_id": config["config_id"], **epoch_entry} for epoch_entry in epoch_history]

    # 3) free the model before the next run (GPU memory)
    del model_result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result_row, epoch_rows


def failed_result_row(config: dict, error: Exception) -> dict:
    parameters = parameters_of_config(config)
    return {"config_id": config["config_id"], "round": config.get("round", ""), "status": "failed",
            "model": "svdpp" if parameters["use_implicit_feedback"] else "svd", **parameters,
            "split_seed": config.get("split_seed", 0), "device": str(DEVICE),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "note": config.get("note", ""),
            "error": f"{type(error).__name__}: {error}"[:300]}


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# The sweep loop
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Runs every config in sweep_queue.json that has no row in sweep_results.csv yet, in queue order.
# The queue is re-read after every run, so new rounds can be added while this keeps running.
# Stops when the queue says "finished": true, or after max_idle_hours with nothing to do.
# Interrupt the cell to stop early: finished runs are already saved, the current one is lost.
def run_sweep_queue(poll_seconds: int = 30, max_idle_hours: float = 2.0):
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    announce(f"sweep worker started | device {DEVICE} ({gpu_name}) | torch {torch.__version__}")
    convert_results_file_to_split_bias_regularization()
    keep_computer_awake(True)
    idle_since = None

    try:
        while True:
            # 1) what is left to do: queue entries without a result row
            queue = read_queue()
            finished_config_ids = {row["config_id"] for row in read_result_rows()}
            pending_configs = [config for config in queue.get("configs", []) if config["config_id"] not in finished_config_ids]

            # 2) run the next config and save its row immediately (a crash later loses nothing)
            if pending_configs:
                idle_since = None
                config = pending_configs[0]
                announce(f"start {config['config_id']} ({len(pending_configs)} pending): {describe_parameters(parameters_of_config(config))}")
                try:
                    result_row, epoch_rows = run_one_config(config)
                except Exception as error:
                    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
                        traceback.print_exc(file=log_file)
                    result_row, epoch_rows = failed_result_row(config, error), []
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                append_csv_rows(RESULTS_FILE, RESULT_COLUMNS, [result_row])
                append_csv_rows(EPOCH_HISTORY_FILE, EPOCH_HISTORY_COLUMNS, epoch_rows)

                if result_row["status"] == "ok":
                    announce(f"done  {config['config_id']}: best valid MSE {result_row['best_valid_mse']} at epoch "
                             f"{result_row['best_epoch']} of {result_row['epochs_run']} ({result_row['seconds_total']} s)")
                else:
                    announce(f"FAILED {config['config_id']}: {result_row['error']}")
                continue

            # 3) nothing pending: stop if the queue is closed, otherwise wait for the next round
            if queue.get("finished"):
                announce("queue marked as finished -> sweep worker stops")
                break
            idle_since = idle_since or time.time()
            if time.time() - idle_since > max_idle_hours * 3600:
                announce(f"nothing new for {max_idle_hours} h -> sweep worker stops")
                break
            time.sleep(poll_seconds)
    finally:
        keep_computer_awake(False)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Overview of the results so far
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def print_best_results(top_n: int = 20, model: str | None = None):
    convert_results_file_to_split_bias_regularization()
    finished_rows = [row for row in read_result_rows()
                     if row["status"] == "ok" and (model is None or row["model"] == model)]
    finished_rows.sort(key=lambda row: float(row["best_valid_mse"]))

    print(f"{len(finished_rows)} finished runs" + (f" ({model})" if model else ""))
    print(f"{'config_id':<14}{'model':<7}{'k':>5}{'lr':>9}{'reg_f':>8}{'reg_bu':>7}{'reg_bi':>7}{'batch':>7}{'split':>6}"
          f"{'valid MSE':>11}{'epoch':>7}{'s':>8}")
    for row in finished_rows[:top_n]:
        print(f"{row['config_id']:<14}{row['model']:<7}{row['n_factors']:>5}{float(row['learning_rate']):>9g}"
              f"{float(row['factor_regularization']):>8g}{float(row['user_bias_regularization']):>7g}"
              f"{float(row['item_bias_regularization']):>7g}{row['ratings_per_batch']:>7}"
              f"{row['split_seed']:>6}{row['best_valid_mse']:>11}{row['best_epoch']:>7}{float(row['seconds_total']):>8.0f}")
