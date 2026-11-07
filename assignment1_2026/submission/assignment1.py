from pathlib import Path

import torch

from lib.config import DEVICE
from lib.custom_types import ModelResult, RatingData, RatingTask
from lib.data_loading import load_rating_prediction_task, write_prediction_file
from lib.helper_functioins import fit_closed_form_biases, group_by_user
from lib.SVDpp import LatentFactorModel
from lib.torch_trainer import fit_with_early_stopping, predict_ratings, train_for_epochs

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Path Definition for the Assignment
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

SUBMISSION_DIR = Path(__file__).resolve().parent
DATA_DIR = SUBMISSION_DIR.parent / "data"
OUTPUT_DIR = SUBMISSION_DIR / "output"


# #########################################################################################
# Main Pipeline Handlers for the Models
# #########################################################################################

# SVD if use_implicit_feedback=False / SVD++ for True
def train_latent_factor_model(task: RatingTask,
                              use_implicit_feedback: bool = True,
                              n_factors: int = 500,
                              learning_rate: float = 2.5e-4,
                              bias_regularization: float = 3.0,     # best closed-form bias model on the Task 1 split
                              factor_regularization: float = 8,
                              ratings_per_batch: int = 512,
                              max_epochs: int = 1000,
                              early_stopping_patience: int = 20,
                              refit_on_train_and_valid: bool = True,
                              seed: int = 0) -> ModelResult:

    # options shared by the fit and the refit / keyword form, so they can be passed on with **
    training_options = dict(learning_rate=learning_rate,
                            bias_regularization=bias_regularization,
                            factor_regularization=factor_regularization,
                            ratings_per_batch=ratings_per_batch,
                            seed=seed)

    def build_model(training_ratings: RatingData) -> LatentFactorModel:
        torch.manual_seed(seed)  # identical initial factors for the fit and the refit
        global_mean_rating = float(training_ratings.ratings.mean())

        # biases start at the closed-form bias model of these ratings (same regularization as in training)
        # -> epoch 0 already equals the bias baseline, the factors only have to improve on top of it
        initial_user_biases, initial_item_biases = fit_closed_form_biases(
            training_ratings.user_rows, training_ratings.item_rows, training_ratings.ratings,
            task.n_user_rows, task.n_item_rows, global_mean_rating, bias_regularization)

        # SVD++ only: N(u) = items of the ratings this model trains on
        # -> fit: training split only (validation pairs stay unseen), refit: whole training file; never test pairs
        interacted_items_by_user = (group_by_user(training_ratings.user_rows, training_ratings.item_rows, task.n_user_rows)
                                    if use_implicit_feedback else None)

        return LatentFactorModel(
            task.n_user_rows, task.n_item_rows, n_factors,
            global_mean_rating=global_mean_rating,
            interacted_items_by_user=interacted_items_by_user,
            initial_user_biases=initial_user_biases,
            initial_item_biases=initial_item_biases,
        ).to(DEVICE)

    # 1) fit on the training data, early-stop on the validation data
    model = build_model(task.training_data)
    epoch_history = fit_with_early_stopping(model, task.training_data, task.validation_data,
                                            max_epochs=max_epochs,
                                            early_stopping_patience=early_stopping_patience,
                                            **training_options)
    valid_predictions = predict_ratings(model, task.validation_data.user_rows, task.validation_data.item_rows)

    # 2) optional: retrain from scratch on 100 % for the best number of epochs (uses the extra 10 % of data)
    #    best epoch 0 = the factors never helped -> the refit keeps the closed-form biases of all data
    if refit_on_train_and_valid:
        best_epoch = min(epoch_history, key=lambda epoch_entry: epoch_entry["valid_mse"])["epoch"]
        model = build_model(task.training_and_validation_data)
        train_for_epochs(model, task.training_and_validation_data, n_epochs=best_epoch, **training_options)

    test_predictions = predict_ratings(model, task.test_user_rows, task.test_item_rows)
    return ModelResult(model, epoch_history, valid_predictions, test_predictions)


# #########################################################################################
# Task 1 - rating prediction from (user, item) only
# #########################################################################################

TASK1_TRAINING_FILE = DATA_DIR / "tasks_1_2_train.csv"
TASK1_TEST_FILE = DATA_DIR / "task_1_test.csv"
TASK1_PREDICTION_FILE = OUTPUT_DIR / "task_1_predictions.csv"


def load_task1() -> RatingTask:
    return load_rating_prediction_task(TASK1_TRAINING_FILE, TASK1_TEST_FILE)


# final hyperparameters live here, so this file reproduces the submitted predictions
def train_task1_models(task: RatingTask) -> dict[str, ModelResult]:
    return {
        "svd": train_latent_factor_model(task, use_implicit_feedback=False),
        # SVD++ rebuilds N(u) sums for every user in a batch -> bigger batches keep an epoch affordable
        "svdpp": train_latent_factor_model(task, use_implicit_feedback=True, ratings_per_batch=16384, learning_rate=2e-3),
    }


def write_task1_predictions(test_predictions) -> None:
    write_prediction_file(TASK1_TEST_FILE, TASK1_PREDICTION_FILE, test_predictions)


def run_task1():
    task = load_task1()
    model_results = train_task1_models(task)
    # later: replace with the blended ensemble prediction
    write_task1_predictions(model_results["svdpp"].test_predictions)
    return task, model_results


# #########################################################################################
# Task 3 - rating prediction, timestamp available
# for now: same models as Task 1, timestamps ignored (like the baseline)
# #########################################################################################

TASK3_TRAINING_FILE = DATA_DIR / "tasks_3_4_train.csv"
TASK3_TEST_FILE = DATA_DIR / "task_3_test.csv"
TASK3_PREDICTION_FILE = OUTPUT_DIR / "task_3_predictions.csv"


def load_task3() -> RatingTask:
    return load_rating_prediction_task(TASK3_TRAINING_FILE, TASK3_TEST_FILE)


def train_task3_models(task: RatingTask) -> dict[str, ModelResult]:
    return {
        "svd": train_latent_factor_model(task, use_implicit_feedback=False),
        # SVD++ rebuilds N(u) sums for every user in a batch -> bigger batches keep an epoch affordable
        "svdpp": train_latent_factor_model(task, use_implicit_feedback=True, ratings_per_batch=16384, learning_rate=2e-3),
    }


def write_task3_predictions(test_predictions) -> None:
    write_prediction_file(TASK3_TEST_FILE, TASK3_PREDICTION_FILE, test_predictions)


def run_task3():
    task = load_task3()
    model_results = train_task3_models(task)
    write_task3_predictions(model_results["svdpp"].test_predictions)
    return task, model_results


# #########################################################################################
# Tasks 2, 4, 5, 6: to do (baselines still run from the notebook)
# #########################################################################################


if __name__ == "__main__":
    run_task1()
    run_task3()
