from dataclasses import asdict
from pathlib import Path

import torch

from lib.config import (DEVICE, SPLIT_SEED, VALIDATION_FRACTION,
                        TASK1_BIAS_MODEL, TASK1_FACTORIZATION_MACHINE, TASK1_ITEM_KNN, TASK1_SUBMITTED_BLENDER, TASK1_SVD,
                        TASK1_SVDPP, TASK1_ZERO_MODEL,
                        TASK2_EASE, TASK2_ITEM_COSINE, TASK2_MULT_VAE, TASK2_PAIR_BLENDER,
                        TASK3_BIAS_MODEL, TASK3_ITEM_KNN, TASK3_SUBMITTED_BLENDER, TASK3_SVD, TASK3_SVDPP, TASK3_TIME_BIAS_MODEL,
                        TASK3_TIME_ITEM_KNN, TASK3_TIME_SVDPP, TASK3_TIME_ZERO_MODEL, TASK3_ZERO_MODEL,
                        TASK4_EASE, TASK4_ITEM_COSINE, TASK4_MULT_VAE, TASK4_PAIR_BLENDER, TASK4_TIME_EASE,
                        TASK5_EASE, TASK5_ITEM_COSINE, TASK5_MULT_VAE, TASK5_PAIR_BLENDER, TASK5_SASREC, TASK5_TRANSITION,
                        TASK6_LIGHTGBM_GAP_MODEL, TASK6_LINEAR_GAP_MODEL, TASK6_SUBMITTED_GAP_MODEL)
from lib.custom_types import BlenderConfig, ModelResult, PairChoiceTask, RatingData, RatingTask, TimeGapTask
from lib.data_loading import (load_next_item_task, load_pair_choice_task, load_rating_prediction_task, load_time_gap_task,
                              write_choice_prediction_file, write_prediction_file)
from lib.helper_functioins import fit_closed_form_biases
from lib.models.bias_model import train_bias_model
from lib.models.blender import train_blender
from lib.models.ease_model import train_ease_model, train_plain_and_time_weighted_ease
from lib.models.factorization_machine import train_factorization_machine
from lib.models.gap_regressor import train_gap_regressor, train_median_gap_baseline
from lib.models.item_cosine_model import train_item_cosine_model
from lib.models.item_knn import train_item_knn
from lib.models.mult_vae import train_mult_vae
from lib.models.pair_blender import train_pair_blender
from lib.models.popularity_model import train_popularity_model
from lib.models.sasrec import train_sasrec
from lib.models.SVDpp import LatentFactorModel
from lib.models.time_bias_model import train_time_bias_model
from lib.models.time_item_knn import train_time_item_knn
from lib.models.time_svdpp import train_time_svdpp
from lib.models.time_zero_model import train_time_zero_model
from lib.models.transition_model import train_transition_model
from lib.models.zero_model import train_zero_model
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
                              user_bias_regularization: float = 1.0,   # best closed-form bias model on the Task 1 split
                              item_bias_regularization: float = 5.0,
                              factor_regularization: float = 8,
                              ratings_per_batch: int = 512,
                              max_epochs: int = 1000,
                              early_stopping_patience: int = 6,
                              refit_on_train_and_valid: bool = True,
                              seed: int = 0) -> ModelResult:

    # options shared by the fit and the refit / keyword form, so they can be passed on with **
    training_options = dict(learning_rate=learning_rate,
                            user_bias_regularization=user_bias_regularization,
                            item_bias_regularization=item_bias_regularization,
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
            task.n_user_rows, task.n_item_rows, global_mean_rating, user_bias_regularization, item_bias_regularization)

        # SVD++ only: N(u) = items of the ratings this model trains on
        # -> fit: training split only (validation pairs stay unseen), refit: whole training file; never test pairs
        return LatentFactorModel(
            task.n_user_rows, task.n_item_rows, n_factors,
            global_mean_rating=global_mean_rating,
            implicit_feedback_ratings=training_ratings if use_implicit_feedback else None,
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
    return load_rating_prediction_task(TASK1_TRAINING_FILE, TASK1_TEST_FILE, VALIDATION_FRACTION, SPLIT_SEED)


# one function per sub-model, so each can be trained on its own from the notebook
# hyperparameters live in lib/config.py (Task 1 section), so this file reproduces the submitted predictions
def train_task1_bias_model(task: RatingTask) -> ModelResult:
    return train_bias_model(task, TASK1_BIAS_MODEL)


def train_task1_svd(task: RatingTask) -> ModelResult:
    return train_latent_factor_model(task, **asdict(TASK1_SVD))


def train_task1_svdpp(task: RatingTask) -> ModelResult:
    return train_latent_factor_model(task, **asdict(TASK1_SVDPP))


def train_task1_item_knn(task: RatingTask) -> ModelResult:
    return train_item_knn(task, TASK1_ITEM_KNN)


def train_task1_zero_model(task: RatingTask) -> ModelResult:
    return train_zero_model(task, TASK1_ZERO_MODEL)


def train_task1_factorization_machine(task: RatingTask) -> ModelResult:
    return train_factorization_machine(task, TASK1_FACTORIZATION_MACHINE)


# sub-model name (as in the blender configs and in trained_models/) -> its training function
TASK1_SUB_MODEL_TRAINERS = {
    "bias_model": train_task1_bias_model,
    "svd": train_task1_svd,
    "svdpp": train_task1_svdpp,
    "item_knn": train_task1_item_knn,
    "zero_model": train_task1_zero_model,
    "factorization_machine": train_task1_factorization_machine,
}


# every sub-model the submitted blend uses
def train_task1_models(task: RatingTask) -> dict[str, ModelResult]:
    return {model_name: TASK1_SUB_MODEL_TRAINERS[model_name](task) for model_name in TASK1_SUBMITTED_BLENDER.blended_models}


def train_task1_blender(task: RatingTask, model_results: dict[str, ModelResult],
                        blender_config: BlenderConfig = TASK1_SUBMITTED_BLENDER) -> ModelResult:
    return train_blender(task, model_results, blender_config)


def write_task1_predictions(test_predictions) -> None:
    write_prediction_file(TASK1_TEST_FILE, TASK1_PREDICTION_FILE, test_predictions)


def run_task1():
    task = load_task1()
    model_results = train_task1_models(task)
    # the blended ensemble writes the submitted predictions
    ensemble_result = train_task1_blender(task, model_results)
    write_task1_predictions(ensemble_result.test_predictions)
    return task, model_results, ensemble_result


# #########################################################################################
# Task 2 - interaction prediction: which of two books did the user read?
# sub-models score (user, book): popularity, item cosine similarity to the user's history, EASE (linear autoencoder),
# Mult-VAE (non-linear autoencoder) -> LightGBM pair blender decides every pair
# #########################################################################################

TASK2_TRAINING_FILE = DATA_DIR / "tasks_1_2_train.csv"
TASK2_TEST_FILE = DATA_DIR / "task_2_test.csv"
TASK2_PREDICTION_FILE = OUTPUT_DIR / "task_2_predictions.csv"


# validation pairs = held-out interactions vs. negatives drawn like the test negatives (see load_pair_choice_task)
def load_task2() -> PairChoiceTask:
    return load_pair_choice_task(TASK2_TRAINING_FILE, TASK2_TEST_FILE, VALIDATION_FRACTION, SPLIT_SEED)


def train_task2_popularity_model(task: PairChoiceTask) -> ModelResult:
    return train_popularity_model(task)


def train_task2_item_cosine_model(task: PairChoiceTask) -> ModelResult:
    return train_item_cosine_model(task, TASK2_ITEM_COSINE)


def train_task2_ease_model(task: PairChoiceTask) -> ModelResult:
    return train_ease_model(task, TASK2_EASE)


def train_task2_mult_vae(task: PairChoiceTask) -> ModelResult:
    return train_mult_vae(task, TASK2_MULT_VAE)


# sub-model name (as in the pair blender config and in trained_models/) -> its training function
TASK2_SUB_MODEL_TRAINERS = {
    "popularity": train_task2_popularity_model,
    "item_cosine": train_task2_item_cosine_model,
    "ease": train_task2_ease_model,
    "mult_vae": train_task2_mult_vae,
}


def train_task2_models(task: PairChoiceTask) -> dict[str, ModelResult]:
    return {model_name: TASK2_SUB_MODEL_TRAINERS[model_name](task) for model_name in TASK2_PAIR_BLENDER.combined_models}


def train_task2_blender(task: PairChoiceTask, model_results: dict[str, ModelResult]) -> ModelResult:
    return train_pair_blender(task, model_results, TASK2_PAIR_BLENDER)


# first_item_probabilities = the pair blender's P(item1 is the real one) per test row
def write_task2_predictions(first_item_probabilities) -> None:
    write_choice_prediction_file(TASK2_TEST_FILE, TASK2_PREDICTION_FILE, first_item_probabilities > 0.5)


def run_task2():
    task = load_task2()
    model_results = train_task2_models(task)
    ensemble_result = train_task2_blender(task, model_results)
    write_task2_predictions(ensemble_result.test_predictions)
    return task, model_results, ensemble_result


# #########################################################################################
# Task 3 - rating prediction, timestamp available
# Task 1 models (timestamps ignored) + temporal models: time-aware biases, timeSVD++, time-decayed item kNN,
# temporal zero model -> blended (ridge or LightGBM with time features, see TASK3_SUBMITTED_BLENDER)
# #########################################################################################

TASK3_TRAINING_FILE = DATA_DIR / "tasks_3_4_train.csv"
TASK3_TEST_FILE = DATA_DIR / "task_3_test.csv"
TASK3_PREDICTION_FILE = OUTPUT_DIR / "task_3_predictions.csv"


# the test file has timestamps -> RatingTask.test_days is set (needed by the temporal models)
def load_task3() -> RatingTask:
    return load_rating_prediction_task(TASK3_TRAINING_FILE, TASK3_TEST_FILE, VALIDATION_FRACTION, SPLIT_SEED)


# one function per sub-model, so each can be trained on its own from the notebook
# hyperparameters live in lib/config.py (Task 3 section)
def train_task3_bias_model(task: RatingTask) -> ModelResult:
    return train_bias_model(task, TASK3_BIAS_MODEL)


def train_task3_svd(task: RatingTask) -> ModelResult:
    return train_latent_factor_model(task, **asdict(TASK3_SVD))


def train_task3_svdpp(task: RatingTask) -> ModelResult:
    return train_latent_factor_model(task, **asdict(TASK3_SVDPP))


def train_task3_item_knn(task: RatingTask) -> ModelResult:
    return train_item_knn(task, TASK3_ITEM_KNN)


def train_task3_zero_model(task: RatingTask) -> ModelResult:
    return train_zero_model(task, TASK3_ZERO_MODEL)


def train_task3_time_bias_model(task: RatingTask) -> ModelResult:
    return train_time_bias_model(task, TASK3_TIME_BIAS_MODEL)


def train_task3_time_svdpp(task: RatingTask) -> ModelResult:
    return train_time_svdpp(task, TASK3_TIME_SVDPP)


def train_task3_time_item_knn(task: RatingTask) -> ModelResult:
    return train_time_item_knn(task, TASK3_TIME_ITEM_KNN)


def train_task3_time_zero_model(task: RatingTask) -> ModelResult:
    return train_time_zero_model(task, TASK3_TIME_ZERO_MODEL)


# sub-model name (as in the blender configs and in trained_models/) -> its training function
TASK3_SUB_MODEL_TRAINERS = {
    "bias_model": train_task3_bias_model,
    "svd": train_task3_svd,
    "svdpp": train_task3_svdpp,
    "item_knn": train_task3_item_knn,
    "zero_model": train_task3_zero_model,
    "time_bias_model": train_task3_time_bias_model,
    "time_svdpp": train_task3_time_svdpp,
    "time_item_knn": train_task3_time_item_knn,
    "time_zero_model": train_task3_time_zero_model,
}


# every sub-model the submitted blend uses
def train_task3_models(task: RatingTask) -> dict[str, ModelResult]:
    return {model_name: TASK3_SUB_MODEL_TRAINERS[model_name](task) for model_name in TASK3_SUBMITTED_BLENDER.blended_models}


def train_task3_blender(task: RatingTask, model_results: dict[str, ModelResult],
                        blender_config: BlenderConfig = TASK3_SUBMITTED_BLENDER) -> ModelResult:
    return train_blender(task, model_results, blender_config)


def write_task3_predictions(test_predictions) -> None:
    write_prediction_file(TASK3_TEST_FILE, TASK3_PREDICTION_FILE, test_predictions)


def run_task3():
    task = load_task3()
    model_results = train_task3_models(task)
    # the blended ensemble writes the submitted predictions
    ensemble_result = train_task3_blender(task, model_results)
    write_task3_predictions(ensemble_result.test_predictions)
    return task, model_results, ensemble_result


# #########################################################################################
# Task 4 - interaction prediction, timestamp available
# Task 2 sub-models + EASE with a time-weighted history; the pair blender also sees each book's popularity around t
# and its age at t
# #########################################################################################

TASK4_TRAINING_FILE = DATA_DIR / "tasks_3_4_train.csv"
TASK4_TEST_FILE = DATA_DIR / "task_4_test.csv"
TASK4_PREDICTION_FILE = OUTPUT_DIR / "task_4_predictions.csv"


def load_task4() -> PairChoiceTask:
    return load_pair_choice_task(TASK4_TRAINING_FILE, TASK4_TEST_FILE, VALIDATION_FRACTION, SPLIT_SEED)


def train_task4_popularity_model(task: PairChoiceTask) -> ModelResult:
    return train_popularity_model(task)


def train_task4_item_cosine_model(task: PairChoiceTask) -> ModelResult:
    return train_item_cosine_model(task, TASK4_ITEM_COSINE)


def train_task4_ease_model(task: PairChoiceTask) -> ModelResult:
    return train_ease_model(task, TASK4_EASE)


def train_task4_time_ease_model(task: PairChoiceTask) -> ModelResult:
    return train_ease_model(task, TASK4_TIME_EASE, model_name="time-weighted EASE")


def train_task4_mult_vae(task: PairChoiceTask) -> ModelResult:
    return train_mult_vae(task, TASK4_MULT_VAE)


TASK4_SUB_MODEL_TRAINERS = {
    "popularity": train_task4_popularity_model,
    "item_cosine": train_task4_item_cosine_model,
    "ease": train_task4_ease_model,
    "time_ease": train_task4_time_ease_model,
    "mult_vae": train_task4_mult_vae,
}


def train_task4_models(task: PairChoiceTask) -> dict[str, ModelResult]:
    return {model_name: TASK4_SUB_MODEL_TRAINERS[model_name](task) for model_name in TASK4_PAIR_BLENDER.combined_models}


def train_task4_blender(task: PairChoiceTask, model_results: dict[str, ModelResult]) -> ModelResult:
    return train_pair_blender(task, model_results, TASK4_PAIR_BLENDER)


def write_task4_predictions(first_item_probabilities) -> None:
    write_choice_prediction_file(TASK4_TEST_FILE, TASK4_PREDICTION_FILE, first_item_probabilities > 0.5)


def run_task4():
    task = load_task4()
    model_results = train_task4_models(task)
    ensemble_result = train_task4_blender(task, model_results)
    write_task4_predictions(ensemble_result.test_predictions)
    return task, model_results, ensemble_result


# #########################################################################################
# Task 5 - which book comes next? (the test pair asks for the book after each user's last training interaction)
# Task 4 sub-models + transition model (which book follows which) + SASRec -> LightGBM pair blender
# #########################################################################################

TASK5_TRAINING_FILE = DATA_DIR / "task_5_train.csv"
TASK5_TEST_FILE = DATA_DIR / "task_5_test.csv"
TASK5_PREDICTION_FILE = OUTPUT_DIR / "task_5_predictions.csv"


# validation pairs = each user's last training interaction vs. a popularity-drawn negative ("leave last out")
def load_task5() -> PairChoiceTask:
    return load_next_item_task(TASK5_TRAINING_FILE, TASK5_TEST_FILE, SPLIT_SEED)


def train_task5_popularity_model(task: PairChoiceTask) -> ModelResult:
    return train_popularity_model(task)


def train_task5_item_cosine_model(task: PairChoiceTask) -> ModelResult:
    return train_item_cosine_model(task, TASK5_ITEM_COSINE)


def train_task5_transition_model(task: PairChoiceTask) -> ModelResult:
    return train_transition_model(task, TASK5_TRANSITION)


# plain and time-weighted EASE share one set of weights (50k books in RAM) -> (ease result, time_ease result)
def train_task5_ease_models(task: PairChoiceTask) -> tuple[ModelResult, ModelResult]:
    return train_plain_and_time_weighted_ease(task, TASK5_EASE)


def train_task5_mult_vae(task: PairChoiceTask) -> ModelResult:
    return train_mult_vae(task, TASK5_MULT_VAE)


def train_task5_sasrec(task: PairChoiceTask) -> ModelResult:
    return train_sasrec(task, TASK5_SASREC)


def train_task5_models(task: PairChoiceTask) -> dict[str, ModelResult]:
    ease_result, time_ease_result = train_task5_ease_models(task)
    return {
        "popularity": train_task5_popularity_model(task),
        "item_cosine": train_task5_item_cosine_model(task),
        "transition": train_task5_transition_model(task),
        "ease": ease_result,
        "time_ease": time_ease_result,
        "mult_vae": train_task5_mult_vae(task),
        "sasrec": train_task5_sasrec(task),
    }


def train_task5_blender(task: PairChoiceTask, model_results: dict[str, ModelResult]) -> ModelResult:
    return train_pair_blender(task, model_results, TASK5_PAIR_BLENDER)


def write_task5_predictions(first_item_probabilities) -> None:
    write_choice_prediction_file(TASK5_TEST_FILE, TASK5_PREDICTION_FILE, first_item_probabilities > 0.5)


def run_task5():
    task = load_task5()
    model_results = train_task5_models(task)
    ensemble_result = train_task5_blender(task, model_results)
    write_task5_predictions(ensemble_result.test_predictions)
    return task, model_results, ensemble_result


# #########################################################################################
# Task 6 - time until the user's next interaction (the next book is given)
# per-user median (baseline) | linear median autoregression | LightGBM with L1 loss on log(1 + gap)
# #########################################################################################

TASK6_TRAINING_FILE = DATA_DIR / "task_6_train.csv"
TASK6_TEST_FILE = DATA_DIR / "task_6_test.csv"
TASK6_PREDICTION_FILE = OUTPUT_DIR / "task_6_predictions.csv"


# validation rows = each user's last training interaction; target = its gap to the previous one
def load_task6() -> TimeGapTask:
    return load_time_gap_task(TASK6_TRAINING_FILE, TASK6_TEST_FILE)


def train_task6_median_baseline(task: TimeGapTask) -> ModelResult:
    return train_median_gap_baseline(task)


def train_task6_linear_gap_model(task: TimeGapTask) -> ModelResult:
    return train_gap_regressor(task, TASK6_LINEAR_GAP_MODEL)


def train_task6_lightgbm_gap_model(task: TimeGapTask) -> ModelResult:
    return train_gap_regressor(task, TASK6_LIGHTGBM_GAP_MODEL)


# predicted gaps in seconds, one per test row
def write_task6_predictions(predicted_gaps) -> None:
    write_prediction_file(TASK6_TEST_FILE, TASK6_PREDICTION_FILE, predicted_gaps)


def run_task6():
    task = load_task6()
    gap_model_result = train_gap_regressor(task, TASK6_SUBMITTED_GAP_MODEL)
    write_task6_predictions(gap_model_result.test_predictions)
    return task, gap_model_result


if __name__ == "__main__":
    run_task1()
    run_task2()
    run_task3()
    run_task4()
    run_task5()
    run_task6()
