from dataclasses import replace

import myfm
import numpy as np
import scipy.sparse as sp

from lib.config import RATING_MAX, RATING_MIN
from lib.custom_types import BayesianFactorizationMachineConfig, ModelResult, RatingData, RatingTask
from lib.data_loading import known_interaction_pairs
from lib.helper_functioins import log, mse, sparse_matrix_from_entries
from lib.model_storage import load_model_result, save_model_result
from lib.temporal_features import CALENDAR_FIELD_SIZES, calendar_field_rows, first_and_last_year, require_test_timestamps

# Bayesian factorization machine fitted by Gibbs sampling (Freudenthaler, Schmidt-Thieme & Rendle 2011; Rendle 2012,
# "Factorization Machines with libFM"), implemented by myFM (https://myfm.readthedocs.io):
#   r = mu + sum_f w_f x_f + sum_{f < g} <v_f, v_g> x_f x_g          (the factorization machine, PML Eq. 6.2)
#   the parameters are not one optimum: Gibbs sampling draws them from their posterior, the prediction is the mean over
#   the kept samples; every feature group gets its own priors, which are sampled too -> no learning rate, no lambda to
#   tune and no early stopping (Rendle et al. 2019, "On the Difficulty of Evaluating Baselines": well-run Bayesian
#   factorization models are among the strongest rating predictors)
# features, one prior group each: user | the user's implicit feedback N(u) | item | the item's implicit feedback N(i)
#   N(u) = the items of u, each weighted |N(u)|^-1/2 (as in SVD++); N(i) = the users of i, each weighted |N(i)|^-1/2
# time-aware variant (Task 3, config.calendar_fields): + one-hot calendar fields of the rating's timestamp, one prior group
#   per field, as in the time-aware factorization machine (lib/models/time_factorization_machine.py, lib/temporal_features.py)
#   -> <v_item, v_month> = a book's seasonal popularity, <v_item, v_year> = how its ratings drift over the years, ...
# user periods (config.user_period_days, Task 3): + one-hot (user, day // period) for every period length, one prior group
#   each -> a bias and a factor vector for the user's day / month of the rating (the day-specific user bias b_ut of
#   timeSVD++, Koren 2009, here also interacting with the book); only the periods with a training rating get a column,
#   an asked row in a period without one gets none (= no period effect)
# relation blocks (Rendle 2013, "Scaling Factorization Machines to Relational Data"): the features of a user / an item are
# stored once, every rating only points at its user's and its item's row -> N(u) is not copied into every rating;
# the calendar fields differ per rating -> they are the direct feature matrix X (myFM orders the features X, then the blocks)
# runs on the CPU (C++, one core); myFM shows a progress bar per fit
# the validation and test pairs are predicted while sampling (posterior_mean_predictions) -> no model to predict later


# (rows, rows [+ partners]): one-hot row id [| the row's implicit feedback: its partners, each weighted |partners|^-1/2]
def feature_block(rows: np.ndarray, partners: np.ndarray, n_rows: int, n_partners: int, use_implicit_feedback: bool) -> sp.csr_matrix:
    one_hot_ids = sp.identity(n_rows, dtype=np.float64, format="csr")
    if not use_implicit_feedback:
        return one_hot_ids
    partners_per_row = np.bincount(rows, minlength=n_rows)
    implicit_feedback = sparse_matrix_from_entries(rows, partners, n_rows, n_partners, values=1.0 / np.sqrt(partners_per_row[rows]))
    return sp.hstack([one_hot_ids, implicit_feedback], format="csr", dtype=np.float64)


# number of values of every calendar field; the year field covers the years of the training ratings
def calendar_field_sizes(calendar_fields: tuple[str, ...], year_range: tuple[int, int]) -> list[int]:
    first_year, last_year = year_range
    field_sizes = {**CALENDAR_FIELD_SIZES, "year": last_year - first_year + 1}
    return [field_sizes[name] for name in calendar_fields]


# direct features of every rating / asked pair: (n, sum of field sizes) one-hot calendar fields, one 1 per field;
# no calendar fields (Task 1) -> no columns (every feature comes from the relation blocks)
def direct_features(n_rows: int, timestamps: np.ndarray | None, calendar_fields: tuple[str, ...],
                    year_range: tuple[int, int] | None) -> sp.csr_matrix:
    if not calendar_fields:
        return sp.csr_matrix((n_rows, 0), dtype=np.float64)
    field_sizes = calendar_field_sizes(calendar_fields, year_range)
    column_offsets = np.concatenate([[0], np.cumsum(field_sizes)[:-1]])
    columns = calendar_field_rows(timestamps, calendar_fields, *year_range) + column_offsets    # (n, fields)
    n_fields = len(calendar_fields)
    return sp.csr_matrix((np.ones(n_rows * n_fields), columns.ravel(), np.arange(0, n_rows * n_fields + 1, n_fields)),
                         shape=(n_rows, sum(field_sizes)))


# key of the (user, period) of every row: user * USER_PERIOD_KEY_STRIDE + day // period_days (days < 2^32 -> no overlap)
USER_PERIOD_KEY_STRIDE = 1 << 32


def user_period_keys(user_rows: np.ndarray, days: np.ndarray, period_days: int) -> np.ndarray:
    return user_rows.astype(np.int64) * USER_PERIOD_KEY_STRIDE + days.astype(np.int64) // period_days


# the sorted (user, period) keys with a training rating, one array per period length = the columns of the user-period groups
def known_user_periods(training_ratings: RatingData, user_period_days: tuple[int, ...]) -> list[np.ndarray]:
    return [np.unique(user_period_keys(training_ratings.user_rows, training_ratings.days, period_days))
            for period_days in user_period_days]


# (n, sum of known periods) one-hot (user, period) of every row; a row whose period has no training rating gets no 1 there
def user_period_features(user_rows: np.ndarray, days: np.ndarray, user_period_days: tuple[int, ...],
                         known_periods: list[np.ndarray]) -> sp.csr_matrix:
    blocks = []
    for period_days, keys in zip(user_period_days, known_periods):
        row_keys = user_period_keys(user_rows, days, period_days)
        columns = np.minimum(np.searchsorted(keys, row_keys), len(keys) - 1)
        is_known = keys[columns] == row_keys
        blocks.append(sp.csr_matrix((np.ones(is_known.sum()), (np.flatnonzero(is_known), columns[is_known])),
                                    shape=(len(user_rows), len(keys))))
    return sp.hstack(blocks, format="csr", dtype=np.float64)


# Gibbs sampling on training_ratings -> the posterior mean prediction of every asked row set (e.g. [validation, test rows])
# asked_rows: (user rows, item rows, timestamps or None) per set; implicit_feedback_pairs = the (user, item) pairs of
# N(u) / N(i): the ratings it trains on, or every known pair
# every kept draw predicts the asked rows and adds them to a running sum; the draws themselves are not kept: myFM would
# store each one in RAM (features x (rank + 1) float64: ~43 MB at rank 32 -> ~7.8 GB for 180 draws, more for larger ranks)
def posterior_mean_predictions(training_ratings: RatingData, implicit_feedback_pairs: RatingData, n_user_rows: int,
                               n_item_rows: int, config: BayesianFactorizationMachineConfig,
                               asked_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]]) -> list[np.ndarray]:

    # 1) one feature row per user and per item (+ the size of every prior group, in column order: X first, then the blocks)
    user_rows, item_rows = implicit_feedback_pairs.user_rows, implicit_feedback_pairs.item_rows
    user_features = feature_block(user_rows, item_rows, n_user_rows, n_item_rows, config.use_implicit_feedback)
    item_features = feature_block(item_rows, user_rows, n_item_rows, n_user_rows, config.use_implicit_feedback)
    year_range = first_and_last_year(training_ratings.timestamps) if config.calendar_fields else None
    known_periods = known_user_periods(training_ratings, config.user_period_days) if config.user_period_days else []
    group_shapes = ((calendar_field_sizes(config.calendar_fields, year_range) if config.calendar_fields else [])
                    + [len(keys) for keys in known_periods]
                    + ([n_user_rows, n_item_rows, n_item_rows, n_user_rows] if config.use_implicit_feedback
                       else [n_user_rows, n_item_rows]))

    # direct features X of every rating / asked row: calendar fields | user periods
    def row_features(users: np.ndarray, timestamps: np.ndarray | None, days: np.ndarray | None) -> sp.csr_matrix:
        calendar = direct_features(len(users), timestamps, config.calendar_fields, year_range)
        if not config.user_period_days:
            return calendar
        return sp.hstack([calendar, user_period_features(users, days, config.user_period_days, known_periods)],
                         format="csr", dtype=np.float64)

    # every rating / asked row only points at its user's and its item's feature row
    def relation_blocks(users: np.ndarray, items: np.ndarray) -> list:
        return [myfm.RelationBlock(users, user_features), myfm.RelationBlock(items, item_features)]

    # 2) called by myFM after every draw (iteration 0, 1, ...): the draws after the burn-in add their predictions
    asked_inputs = [(row_features(users, timestamps, days), relation_blocks(users, items))
                    for users, items, timestamps, days in asked_rows]
    prediction_sums = [np.zeros(len(users)) for users, *_ in asked_rows]

    def add_kept_draw(iteration, draw, hyperparameters, history):
        if iteration >= config.n_burn_in_iterations:
            for prediction_sum, (features, blocks) in zip(prediction_sums, asked_inputs):
                prediction_sum += draw.predict_score(features, blocks)
        return False, None                  # never stop early, no progress message

    # 3) Gibbs sampling: burn-in draws, then n_kept_samples draws (myFM itself keeps only its last draw)
    sampler = myfm.MyFMRegressor(rank=config.rank, random_seed=config.seed)
    sampler.fit(row_features(training_ratings.user_rows, training_ratings.timestamps, training_ratings.days),
                training_ratings.ratings,
                X_rel=relation_blocks(training_ratings.user_rows, training_ratings.item_rows),
                n_iter=config.n_burn_in_iterations + config.n_kept_samples, n_kept_samples=1,
                group_shapes=group_shapes, callback=add_kept_draw)
    return [np.clip(prediction_sum / config.n_kept_samples, RATING_MIN, RATING_MAX).astype(np.float32)
            for prediction_sum in prediction_sums]


def train_bayesian_factorization_machine(task: RatingTask, config: BayesianFactorizationMachineConfig) -> ModelResult:
    # n_chains > 1: independent chains with seeds seed, seed + 1, ... averaged -> less sampling noise (a longer single chain
    # made Task 1 worse: 30 + 380 draws 0.55708 vs 20 + 180 draws 0.54871)
    if config.n_chains > 1:
        chains = [train_bayesian_factorization_machine(task, replace(config, seed=config.seed + chain, n_chains=1))
                  for chain in range(config.n_chains)]
        return ModelResult(config, [], np.mean([chain.valid_predictions for chain in chains], axis=0),
                           np.mean([chain.test_predictions for chain in chains], axis=0))
    if config.calendar_fields or config.user_period_days:
        require_test_timestamps(task, "time-aware Bayesian factorization machine")
    validation = task.validation_data
    validation_rows = (validation.user_rows, validation.item_rows, validation.timestamps, validation.days)
    test_rows = (task.test_user_rows, task.test_item_rows, task.test_timestamps, task.test_days)

    # N(u) / N(i): the ratings the model trains on, or every known pair of the training + test file (ratings never used)
    def implicit_feedback_pairs(training_ratings: RatingData) -> RatingData:
        return known_interaction_pairs(task) if config.use_test_pairs_as_implicit_feedback else training_ratings

    # 1) sampled on the training split -> validation predictions (input of the blender) [+ test predictions without refit]
    asked_rows = [validation_rows] + ([] if config.refit_on_train_and_valid else [test_rows])
    predictions = posterior_mean_predictions(task.training_data, implicit_feedback_pairs(task.training_data),
                                             task.n_user_rows, task.n_item_rows, config, asked_rows)
    valid_predictions = predictions[0]
    calendar_text = (f", calendar fields {'/'.join(config.calendar_fields)}" if config.calendar_fields else "") + \
                    (f", user periods {'/'.join(map(str, config.user_period_days))} days" if config.user_period_days else "")
    log(f"Bayesian factorization machine | rank {config.rank}, {config.n_burn_in_iterations} + {config.n_kept_samples} draws"
        f"{calendar_text} | valid MSE {mse(valid_predictions, validation.ratings):.4f}")

    # 2) optional: sampled again on 100 % of the training file -> test predictions
    if config.refit_on_train_and_valid:
        [test_predictions] = posterior_mean_predictions(task.training_and_validation_data,
                                                        implicit_feedback_pairs(task.training_and_validation_data),
                                                        task.n_user_rows, task.n_item_rows, config, [test_rows])
    else:
        test_predictions = predictions[1]

    # no model object to store (the draws are not kept) -> the config records how the predictions were made
    return ModelResult(config, [], valid_predictions, test_predictions)


def store_bayesian_factorization_machine(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "bayesian_factorization_machine")


def load_bayesian_factorization_machine(task_name: str) -> ModelResult:
    return load_model_result(task_name, "bayesian_factorization_machine")


# the time-aware variant (Task 3) is stored under its own name
def store_time_bayesian_factorization_machine(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "time_bayesian_factorization_machine")


def load_time_bayesian_factorization_machine(task_name: str) -> ModelResult:
    return load_model_result(task_name, "time_bayesian_factorization_machine")
