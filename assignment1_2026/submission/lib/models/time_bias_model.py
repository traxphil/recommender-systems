from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

from lib.config import RATING_MAX, RATING_MIN
from lib.custom_types import ModelResult, RatingData, RatingTask, TemporalBiasConfig, TimeBiasModelConfig
from lib.helper_functioins import log, mse
from lib.model_storage import load_model_result, save_model_result
from lib.temporal_features import KEY_STRIDE, require_test_days

# Time-aware bias model (PML Sec. 7.2.2, Eq. 7.8 - 7.12; Koren 2009, "Collaborative Filtering with Temporal Dynamics" (KDD),
# time-changing baseline predictors):
#   r(u,i,t) = mu + b_u + b_i + b_bin(t) + b_{i,bin(t)} + alpha_u * dev_u(t) + sum_s b_{u,period_s(t)}
#   b_bin(t), b_{i,bin(t)}  slow drift of the overall level / of an item's popularity (PML Eq. 7.9)
#   alpha_u * dev_u(t)      a user's ratings drift up or down over their rating lifetime (PML Eq. 7.11, 7.12)
#   b_{u,period(t)}         what is special about the user on that day / in that week / month / ... (Koren's per-day bias
#                           b_{u,t}, here for several period lengths)
# the Task 3 test rows lie inside the training period -> period biases learned from the user's other ratings in the same
# period apply to the test rows (interpolation, not forecasting)
#
# fitted in closed form (alternating exact updates, as fit_closed_form_biases): most of these parameters belong to 1 - 3
# ratings, and Adam moves every parameter by about the same step whatever its gradient -> with Adam they overfit after
# one epoch for any regularization; the exact updates respect the regularization
# TemporalBiases (PyTorch version of the same terms) lets timeSVD++ (time_svdpp.py) start from this fit


class TemporalIndex:

    # which time bin / item-bin row / (user, period) row a rating falls into, built from the training ratings
    def __init__(self, training_ratings: RatingData, n_user_rows: int, n_item_rows: int, config: TemporalBiasConfig):
        self.n_time_bins = config.n_time_bins
        self.n_item_rows = n_item_rows
        self.user_period_lengths_in_days = config.user_period_lengths_in_days
        self.drift_exponent = config.drift_exponent
        user_rows, item_rows, days = training_ratings.user_rows, training_ratings.item_rows, training_ratings.days

        # time bins = quantiles of the training days -> about the same number of ratings per bin
        # edges are whole days ("lower"), so numpy and torch.bucketize put every day into the same bin
        self.time_bin_edges = np.quantile(days, np.linspace(0, 1, config.n_time_bins + 1)[1:-1], method="lower").astype(np.int64)
        self.ratings_per_item_time_bin = np.bincount(self.item_time_bin_rows(item_rows, days), minlength=n_item_rows * self.n_time_bins)

        # dev_u(t) is measured from the user's mean rating day t_u
        ratings_per_user = np.bincount(user_rows, minlength=n_user_rows)
        self.mean_day_of_user = np.bincount(user_rows, weights=days, minlength=n_user_rows) / np.maximum(ratings_per_user, 1)

        # (user, period) pairs with training ratings, sorted by key user * KEY_STRIDE + period -> binary search
        self.period_pair_keys, self.ratings_per_period_pair = [], []
        for period_length in config.user_period_lengths_in_days:
            pair_keys, ratings_per_pair = np.unique(user_rows.astype(np.int64) * KEY_STRIDE + days // period_length, return_counts=True)
            self.period_pair_keys.append(pair_keys)
            self.ratings_per_period_pair.append(ratings_per_pair)

    def time_bins(self, days: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.time_bin_edges, days, side="right")

    def item_time_bin_rows(self, item_rows: np.ndarray, days: np.ndarray) -> np.ndarray:
        return item_rows * self.n_time_bins + self.time_bins(days)

    # row of every (user, period) in its bias table / 0 = the user has no training rating in that period -> bias 0
    def period_pair_rows(self, period_number: int, user_rows: np.ndarray, days: np.ndarray) -> np.ndarray:
        pair_keys = self.period_pair_keys[period_number]
        query_keys = user_rows.astype(np.int64) * KEY_STRIDE + days // self.user_period_lengths_in_days[period_number]
        positions = np.minimum(np.searchsorted(pair_keys, query_keys), len(pair_keys) - 1)
        return np.where(pair_keys[positions] == query_keys, positions + 1, 0)

    # dev_u(t) = sign(t - t_u) * |t - t_u|^exponent, t in days (PML Eq. 7.11; Koren 2009 uses the exponent 0.4)
    def user_drift(self, user_rows: np.ndarray, days: np.ndarray) -> np.ndarray:
        days_from_user_mean = days - self.mean_day_of_user[user_rows]
        return np.sign(days_from_user_mean) * np.abs(days_from_user_mean) ** self.drift_exponent


class TimeAwareBiases(NamedTuple):
    user_biases: np.ndarray                 # b_u           (U+1,)
    item_biases: np.ndarray                 # b_i           (I+1,)
    time_bin_biases: np.ndarray             # b_bin         (n_bins,)
    item_time_bin_biases: np.ndarray        # b_{i,bin}     ((I+1) * n_bins,)      // row = item * n_bins + bin
    user_drift_slopes: np.ndarray           # alpha_u       (U+1,)
    user_period_biases: list[np.ndarray]    # b_{u,period}  (#pairs + 1,) per period length, row 0 = 0


# alternating exact updates of every bias table: table row = sum of residuals / (number of ratings + regularization),
# the residual leaving out exactly this table; alpha_u = ridge regression of the user's residuals on dev_u(t)
# Source: Koren 2008, "Factorization Meets the Neighborhood" (KDD), Sec. 2.1 (baseline estimates by alternating updates)
def fit_time_aware_biases(rating_data: RatingData, index: TemporalIndex, n_user_rows: int, n_item_rows: int,
                          global_mean_rating: float, config: TimeBiasModelConfig) -> TimeAwareBiases:

    user_rows, item_rows, ratings, days = rating_data.user_rows, rating_data.item_rows, rating_data.ratings, rating_data.days
    temporal = config.temporal_biases

    # table rows of every training rating: (rows, number of table rows, regularization) per bias table
    bias_tables = [(item_rows, n_item_rows, config.bias_regularization),
                   (user_rows, n_user_rows, config.bias_regularization),
                   (index.time_bins(days), temporal.n_time_bins, config.bias_regularization),
                   (index.item_time_bin_rows(item_rows, days), n_item_rows * temporal.n_time_bins, temporal.time_bias_regularization)]
    for period_number, pair_keys in enumerate(index.period_pair_keys):
        bias_tables.append((index.period_pair_rows(period_number, user_rows, days), len(pair_keys) + 1, temporal.time_bias_regularization))

    table_values = [np.zeros(n_table_rows) for _, n_table_rows, _ in bias_tables]
    drift = index.user_drift(user_rows, days)
    drift_slopes = np.zeros(n_user_rows)
    predictions = np.full(len(ratings), global_mean_rating)

    for _ in range(config.n_iterations):
        for table_number, (rows, n_table_rows, regularization) in enumerate(bias_tables):
            residuals = ratings - predictions + table_values[table_number][rows]
            new_values = np.bincount(rows, weights=residuals, minlength=n_table_rows) / (np.bincount(rows, minlength=n_table_rows) + regularization)
            predictions += new_values[rows] - table_values[table_number][rows]
            table_values[table_number] = new_values

        residuals = ratings - predictions + drift_slopes[user_rows] * drift
        new_slopes = (np.bincount(user_rows, weights=drift * residuals, minlength=n_user_rows)
                      / (np.bincount(user_rows, weights=drift ** 2, minlength=n_user_rows) + temporal.drift_regularization))
        predictions += (new_slopes - drift_slopes)[user_rows] * drift
        drift_slopes = new_slopes

    item_biases, user_biases, time_bin_biases, item_time_bin_biases, *user_period_biases = table_values
    return TimeAwareBiases(user_biases, item_biases, time_bin_biases, item_time_bin_biases, drift_slopes, user_period_biases)


class TimeBiasModel:

    def __init__(self, rating_data: RatingData, n_user_rows: int, n_item_rows: int, config: TimeBiasModelConfig):
        self.global_mean_rating = float(rating_data.ratings.mean())
        self.temporal_bias_config = config.temporal_biases
        self.index = TemporalIndex(rating_data, n_user_rows, n_item_rows, config.temporal_biases)
        self.biases = fit_time_aware_biases(rating_data, self.index, n_user_rows, n_item_rows, self.global_mean_rating, config)

    # unknown users / items / (user, period) pairs have bias 0 -> mu plus whatever part is known
    def predict(self, user_rows: np.ndarray, item_rows: np.ndarray, days: np.ndarray) -> np.ndarray:
        index, biases = self.index, self.biases
        predictions = (self.global_mean_rating + biases.user_biases[user_rows] + biases.item_biases[item_rows]
                       + biases.time_bin_biases[index.time_bins(days)]
                       + biases.item_time_bin_biases[index.item_time_bin_rows(item_rows, days)]
                       + biases.user_drift_slopes[user_rows] * index.user_drift(user_rows, days))
        for period_number, period_biases in enumerate(biases.user_period_biases):
            predictions += period_biases[index.period_pair_rows(period_number, user_rows, days)]
        return np.clip(predictions, RATING_MIN, RATING_MAX).astype(np.float32)


def train_time_bias_model(task: RatingTask, config: TimeBiasModelConfig) -> ModelResult:
    require_test_days(task, "time-aware bias model")

    # 1) fit on the training split -> validation predictions (input of the blender)
    model = TimeBiasModel(task.training_data, task.n_user_rows, task.n_item_rows, config)
    validation = task.validation_data
    valid_predictions = model.predict(validation.user_rows, validation.item_rows, validation.days)
    log(f"time bias model | valid MSE {mse(valid_predictions, validation.ratings):.4f}")

    # 2) refit on 100 % of the training file -> test predictions
    model = TimeBiasModel(task.training_and_validation_data, task.n_user_rows, task.n_item_rows, config)
    test_predictions = model.predict(task.test_user_rows, task.test_item_rows, task.test_days)

    return ModelResult(model, [], valid_predictions, test_predictions)


def store_time_bias_model(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "time_bias_model")


def load_time_bias_model(task_name: str) -> ModelResult:
    return load_model_result(task_name, "time_bias_model")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PyTorch version of the temporal terms (timeSVD++)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class TemporalBiases(nn.Module):

    # Parameters (start at the closed-form TimeAwareBiases):
    # time_bin_biases           b_bin           (n_bins, 1)
    # item_time_bin_biases      b_{i,bin}       ((I+1) * n_bins, 1)
    # user_drift_slopes         alpha_u         (U+1, 1)
    # user_period_biases[s]     b_{u,period_s}  (#pairs + 1, 1)

    def __init__(self, index: TemporalIndex, fitted_biases: TimeAwareBiases, config: TemporalBiasConfig):

        super().__init__()
        self.n_time_bins = index.n_time_bins
        self.user_period_lengths_in_days = index.user_period_lengths_in_days
        self.drift_exponent = index.drift_exponent
        self.time_bias_regularization = config.time_bias_regularization
        self.drift_regularization = config.drift_regularization

        # lookup structures of the TemporalIndex as buffers -> they move to the GPU together with the model
        self.register_buffer("time_bin_edges", torch.as_tensor(index.time_bin_edges, dtype=torch.long))
        self.register_buffer("mean_day_of_user", torch.as_tensor(index.mean_day_of_user, dtype=torch.float32))
        self.register_buffer("item_time_bin_penalty_weights",
                             torch.as_tensor(1.0 / np.maximum(index.ratings_per_item_time_bin, 1), dtype=torch.float32))
        for period_number, (pair_keys, ratings_per_pair) in enumerate(zip(index.period_pair_keys, index.ratings_per_period_pair)):
            self.register_buffer(f"period_pair_keys_{period_number}", torch.as_tensor(pair_keys, dtype=torch.long))
            self.register_buffer(f"period_penalty_weights_{period_number}",
                                 torch.as_tensor(np.concatenate([[0.0], 1.0 / ratings_per_pair]), dtype=torch.float32))

        def table_from(values: np.ndarray) -> nn.Embedding:
            return nn.Embedding.from_pretrained(torch.as_tensor(values, dtype=torch.float32).unsqueeze(1), freeze=False)

        self.time_bin_biases = table_from(fitted_biases.time_bin_biases)
        self.item_time_bin_biases = table_from(fitted_biases.item_time_bin_biases)
        self.user_drift_slopes = table_from(fitted_biases.user_drift_slopes)
        self.user_period_biases = nn.ModuleList([table_from(values) for values in fitted_biases.user_period_biases])

    def time_bins(self, days):
        return torch.bucketize(days, self.time_bin_edges, right=True)

    def user_drift(self, user_rows, days):
        days_from_user_mean = days.float() - self.mean_day_of_user[user_rows]
        return torch.sign(days_from_user_mean) * days_from_user_mean.abs() ** self.drift_exponent

    def period_pair_rows(self, period_number, user_rows, days):
        pair_keys = getattr(self, f"period_pair_keys_{period_number}")
        period_length = self.user_period_lengths_in_days[period_number]
        query_keys = user_rows * KEY_STRIDE + torch.div(days, period_length, rounding_mode="floor")
        positions = torch.searchsorted(pair_keys, query_keys).clamp(max=len(pair_keys) - 1)
        return torch.where(pair_keys[positions] == query_keys, positions + 1, torch.zeros_like(positions))

    def forward(self, user_rows, item_rows, days):

        time_bins = self.time_bins(days)
        temporal_biases = (self.time_bin_biases(time_bins).squeeze(1)
                           + self.item_time_bin_biases(item_rows * self.n_time_bins + time_bins).squeeze(1)
                           + self.user_drift_slopes(user_rows).squeeze(1) * self.user_drift(user_rows, days))

        for period_number, period_biases in enumerate(self.user_period_biases):
            temporal_biases = temporal_biases + period_biases(self.period_pair_rows(period_number, user_rows, days)).squeeze(1)

        return temporal_biases

    # only needed when the temporal biases are trained further (TimeSVDppConfig.train_temporal_biases)
    # same scheme as LatentFactorModel.l2_penalty: a parameter's penalty is spread over the training ratings that use it
    def l2_penalty(self, user_rows, item_rows, days, user_penalty_weights):

        item_time_bin_rows = item_rows * self.n_time_bins + self.time_bins(days)
        time_bias_penalty = (self.item_time_bin_penalty_weights[item_time_bin_rows]
                             * self.item_time_bin_biases(item_time_bin_rows).squeeze(1) ** 2)

        for period_number, period_biases in enumerate(self.user_period_biases):
            pair_rows = self.period_pair_rows(period_number, user_rows, days)
            penalty_weights = getattr(self, f"period_penalty_weights_{period_number}")
            time_bias_penalty = time_bias_penalty + penalty_weights[pair_rows] * period_biases(pair_rows).squeeze(1) ** 2

        drift_penalty = user_penalty_weights * self.user_drift_slopes(user_rows).squeeze(1) ** 2

        return (self.time_bias_regularization * time_bias_penalty + self.drift_regularization * drift_penalty).mean()
