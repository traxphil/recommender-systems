import numpy as np
import torch
import torch.nn as nn

from lib.config import DEVICE
from lib.custom_types import FactorizationMachineConfig, ModelResult, RatingData, RatingTask
from lib.helper_functioins import fit_closed_form_biases
from lib.model_storage import load_model_result, save_model_result
from lib.torch_trainer import fit_refit_and_predict

# Factorization machine (Rendle 2010; PML Sec. 6.1), one active feature per field:
#   fields: user | item | user activity bucket | item popularity bucket
#   r = mu + sum_f w_f + sum_{f < g} <v_f, v_g>        (all pairwise interactions of the active features)
# user x item is the SVD term; user x item bucket = a user's taste for popular vs. niche books, etc.
# no time -> Task 1; the time-aware version with calendar fields (Task 3): time_factorization_machine.py
# same interface as LatentFactorModel (forward / l2_penalty) -> trained with the functions of lib/torch_trainer.py


class FactorizationMachine(nn.Module):

    # no time terms -> lib/torch_trainer.py only passes (user_rows, item_rows); the time-aware version sets uses_timestamps = True
    uses_days = False
    uses_timestamps = False

    # Parameters (one shared table, fields at consecutive offsets):
    # feature_weights           w_f     (U+1 + I+1 + 2 * buckets [+ extra fields], 1)
    # feature_factors           v_f     (U+1 + I+1 + 2 * buckets [+ extra fields], n_factors)
    # extra_field_sizes: fields a subclass appends after the four base fields (the calendar fields of the time-aware version)

    def __init__(self, n_user_rows, n_item_rows, user_activity_buckets, item_popularity_buckets, n_activity_buckets,
                 n_factors, global_mean_rating, initial_user_biases, initial_item_biases, initial_factor_std=0.01,
                 extra_field_sizes: tuple[int, ...] = ()):

        super().__init__()
        self.global_mean_rating = global_mean_rating

        # number of features of every field -> each field's features start where the fields before it end
        field_sizes = [n_user_rows, n_item_rows, n_activity_buckets, n_activity_buckets, *extra_field_sizes]
        self.register_buffer("field_offsets", torch.as_tensor(np.cumsum([0, *field_sizes[:-1]]), dtype=torch.long))
        self.item_offset = n_user_rows
        n_features = sum(field_sizes)

        self.feature_weights = nn.Embedding(n_features, 1)
        self.feature_factors = nn.Embedding(n_features, n_factors)

        # user / item weights start at the closed-form biases, bucket weights at 0 -> epoch 0 = the bias baseline
        nn.init.zeros_(self.feature_weights.weight)
        nn.init.normal_(self.feature_factors.weight, std=initial_factor_std)
        with torch.no_grad():
            self.feature_weights.weight[:n_user_rows, 0] = torch.as_tensor(initial_user_biases, dtype=torch.float32)
            self.feature_weights.weight[self.item_offset:self.item_offset + n_item_rows, 0] = torch.as_tensor(initial_item_biases, dtype=torch.float32)
            # row 0 of users / items = unknown -> no factors
            self.feature_factors.weight[0] = 0.0
            self.feature_factors.weight[self.item_offset] = 0.0

        # bucket of every user / item row; buffers move to the GPU together with the model
        self.register_buffer("user_bucket_of_row", torch.as_tensor(user_activity_buckets, dtype=torch.long))
        self.register_buffer("item_bucket_of_row", torch.as_tensor(item_popularity_buckets, dtype=torch.long))

    # (B, 4) row of every rating within each field: user, item, user activity bucket, item popularity bucket
    def field_rows(self, user_rows, item_rows):
        return torch.stack([user_rows, item_rows, self.user_bucket_of_row[user_rows], self.item_bucket_of_row[item_rows]], dim=1)

    def forward(self, user_rows, item_rows):
        return self.prediction(self.field_rows(user_rows, item_rows) + self.field_offsets)

    # r = mu + sum_f w_f + sum_{f < g} <v_f, v_g> over the (B, fields) active feature indices of every rating
    def prediction(self, active_features):

        linear_part = self.feature_weights(active_features).sum(dim=(1, 2))

        # sum over pairs f < g of <v_f, v_g> = 1/2 * sum_k ((sum_f v_fk)^2 - sum_f v_fk^2) -> O(fields * k) instead of O(fields^2 * k)
        # Source: Rendle 2010, "Factorization Machines" (ICDM), Lemma 3.1; PML Eq. 6.3
        factors = self.feature_factors(active_features)
        pairwise_part = 0.5 * (factors.sum(dim=1) ** 2 - (factors ** 2).sum(dim=1)).sum(dim=1)

        return self.global_mean_rating + linear_part + pairwise_part

    # same scheme as LatentFactorModel.l2_penalty: user / item parameters weighted by 1 / their number of ratings
    # bucket features are shared by thousands of ratings each -> left unregularised
    # same signature as LatentFactorModel.l2_penalty (shared trainer); the config passes one lambda for both biases
    def l2_penalty(self, user_rows, item_rows, user_penalty_weights, item_penalty_weights,
                   user_bias_regularization, item_bias_regularization, factor_regularization):

        item_features = self.item_offset + item_rows

        # squared norms of the parameters used by each rating
        user_bias_squared = self.feature_weights(user_rows).squeeze(1) ** 2
        item_bias_squared = self.feature_weights(item_features).squeeze(1) ** 2
        user_factor_squared = (self.feature_factors(user_rows) ** 2).sum(dim=1)
        item_factor_squared = (self.feature_factors(item_features) ** 2).sum(dim=1)

        # weight per rating, then average over the batch
        bias_penalty = (user_bias_regularization * user_penalty_weights * user_bias_squared
                        + item_bias_regularization * item_penalty_weights * item_bias_squared)
        factor_penalty = user_penalty_weights * user_factor_squared + item_penalty_weights * item_factor_squared

        return (bias_penalty + factor_regularization * factor_penalty).mean()


# quantile buckets of the number of ratings: bucket 0 = fewest ratings (also unknown rows) ... n_buckets - 1 = most
def activity_buckets(rows: np.ndarray, n_rows: int, n_buckets: int) -> np.ndarray:
    ratings_per_row = np.bincount(rows, minlength=n_rows)
    bucket_edges = np.quantile(ratings_per_row[ratings_per_row > 0], np.linspace(0, 1, n_buckets + 1)[1:-1])
    return np.searchsorted(bucket_edges, ratings_per_row, side="right")


# the keyword arguments of the four base fields, from the ratings the model will train on (shared with the time-aware version)
# weights start at the closed-form bias model of these ratings -> epoch 0 already equals the bias baseline
def base_field_arguments(training_ratings: RatingData, n_user_rows: int, n_item_rows: int, config: FactorizationMachineConfig) -> dict:
    global_mean_rating = float(training_ratings.ratings.mean())
    initial_user_biases, initial_item_biases = fit_closed_form_biases(
        training_ratings.user_rows, training_ratings.item_rows, training_ratings.ratings,
        n_user_rows, n_item_rows, global_mean_rating, config.bias_regularization, config.bias_regularization)

    return dict(n_user_rows=n_user_rows,
                n_item_rows=n_item_rows,
                user_activity_buckets=activity_buckets(training_ratings.user_rows, n_user_rows, config.n_activity_buckets),
                item_popularity_buckets=activity_buckets(training_ratings.item_rows, n_item_rows, config.n_activity_buckets),
                n_activity_buckets=config.n_activity_buckets,
                n_factors=config.n_factors,
                global_mean_rating=global_mean_rating,
                initial_user_biases=initial_user_biases,
                initial_item_biases=initial_item_biases)


# options of lib/torch_trainer.fit_refit_and_predict; one bias lambda for both sides (config.bias_regularization is not split)
def training_options(config: FactorizationMachineConfig) -> dict:
    return dict(max_epochs=config.max_epochs,
                early_stopping_patience=config.early_stopping_patience,
                refit_on_train_and_valid=config.refit_on_train_and_valid,
                learning_rate=config.learning_rate,
                user_bias_regularization=config.bias_regularization,
                item_bias_regularization=config.bias_regularization,
                factor_regularization=config.factor_regularization,
                ratings_per_batch=config.ratings_per_batch,
                seed=config.seed)


def train_factorization_machine(task: RatingTask, config: FactorizationMachineConfig) -> ModelResult:

    def build_model(training_ratings: RatingData) -> FactorizationMachine:
        torch.manual_seed(config.seed)  # identical initial factors for the fit and the refit
        return FactorizationMachine(**base_field_arguments(training_ratings, task.n_user_rows, task.n_item_rows, config)).to(DEVICE)

    # fit with early stopping -> validation predictions -> optional refit on 100 % -> test predictions
    return fit_refit_and_predict(task, build_model, **training_options(config))


def store_factorization_machine(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "factorization_machine")


def load_factorization_machine(task_name: str) -> ModelResult:
    return load_model_result(task_name, "factorization_machine")
