import torch
import torch.nn as nn

from lib.config import DEVICE
from lib.custom_types import ModelResult, RatingData, RatingTask, TimeSVDppConfig
from lib.data_loading import known_interaction_pairs
from lib.model_storage import load_model_result, save_model_result
from lib.models.SVDpp import LatentFactorModel
from lib.models.time_bias_model import TemporalBiases, TimeBiasModel
from lib.torch_trainer import fit_refit_and_predict

# timeSVD++ (Koren 2009, "Collaborative Filtering with Temporal Dynamics" (KDD), time-changing factor model; PML Sec. 7.2.2, Eq. 7.13):
#   r(u,i,t) = mu + b_u + b_i + temporal biases(u, i, t) + q_i . (p_u(t) + |N(u)|^-1/2 * sum_{j in N(u)} y_j)
#   p_u(t)   = p_u + alpha_u^p * dev_u(t)       with use_factor_drift, otherwise p_u(t) = p_u
# = LatentFactorModel (SVD / SVD++) + the TemporalBiases of the time-aware bias model (time_bias_model.py)
# all biases start at the closed-form time-aware bias model -> epoch 0 = that model, the factors learn what is left
# the temporal biases stay fixed by default (Adam overfits these rarely used parameters, see time_bias_model.py)
# PML Sec. 7.2.3: most temporal variation sits in the biases, drifting factors matter less -> factor drift is optional


class TimeSVDpp(LatentFactorModel):

    # lib/torch_trainer.py passes the day of every rating
    uses_days = True

    # Parameters on top of LatentFactorModel:
    # temporal_biases           see TemporalBiases (time_bias_model.py)
    # user_factor_drift         alpha_u^p   (U+1, n_factors)    // only with use_factor_drift

    def __init__(self, time_bias_model: TimeBiasModel, n_user_rows, n_item_rows, n_factors, use_factor_drift: bool,
                 train_temporal_biases: bool, implicit_feedback_ratings: RatingData | None = None):

        fitted_biases = time_bias_model.biases
        super().__init__(n_user_rows, n_item_rows, n_factors, time_bias_model.global_mean_rating, implicit_feedback_ratings,
                         initial_user_biases=fitted_biases.user_biases, initial_item_biases=fitted_biases.item_biases)
        self.temporal_biases = TemporalBiases(time_bias_model.index, fitted_biases, time_bias_model.temporal_bias_config)
        self.train_temporal_biases = train_temporal_biases
        self.temporal_biases.requires_grad_(train_temporal_biases)

        # drift of the user factors starts at 0 -> epoch 0 = SVD(++) without time terms
        self.use_factor_drift = use_factor_drift
        if use_factor_drift:
            self.user_factor_drift = nn.Embedding(n_user_rows, n_factors, padding_idx=0)
            nn.init.zeros_(self.user_factor_drift.weight)

    def forward(self, user_rows, item_rows, days):

        user_vectors = self.user_factors(user_rows)

        if self.uses_implicit_feedback:
            user_vectors = user_vectors + self.implicit_feedback_vectors(user_rows)

        if self.use_factor_drift:
            user_vectors = user_vectors + self.user_factor_drift(user_rows) * self.temporal_biases.user_drift(user_rows, days).unsqueeze(1)

        user_item_affinity = (user_vectors * self.item_factors(item_rows)).sum(dim=1)

        static_biases = self.user_biases(user_rows).squeeze(1) + self.item_biases(item_rows).squeeze(1)

        return self.global_mean_rating + static_biases + self.temporal_biases(user_rows, item_rows, days) + user_item_affinity

    def l2_penalty(self, user_rows, item_rows, days, user_penalty_weights, item_penalty_weights,
                   user_bias_regularization, item_bias_regularization, factor_regularization):

        # SVD(++) part as in LatentFactorModel (+ the temporal biases if they are trained)
        penalty = super().l2_penalty(user_rows, item_rows, user_penalty_weights, item_penalty_weights,
                                     user_bias_regularization, item_bias_regularization, factor_regularization)
        if self.train_temporal_biases:
            penalty = penalty + self.temporal_biases.l2_penalty(user_rows, item_rows, days, user_penalty_weights)

        # the factor drift is regularised like p_u: through the ratings of user u
        if self.use_factor_drift:
            drift_squared = (self.user_factor_drift(user_rows) ** 2).sum(dim=1)
            penalty = penalty + factor_regularization * (user_penalty_weights * drift_squared).mean()

        return penalty


def train_time_svdpp(task: RatingTask, config: TimeSVDppConfig) -> ModelResult:

    def build_model(training_ratings: RatingData) -> TimeSVDpp:
        torch.manual_seed(config.seed)  # identical initial factors for the fit and the refit

        # all biases start at the closed-form time-aware bias model of these ratings
        time_bias_model = TimeBiasModel(training_ratings, task.n_user_rows, task.n_item_rows, config.time_biases)

        # SVD++ only: N(u) = items of the ratings this model trains on, or with config.use_test_pairs_as_implicit_feedback
        # every known pair of the training + test file (only which pairs exist, never their ratings)
        implicit_feedback_ratings = known_interaction_pairs(task) if config.use_test_pairs_as_implicit_feedback else training_ratings
        return TimeSVDpp(time_bias_model, task.n_user_rows, task.n_item_rows, config.n_factors,
                         use_factor_drift=config.use_factor_drift,
                         train_temporal_biases=config.train_temporal_biases,
                         implicit_feedback_ratings=implicit_feedback_ratings if config.use_implicit_feedback else None).to(DEVICE)

    return fit_refit_and_predict(task, build_model,
                                 max_epochs=config.max_epochs,
                                 early_stopping_patience=config.early_stopping_patience,
                                 refit_on_train_and_valid=config.refit_on_train_and_valid,
                                 learning_rate=config.learning_rate,
                                 user_bias_regularization=config.bias_regularization,   # one lambda for both (not split here)
                                 item_bias_regularization=config.bias_regularization,
                                 factor_regularization=config.factor_regularization,
                                 ratings_per_batch=config.ratings_per_batch,
                                 seed=config.seed)


def store_time_svdpp(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "time_svdpp")


def load_time_svdpp(task_name: str) -> ModelResult:
    return load_model_result(task_name, "time_svdpp")
