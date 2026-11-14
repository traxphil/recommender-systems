import myfm
import numpy as np
import scipy.sparse as sp

from lib.config import RATING_MAX, RATING_MIN
from lib.custom_types import BayesianFactorizationMachineConfig, ModelResult, RatingData, RatingTask
from lib.data_loading import known_interaction_pairs
from lib.helper_functioins import log, mse, sparse_matrix_from_entries
from lib.model_storage import load_model_result, save_model_result

# Bayesian factorization machine fitted by Gibbs sampling (Freudenthaler, Schmidt-Thieme & Rendle 2011; Rendle 2012,
# "Factorization Machines with libFM"), implemented by myFM (https://myfm.readthedocs.io):
#   r = mu + sum_f w_f x_f + sum_{f < g} <v_f, v_g> x_f x_g          (the factorization machine, PML Eq. 6.2)
#   the parameters are not one optimum: Gibbs sampling draws them from their posterior, the prediction is the mean over
#   the kept samples; every feature group gets its own priors, which are sampled too -> no learning rate, no lambda to
#   tune and no early stopping (Rendle et al. 2019, "On the Difficulty of Evaluating Baselines": well-run Bayesian
#   factorization models are among the strongest rating predictors)
# features, one prior group each: user | the user's implicit feedback N(u) | item | the item's implicit feedback N(i)
#   N(u) = the items of u, each weighted |N(u)|^-1/2 (as in SVD++); N(i) = the users of i, each weighted |N(i)|^-1/2
# relation blocks (Rendle 2013, "Scaling Factorization Machines to Relational Data"): the features of a user / an item are
# stored once, every rating only points at its user's and its item's row -> N(u) is not copied into every rating
# runs on the CPU (C++); myFM shows a progress bar per fit


# (rows, rows [+ partners]): one-hot row id [| the row's implicit feedback: its partners, each weighted |partners|^-1/2]
def feature_block(rows: np.ndarray, partners: np.ndarray, n_rows: int, n_partners: int, use_implicit_feedback: bool) -> sp.csr_matrix:
    one_hot_ids = sp.identity(n_rows, dtype=np.float64, format="csr")
    if not use_implicit_feedback:
        return one_hot_ids
    partners_per_row = np.bincount(rows, minlength=n_rows)
    implicit_feedback = sparse_matrix_from_entries(rows, partners, n_rows, n_partners, values=1.0 / np.sqrt(partners_per_row[rows]))
    return sp.hstack([one_hot_ids, implicit_feedback], format="csr", dtype=np.float64)


class BayesianFactorizationMachine:

    # implicit_feedback_pairs = the (user, item) pairs of N(u) / N(i): the ratings it trains on, or every known pair
    def __init__(self, training_ratings: RatingData, implicit_feedback_pairs: RatingData, n_user_rows: int, n_item_rows: int,
                 config: BayesianFactorizationMachineConfig):

        # 1) one feature row per user and per item (+ the size of every prior group, in column order)
        user_rows, item_rows = implicit_feedback_pairs.user_rows, implicit_feedback_pairs.item_rows
        self.user_features = feature_block(user_rows, item_rows, n_user_rows, n_item_rows, config.use_implicit_feedback)
        self.item_features = feature_block(item_rows, user_rows, n_item_rows, n_user_rows, config.use_implicit_feedback)
        group_shapes = ([n_user_rows, n_item_rows, n_item_rows, n_user_rows] if config.use_implicit_feedback
                        else [n_user_rows, n_item_rows])

        # 2) Gibbs sampling: burn-in draws (thrown away), then n_kept_samples draws, averaged for the predictions
        self.sampler = myfm.MyFMRegressor(rank=config.rank, random_seed=config.seed)
        self.sampler.fit(self.no_direct_features(len(training_ratings.ratings)), training_ratings.ratings,
                         X_rel=self.relation_blocks(training_ratings.user_rows, training_ratings.item_rows),
                         n_iter=config.n_burn_in_iterations + config.n_kept_samples, n_kept_samples=config.n_kept_samples,
                         group_shapes=group_shapes)

    # every feature comes from the relation blocks -> the direct feature matrix has no columns
    @staticmethod
    def no_direct_features(n_ratings: int) -> sp.csr_matrix:
        return sp.csr_matrix((n_ratings, 0), dtype=np.float64)

    def relation_blocks(self, user_rows: np.ndarray, item_rows: np.ndarray) -> list:
        return [myfm.RelationBlock(user_rows, self.user_features), myfm.RelationBlock(item_rows, self.item_features)]

    def predict(self, user_rows: np.ndarray, item_rows: np.ndarray) -> np.ndarray:
        predictions = self.sampler.predict(self.no_direct_features(len(user_rows)), self.relation_blocks(user_rows, item_rows))
        return np.clip(predictions, RATING_MIN, RATING_MAX).astype(np.float32)


def train_bayesian_factorization_machine(task: RatingTask, config: BayesianFactorizationMachineConfig) -> ModelResult:

    # N(u) / N(i): the ratings the model trains on, or every known pair of the training + test file (ratings never used)
    def implicit_feedback_pairs(training_ratings: RatingData) -> RatingData:
        return known_interaction_pairs(task) if config.use_test_pairs_as_implicit_feedback else training_ratings

    # 1) sampled on the training split -> validation predictions (input of the blender)
    model = BayesianFactorizationMachine(task.training_data, implicit_feedback_pairs(task.training_data),
                                         task.n_user_rows, task.n_item_rows, config)
    valid_predictions = model.predict(task.validation_data.user_rows, task.validation_data.item_rows)
    log(f"Bayesian factorization machine | rank {config.rank}, {config.n_burn_in_iterations} + {config.n_kept_samples} draws | "
        f"valid MSE {mse(valid_predictions, task.validation_data.ratings):.4f}")

    # 2) optional: sampled again on 100 % of the training file -> test predictions
    if config.refit_on_train_and_valid:
        model = BayesianFactorizationMachine(task.training_and_validation_data, implicit_feedback_pairs(task.training_and_validation_data),
                                             task.n_user_rows, task.n_item_rows, config)
    test_predictions = model.predict(task.test_user_rows, task.test_item_rows)

    return ModelResult(model, [], valid_predictions, test_predictions)


def store_bayesian_factorization_machine(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "bayesian_factorization_machine")


def load_bayesian_factorization_machine(task_name: str) -> ModelResult:
    return load_model_result(task_name, "bayesian_factorization_machine")
