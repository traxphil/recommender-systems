import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib.custom_types import ModelResult, RatingData, RatingTask, ZeroModelConfig
from lib.helper_functioins import log
from lib.model_storage import load_model_result, save_model_result

# P(rating = 0 | user, item): in the Goodreads data a 0 mostly means "shelved without stars" -> its own behaviour
# 0-star ratings are 1.5 % of the Task 1 ratings but ~21 % of the squared error of the bias model,
# and they cluster in a few users (users with >= 30 % zeros in training: 63 % of their validation ratings are 0)
# -> logistic regression on how often this user / this item got a 0 elsewhere + how many ratings both have
# the blender gets P(0) as an input and can pull predictions towards 0 where a 0 is likely


# log-odds of a zero share shrunk towards the global share: few ratings -> close to the global share
def shrunk_zero_log_odds(zero_counts: np.ndarray, rating_counts: np.ndarray, global_zero_share: float, prior_strength: float) -> np.ndarray:
    zero_share = (zero_counts + prior_strength * global_zero_share) / (rating_counts + prior_strength)
    return np.log(zero_share / (1.0 - zero_share))


class ZeroModel:

    def __init__(self, rating_data: RatingData, n_user_rows: int, n_item_rows: int, config: ZeroModelConfig):
        self.config = config

        # 1) zeros and ratings per user / per item in the data this model learns from
        is_zero = (rating_data.ratings == 0).astype(np.float64)
        self.global_zero_share = float(is_zero.mean())
        self.zeros_of_user = np.bincount(rating_data.user_rows, weights=is_zero, minlength=n_user_rows)
        self.ratings_of_user = np.bincount(rating_data.user_rows, minlength=n_user_rows).astype(np.float64)
        self.zeros_of_item = np.bincount(rating_data.item_rows, weights=is_zero, minlength=n_item_rows)
        self.ratings_of_item = np.bincount(rating_data.item_rows, minlength=n_item_rows).astype(np.float64)

        # 2) features of the training ratings without the rating itself (leave-one-out), otherwise the feature contains the answer
        training_features = self.training_features(rating_data, is_zero)

        # 3) logistic regression on standardised features
        self.classifier = make_pipeline(StandardScaler(), LogisticRegression(C=config.inverse_regularization, max_iter=config.max_iterations))
        self.classifier.fit(training_features, is_zero.astype(np.int64))

    # (N, 4): log-odds of the user's / item's zero share, log(1 + #ratings) of user / item
    # own_rating_is_zero given -> the queried ratings are part of the counts and are taken out again
    def zero_features(self, user_rows: np.ndarray, item_rows: np.ndarray, own_rating_is_zero: np.ndarray | None = None) -> np.ndarray:
        own_zero = np.zeros(len(user_rows)) if own_rating_is_zero is None else own_rating_is_zero
        own_count = np.zeros(len(user_rows)) if own_rating_is_zero is None else np.ones(len(user_rows))

        user_zeros, user_ratings = self.zeros_of_user[user_rows] - own_zero, self.ratings_of_user[user_rows] - own_count
        item_zeros, item_ratings = self.zeros_of_item[item_rows] - own_zero, self.ratings_of_item[item_rows] - own_count

        return np.column_stack([
            shrunk_zero_log_odds(user_zeros, user_ratings, self.global_zero_share, self.config.user_prior_strength),
            shrunk_zero_log_odds(item_zeros, item_ratings, self.global_zero_share, self.config.item_prior_strength),
            np.log1p(user_ratings),
            np.log1p(item_ratings),
        ])

    # features of the ratings the classifier learns from (the temporal zero model adds its own columns here)
    def training_features(self, rating_data: RatingData, is_zero: np.ndarray) -> np.ndarray:
        return self.zero_features(rating_data.user_rows, rating_data.item_rows, own_rating_is_zero=is_zero)

    def predict_zero_probability(self, user_rows: np.ndarray, item_rows: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.zero_features(user_rows, item_rows))[:, 1].astype(np.float32)


def train_zero_model(task: RatingTask, config: ZeroModelConfig) -> ModelResult:

    # 1) fit on the training split -> validation P(0) (input of the blender)
    model = ZeroModel(task.training_data, task.n_user_rows, task.n_item_rows, config)
    valid_zero_probabilities = model.predict_zero_probability(task.validation_data.user_rows, task.validation_data.item_rows)
    valid_is_zero = task.validation_data.ratings == 0
    log(f"zero model | valid AUC {roc_auc_score(valid_is_zero, valid_zero_probabilities):.4f} | "
        f"log loss {log_loss(valid_is_zero, valid_zero_probabilities):.4f} | mean P(0) {valid_zero_probabilities.mean():.4f} vs share {valid_is_zero.mean():.4f}")

    # 2) refit on 100 % of the training file -> test P(0)
    model = ZeroModel(task.training_and_validation_data, task.n_user_rows, task.n_item_rows, config)
    test_zero_probabilities = model.predict_zero_probability(task.test_user_rows, task.test_item_rows)

    # "predictions" of this model are probabilities, not ratings
    return ModelResult(model, [], valid_zero_probabilities, test_zero_probabilities)


def store_zero_model(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "zero_model")


def load_zero_model(task_name: str) -> ModelResult:
    return load_model_result(task_name, "zero_model")
