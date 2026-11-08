import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from lib.custom_types import ModelResult, RatingData, RatingTask, TimeZeroModelConfig
from lib.helper_functioins import log
from lib.model_storage import load_model_result, save_model_result
from lib.models.zero_model import ZeroModel, shrunk_zero_log_odds
from lib.temporal_features import UserTimeline, require_test_days

# P(rating = 0 | user, item, day): the zero model + what the user did around that day
# Task 3 data: 1.7 % of the ratings are 0, but 45 % when the same user gave another 0 on the same day
# (shelving several books at once without stars) -> per window around t:
#   log-odds of the zero share among the user's other ratings in the window, log(1 + #those ratings)


class TimeZeroModel(ZeroModel):

    def __init__(self, rating_data: RatingData, n_user_rows: int, n_item_rows: int, config: TimeZeroModelConfig):
        # the user's ratings and zeros on the time line -> window counts (already needed while ZeroModel.__init__ fits)
        self.user_timeline = UserTimeline(rating_data.user_rows, rating_data.days, is_zero=rating_data.ratings == 0)
        super().__init__(rating_data, n_user_rows, n_item_rows, config)

    # (N, 2 * windows); own_rating_is_zero given -> the queried ratings are in the timeline and are taken out again
    def window_features(self, user_rows: np.ndarray, days: np.ndarray, own_rating_is_zero: np.ndarray | None = None) -> np.ndarray:
        own_zero = np.zeros(len(user_rows)) if own_rating_is_zero is None else own_rating_is_zero
        own_count = np.zeros(len(user_rows)) if own_rating_is_zero is None else np.ones(len(user_rows))

        feature_columns = []
        for window_days in self.config.window_lengths_in_days:
            rating_counts, zero_counts = self.user_timeline.window_counts(user_rows, days, window_days)
            other_ratings, other_zeros = rating_counts - own_count, zero_counts - own_zero
            feature_columns += [shrunk_zero_log_odds(other_zeros, other_ratings, self.global_zero_share, self.config.window_prior_strength),
                                np.log1p(other_ratings)]

        return np.column_stack(feature_columns)

    def training_features(self, rating_data: RatingData, is_zero: np.ndarray) -> np.ndarray:
        return np.column_stack([super().training_features(rating_data, is_zero),
                                self.window_features(rating_data.user_rows, rating_data.days, own_rating_is_zero=is_zero)])

    def predict_zero_probability(self, user_rows: np.ndarray, item_rows: np.ndarray, days: np.ndarray) -> np.ndarray:
        features = np.column_stack([self.zero_features(user_rows, item_rows), self.window_features(user_rows, days)])
        return self.classifier.predict_proba(features)[:, 1].astype(np.float32)


def train_time_zero_model(task: RatingTask, config: TimeZeroModelConfig) -> ModelResult:
    require_test_days(task, "temporal zero model")

    # 1) fit on the training split -> validation P(0) (input of the blender)
    model = TimeZeroModel(task.training_data, task.n_user_rows, task.n_item_rows, config)
    validation = task.validation_data
    valid_zero_probabilities = model.predict_zero_probability(validation.user_rows, validation.item_rows, validation.days)
    valid_is_zero = validation.ratings == 0
    log(f"time zero model | valid AUC {roc_auc_score(valid_is_zero, valid_zero_probabilities):.4f} | "
        f"log loss {log_loss(valid_is_zero, valid_zero_probabilities):.4f} | mean P(0) {valid_zero_probabilities.mean():.4f} vs share {valid_is_zero.mean():.4f}")

    # 2) refit on 100 % of the training file -> test P(0)
    model = TimeZeroModel(task.training_and_validation_data, task.n_user_rows, task.n_item_rows, config)
    test_zero_probabilities = model.predict_zero_probability(task.test_user_rows, task.test_item_rows, task.test_days)

    # "predictions" of this model are probabilities, not ratings
    return ModelResult(model, [], valid_zero_probabilities, test_zero_probabilities)


def store_time_zero_model(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "time_zero_model")


def load_time_zero_model(task_name: str) -> ModelResult:
    return load_model_result(task_name, "time_zero_model")
