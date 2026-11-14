import numpy as np

from lib.config import RATING_MAX, RATING_MIN
from lib.custom_types import ModelResult, RatingData, RatingTask, TimeItemKnnConfig
from lib.helper_functioins import log, mse, sparse_matrix_from_entries
from lib.model_storage import load_model_result, save_model_result
from lib.models.item_knn import ItemKnnModel
from lib.temporal_features import require_test_days

# Item-item kNN with time-decayed neighbours (Koren 2009, "Collaborative Filtering with Temporal Dynamics" (KDD),
# temporal dynamics at neighborhood models; recency weighting as in Ding & Li 2005, PML Sec. 7.3.1):
#   r(u,i,t) = baseline(u,i) + sum_j s_ij d_uj(t) (r_uj - baseline(u,j)) / (sum_j s_ij d_uj(t) + prediction_shrinkage)
#   d_uj(t)  = floor + (1 - floor) * exp(-|t - t_uj| / time_decay_days)        t_uj = day on which u rated neighbour j
# floor > 0: far-away neighbours still count (a pure decay leaves too few neighbours -> predictions fall back to the baseline)
# the Task 3 test rows lie inside the training period -> the user's ratings shortly before AND after t count the most
# neighbourhoods (similarities) are built exactly as in ItemKnnModel; only the prediction weights change


class TimeItemKnnModel(ItemKnnModel):

    def __init__(self, rating_data: RatingData, n_user_rows: int, n_item_rows: int, config: TimeItemKnnConfig):
        super().__init__(rating_data, n_user_rows, n_item_rows, config)
        self.time_decay_days = config.time_decay_days
        self.time_decay_floor = config.time_decay_floor

        # user x item matrix of rating days (same entries as residual_matrix / rated_matrix)
        # stored as day + 1 -> ratings on day 0 are not dropped as sparse zeros
        self.rating_day_matrix = sparse_matrix_from_entries(rating_data.user_rows, rating_data.item_rows, n_user_rows, n_item_rows,
                                                            values=rating_data.days + 1.0)

    # in chunks: the rows of very active users make the sparse products large
    def predict(self, user_rows: np.ndarray, item_rows: np.ndarray, days: np.ndarray, chunk_size: int = 20_000) -> np.ndarray:
        predictions = np.empty(len(user_rows), dtype=np.float32)

        for chunk_start in range(0, len(user_rows), chunk_size):
            chunk = slice(chunk_start, chunk_start + chunk_size)
            chunk_users, chunk_items, chunk_days = user_rows[chunk], item_rows[chunk], days[chunk]

            # time decay of every rating of the asking user relative to the asked day (one row per asked rating;
            # the COO row of an entry = the asked rating it belongs to)
            time_decays = self.rating_day_matrix[chunk_users]
            asked_day_per_entry = chunk_days[time_decays.tocoo().row]
            exponential_decay = np.exp(-np.abs(asked_day_per_entry - (time_decays.data - 1.0)) / self.time_decay_days)
            time_decays.data = (self.time_decay_floor + (1.0 - self.time_decay_floor) * exponential_decay).astype(np.float32)

            # neighbour weights s_ij * d_uj(t), only for neighbours the user has rated
            decayed_weights = self.neighbour_similarities[chunk_items].multiply(time_decays).tocsr()
            weighted_residuals = np.asarray(decayed_weights.multiply(self.residual_matrix[chunk_users]).sum(axis=1)).ravel()
            weight_sums = np.asarray(decayed_weights.sum(axis=1)).ravel()

            # no rated neighbours (or unknown user / item) -> 0 / shrinkage = 0 -> the baseline
            predictions[chunk] = self.baseline(chunk_users, chunk_items) + weighted_residuals / (weight_sums + self.prediction_shrinkage)

        return np.clip(predictions, RATING_MIN, RATING_MAX)


def train_time_item_knn(task: RatingTask, config: TimeItemKnnConfig) -> ModelResult:
    require_test_days(task, "time-decayed item kNN")

    # 1) neighbourhoods from the training split -> validation predictions (input of the blender)
    model = TimeItemKnnModel(task.training_data, task.n_user_rows, task.n_item_rows, config)
    validation = task.validation_data
    valid_predictions = model.predict(validation.user_rows, validation.item_rows, validation.days)
    log(f"time item kNN | decay {config.time_decay_days:g} days, floor {config.time_decay_floor:g} | "
        f"valid MSE {mse(valid_predictions, validation.ratings):.4f}")

    # 2) rebuilt on 100 % of the training file -> test predictions
    model = TimeItemKnnModel(task.training_and_validation_data, task.n_user_rows, task.n_item_rows, config)
    test_predictions = model.predict(task.test_user_rows, task.test_item_rows, task.test_days)

    return ModelResult(model, [], valid_predictions, test_predictions)


def store_time_item_knn(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "time_item_knn")


def load_time_item_knn(task_name: str) -> ModelResult:
    return load_model_result(task_name, "time_item_knn")
