import numpy as np

from lib.config import RATING_MAX, RATING_MIN
from lib.custom_types import BiasModelConfig, ModelResult, RatingData, RatingTask
from lib.helper_functioins import fit_closed_form_biases, log, mse
from lib.model_storage import load_model_result, save_model_result

# r(u,i) = mu + b_u + b_i in closed form (no training loop) -> baseline and ensemble input


class BiasModel:

    def __init__(self, rating_data: RatingData, n_user_rows: int, n_item_rows: int, config: BiasModelConfig):
        self.global_mean_rating = float(rating_data.ratings.mean())
        self.user_biases, self.item_biases = fit_closed_form_biases(
            rating_data.user_rows, rating_data.item_rows, rating_data.ratings,
            n_user_rows, n_item_rows, self.global_mean_rating,
            config.user_bias_regularization, config.item_bias_regularization, config.n_iterations)

    # unknown users / items (row 0) have bias 0 -> mu plus whatever part is known
    def predict(self, user_rows: np.ndarray, item_rows: np.ndarray) -> np.ndarray:
        predictions = self.global_mean_rating + self.user_biases[user_rows] + self.item_biases[item_rows]
        return np.clip(predictions, RATING_MIN, RATING_MAX).astype(np.float32)


def train_bias_model(task: RatingTask, config: BiasModelConfig) -> ModelResult:

    # 1) fit on the training split -> validation predictions (input of the blender)
    model = BiasModel(task.training_data, task.n_user_rows, task.n_item_rows, config)
    valid_predictions = model.predict(task.validation_data.user_rows, task.validation_data.item_rows)
    log(f"bias model | valid MSE {mse(valid_predictions, task.validation_data.ratings):.4f}")

    # 2) refit on 100 % of the training file -> test predictions
    model = BiasModel(task.training_and_validation_data, task.n_user_rows, task.n_item_rows, config)
    test_predictions = model.predict(task.test_user_rows, task.test_item_rows)

    return ModelResult(model, [], valid_predictions, test_predictions)


def store_bias_model(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "bias_model")


def load_bias_model(task_name: str) -> ModelResult:
    return load_model_result(task_name, "bias_model")
