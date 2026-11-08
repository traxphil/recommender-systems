import numpy as np

from lib.custom_types import ModelResult, PairChoiceTask, RatingData
from lib.model_storage import load_model_result, save_model_result
from lib.pair_choice import train_pair_scorer

# score(u, i) = log(1 + #interactions of item i), the same for every user
# the test negatives are drawn in proportion to popularity -> popularity alone barely separates a pair;
# it is still an input of the pair blender (e.g. EASE scores mean something different for popular and rare items)


class PopularityModel:

    # no time terms -> lib/pair_choice.py only passes (user_rows, item_rows)
    uses_days = False

    def __init__(self, interactions: RatingData, n_item_rows: int):
        self.interactions_per_item = np.bincount(interactions.item_rows, minlength=n_item_rows)

    def score(self, user_rows: np.ndarray, item_rows: np.ndarray) -> np.ndarray:
        return np.log1p(self.interactions_per_item[item_rows])


def train_popularity_model(task: PairChoiceTask) -> ModelResult:
    return train_pair_scorer(task, lambda interactions: PopularityModel(interactions, task.n_item_rows), "popularity")


def store_popularity_model(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "popularity")


def load_popularity_model(task_name: str) -> ModelResult:
    return load_model_result(task_name, "popularity")
