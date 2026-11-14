import numpy as np

from lib.custom_types import ItemCosineConfig, ModelResult, PairChoiceTask, RatingData
from lib.helper_functioins import sparse_matrix_from_entries
from lib.model_storage import load_model_result, save_model_result
from lib.models.item_knn import subsample_active_users, top_neighbour_similarities
from lib.pair_choice import train_pair_scorer

# Item-to-item similarity to the user's history (PML Ch. 4; the read-prediction idea of "books like the ones you read"):
#   score(u, i) = sum over the books j that u has read of s_ij
#   s_ij        = cosine of the "who read it" columns of i and j * n_ij / (n_ij + similarity_shrinkage), top neighbours of i only
# both books of a pair are compared for the same user -> no normalisation per user needed
# the neighbourhoods come from the same code as the item kNN of Tasks 1 / 3, with the 0/1 "has read" matrix instead of residuals


class ItemCosineModel:

    # no time terms -> lib/pair_choice.py only passes (user_rows, item_rows)
    uses_days = False

    def __init__(self, interactions: RatingData, n_user_rows: int, n_item_rows: int, config: ItemCosineConfig):

        # user x item 0/1 matrix of who read what (CSR -> fast access to the rows of the asking users)
        self.read_matrix = sparse_matrix_from_entries(interactions.user_rows, interactions.item_rows, n_user_rows, n_item_rows)

        # neighbour similarities from a subsample of the most active users' interactions (memory), scores use all interactions
        similarity_positions = subsample_active_users(interactions.user_rows, config.max_interactions_per_user_for_similarity, config.seed)
        similarity_read_matrix = sparse_matrix_from_entries(interactions.user_rows[similarity_positions],
                                                            interactions.item_rows[similarity_positions], n_user_rows, n_item_rows)
        self.neighbour_similarities = top_neighbour_similarities(similarity_read_matrix, similarity_read_matrix, config)

    # in chunks: the rows of very active users make the sparse products large
    def score(self, user_rows: np.ndarray, item_rows: np.ndarray, chunk_size: int = 20_000) -> np.ndarray:
        scores = np.empty(len(user_rows), dtype=np.float32)

        for chunk_start in range(0, len(user_rows), chunk_size):
            chunk = slice(chunk_start, chunk_start + chunk_size)
            # neighbour similarities of each asked item, restricted to the items this user has read
            neighbour_weights = self.neighbour_similarities[item_rows[chunk]]
            scores[chunk] = np.asarray(neighbour_weights.multiply(self.read_matrix[user_rows[chunk]]).sum(axis=1)).ravel()

        return scores


def train_item_cosine_model(task: PairChoiceTask, config: ItemCosineConfig) -> ModelResult:
    return train_pair_scorer(task, lambda interactions: ItemCosineModel(interactions, task.n_user_rows, task.n_item_rows, config),
                             "item cosine")


def store_item_cosine_model(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "item_cosine")


def load_item_cosine_model(task_name: str) -> ModelResult:
    return load_model_result(task_name, "item_cosine")
