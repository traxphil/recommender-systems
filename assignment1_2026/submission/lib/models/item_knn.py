import numpy as np
import scipy.sparse as sp

from lib.config import RATING_MAX, RATING_MIN
from lib.custom_types import ItemCosineConfig, ItemKnnConfig, ModelResult, RatingData, RatingTask
from lib.helper_functioins import log, mse
from lib.model_storage import load_model_result, save_model_result
from lib.models.bias_model import BiasModel

# Item-item kNN on the residuals of the bias model (PML Ch. 4; Koren 2010, "Factor in the Neighbors", Sec. 2):
#   r(u,i) = baseline(u,i) + sum_{j in neighbours(i), rated by u} s_ij * (r_uj - baseline(u,j)) / (sum_j s_ij + prediction_shrinkage)
#   baseline(u,i) = mu + b_u + b_i  (closed form, exactly the bias model's fit -> shared config in lib/config.py)
# Users who rate i similarly to j relative to their baseline make s_ij large -> "people who liked j more than expected
# also liked i more than expected" -> local item-to-item structure that the low-rank factor models smooth out


# shrunk cosine similarity between the residual columns of every item pair, top n_neighbors per item kept
#   s_ij = (sum_u e_ui e_uj) / (|e_i| |e_j|) * n_ij / (n_ij + similarity_shrinkage)     (n_ij = users who rated both)
# computed for items_per_similarity_block items at a time -> the full item x item matrix (1e8+ entries) never exists
# Source: Sarwar et al. 2001 (item-based CF, adjusted cosine); shrinkage n / (n + lambda) from Koren 2010, Eq. 2
# (also used by the item cosine model of Tasks 2 / 4 with the 0/1 "has read" matrix as residual_matrix)
def top_neighbour_similarities(residual_matrix: sp.csr_matrix, rated_matrix: sp.csr_matrix,
                               config: ItemKnnConfig | ItemCosineConfig) -> sp.csr_matrix:

    n_item_rows = residual_matrix.shape[1]

    # 1) column norms |e_i| -> 1 / |e_i| as scaling factors (items without residuals get 0)
    column_norms = np.sqrt(np.asarray(residual_matrix.multiply(residual_matrix).sum(axis=0)).ravel())
    inverse_norms = np.where(column_norms > 1e-8, 1.0 / np.maximum(column_norms, 1e-8), 0.0)

    # column access per block (CSC) / row access for the products (CSR)
    residual_columns, rated_columns = residual_matrix.tocsc(), rated_matrix.tocsc()

    neighbour_rows, neighbour_columns, neighbour_values = [], [], []
    for block_start in range(0, n_item_rows, config.items_per_similarity_block):
        block_items = np.arange(block_start, min(block_start + config.items_per_similarity_block, n_item_rows))

        # 2) residual dot products and co-rating counts of the block items with every item: (block, items)
        residual_products = residual_columns[:, block_items].T @ residual_matrix
        co_rating_counts = (rated_columns[:, block_items].T @ rated_matrix).tocsr()

        # 3) cosine (scale rows and columns by 1 / norm), then shrink by the number of common users
        cosine = sp.diags(inverse_norms[block_items]) @ residual_products @ sp.diags(inverse_norms)
        co_rating_counts.data = co_rating_counts.data / (co_rating_counts.data + config.similarity_shrinkage)
        similarities = cosine.multiply(co_rating_counts).tocsr()

        # 4) keep the n_neighbors largest positive similarities per item (never the item itself)
        rows_in_block = np.repeat(np.arange(len(block_items)), np.diff(similarities.indptr))
        columns, values = similarities.indices, similarities.data
        is_candidate = (values > 0) & (columns != block_items[rows_in_block])
        rows_in_block, columns, values = rows_in_block[is_candidate], columns[is_candidate], values[is_candidate]

        by_row_then_similarity = np.lexsort((-values, rows_in_block))
        rows_in_block, columns, values = rows_in_block[by_row_then_similarity], columns[by_row_then_similarity], values[by_row_then_similarity]
        rank_within_row = np.arange(len(rows_in_block)) - np.searchsorted(rows_in_block, rows_in_block)
        is_kept = rank_within_row < config.n_neighbors

        neighbour_rows.append(block_items[rows_in_block[is_kept]])
        neighbour_columns.append(columns[is_kept])
        neighbour_values.append(values[is_kept].astype(np.float32))

    return sp.csr_matrix((np.concatenate(neighbour_values), (np.concatenate(neighbour_rows), np.concatenate(neighbour_columns))),
                         shape=(n_item_rows, n_item_rows))


# keep at most max_ratings ratings per user (random, seeded) -> the item pair count grows with (ratings per user)^2
def subsample_active_users(user_rows: np.ndarray, max_ratings: int, seed: int) -> np.ndarray:
    random_order = np.random.default_rng(seed).permutation(len(user_rows))
    grouped_by_user = random_order[np.argsort(user_rows[random_order], kind="stable")]
    ratings_per_user = np.bincount(user_rows)
    first_position_of_user = np.cumsum(ratings_per_user) - ratings_per_user
    rank_within_user = np.arange(len(grouped_by_user)) - first_position_of_user[user_rows[grouped_by_user]]
    return np.sort(grouped_by_user[rank_within_user < max_ratings])


class ItemKnnModel:

    def __init__(self, rating_data: RatingData, n_user_rows: int, n_item_rows: int, config: ItemKnnConfig):
        self.prediction_shrinkage = config.prediction_shrinkage

        # 1) baseline mu + b_u + b_i = exactly the bias model's fit (same BiasModel class, same config object in lib/config.py)
        #    -> in the blend, (kNN - bias model) is the pure neighbourhood correction; then the residual of every known rating
        baseline_model = BiasModel(rating_data, n_user_rows, n_item_rows, config.baseline_biases)
        self.global_mean_rating = baseline_model.global_mean_rating
        self.user_biases, self.item_biases = baseline_model.user_biases, baseline_model.item_biases
        residuals = rating_data.ratings - self.baseline(rating_data.user_rows, rating_data.item_rows)

        # 2) user x item matrices: residuals / "has rated" (CSR -> fast access to the rows of the asking users)
        def user_item_matrix(values, positions):
            return sp.csr_matrix((values, (rating_data.user_rows[positions], rating_data.item_rows[positions])),
                                 shape=(n_user_rows, n_item_rows), dtype=np.float32)

        all_positions = np.arange(len(residuals))
        self.residual_matrix = user_item_matrix(residuals, all_positions)
        self.rated_matrix = user_item_matrix(np.ones(len(residuals)), all_positions)

        # 3) neighbour similarities from a subsample of the most active users' ratings (memory), predictions use all ratings
        similarity_positions = subsample_active_users(rating_data.user_rows, config.max_ratings_per_user_for_similarity, config.seed)
        self.neighbour_similarities = top_neighbour_similarities(
            user_item_matrix(residuals[similarity_positions], similarity_positions),
            user_item_matrix(np.ones(len(similarity_positions)), similarity_positions),
            config)

    def baseline(self, user_rows: np.ndarray, item_rows: np.ndarray) -> np.ndarray:
        return self.global_mean_rating + self.user_biases[user_rows] + self.item_biases[item_rows]

    # in chunks: the rows of very active users make the sparse products large
    def predict(self, user_rows: np.ndarray, item_rows: np.ndarray, chunk_size: int = 20_000) -> np.ndarray:
        predictions = np.empty(len(user_rows), dtype=np.float32)

        for chunk_start in range(0, len(user_rows), chunk_size):
            chunk = slice(chunk_start, chunk_start + chunk_size)
            chunk_users, chunk_items = user_rows[chunk], item_rows[chunk]

            # neighbour weights of each asked item, restricted to the items this user has rated (elementwise products)
            neighbour_weights = self.neighbour_similarities[chunk_items]
            weighted_residuals = np.asarray(neighbour_weights.multiply(self.residual_matrix[chunk_users]).sum(axis=1)).ravel()
            weight_sums = np.asarray(neighbour_weights.multiply(self.rated_matrix[chunk_users]).sum(axis=1)).ravel()

            # no rated neighbours (or unknown user / item) -> 0 / shrinkage = 0 -> the baseline
            predictions[chunk] = self.baseline(chunk_users, chunk_items) + weighted_residuals / (weight_sums + self.prediction_shrinkage)

        return np.clip(predictions, RATING_MIN, RATING_MAX)


def train_item_knn(task: RatingTask, config: ItemKnnConfig) -> ModelResult:

    # 1) neighbourhoods from the training split -> validation predictions (input of the blender)
    model = ItemKnnModel(task.training_data, task.n_user_rows, task.n_item_rows, config)
    valid_predictions = model.predict(task.validation_data.user_rows, task.validation_data.item_rows)
    log(f"item kNN | {model.neighbour_similarities.nnz:,} neighbour links | valid MSE {mse(valid_predictions, task.validation_data.ratings):.4f}")

    # 2) rebuilt on 100 % of the training file -> test predictions
    model = ItemKnnModel(task.training_and_validation_data, task.n_user_rows, task.n_item_rows, config)
    test_predictions = model.predict(task.test_user_rows, task.test_item_rows)

    return ModelResult(model, [], valid_predictions, test_predictions)


def store_item_knn(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "item_knn")


def load_item_knn(task_name: str) -> ModelResult:
    return load_model_result(task_name, "item_knn")
