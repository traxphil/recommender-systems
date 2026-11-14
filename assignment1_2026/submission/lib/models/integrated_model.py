import numpy as np
import torch
import torch.nn as nn

from lib.config import DEVICE
from lib.custom_types import IntegratedModelConfig, ModelResult, RatingData, RatingTask
from lib.data_loading import known_interaction_pairs
from lib.helper_functioins import fit_closed_form_biases, sparse_matrix_from_entries
from lib.model_storage import load_model_result, save_model_result
from lib.models.item_knn import subsample_active_users, top_neighbour_similarities
from lib.models.SVDpp import LatentFactorModel
from lib.torch_trainer import fit_refit_and_predict

# Integrated model (Koren 2008, "Factorization Meets the Neighborhood", KDD, Sec. 5, Eq. 16): SVD++ + a learned neighbourhood
#   r(u,i) = mu + b_u + b_i + q_i . (p_u + |N(u)|^-1/2 sum_{j in N(u)} y_j)
#          + |R^k(i;u)|^-1/2 sum_{j in R^k(i;u)} (r_uj - baseline_uj) w_ij  +  |N^k(i;u)|^-1/2 sum_{j in N^k(i;u)} c_ij
#   R^k(i;u) = the books the user rated among the k neighbours of i, N^k(i;u) = the books of N(u) among them
#   baseline_uj = mu + b_u + b_j of the closed-form bias model (fixed, as in the paper)
#   w_ij, c_ij  = one learned weight per neighbour link, starting at 0 -> epoch 0 = SVD++ at its starting point
# unlike the item kNN, the neighbour weights are fitted to the ratings (not fixed cosine similarities), together with the factors
# the k neighbours of every book: the shrunk residual cosine of the item kNN (lib/models/item_knn.py)
# the neighbour lookups run on the CPU (scipy CSR) once per batch, like the N(u) rows of SVD++ -> large batches keep it fast


class IntegratedModel(LatentFactorModel):

    # Parameters on top of LatentFactorModel:
    # neighbour_weights            w_ij    (neighbour links, 1)    start at 0
    # implicit_neighbour_weights   c_ij    (neighbour links, 1)    start at 0
    # (not parameters, scipy CSR on the CPU):
    #   neighbour_links  (I+1, I+1)  entry (i, j) = 1 + the index of the link i -> j (the k neighbours of every book i)
    #   residual_matrix  (U+1, I+1)  r_uj - baseline_uj of the training ratings;  rated_matrix: 1 for every training rating
    #   known_matrix     (U+1, I+1)  1 for every pair of N(u)

    def __init__(self, neighbour_links, residual_matrix, rated_matrix, known_matrix, neighbour_regularization: float,
                 n_training_ratings: int, **latent_factor_arguments):

        super().__init__(**latent_factor_arguments)
        self.neighbour_links, self.residual_matrix = neighbour_links, residual_matrix
        self.rated_matrix, self.known_matrix = rated_matrix, known_matrix
        self.neighbour_regularization, self.n_training_ratings = neighbour_regularization, n_training_ratings

        self.neighbour_weights = nn.Embedding(neighbour_links.nnz, 1)
        self.implicit_neighbour_weights = nn.Embedding(neighbour_links.nnz, 1)
        nn.init.zeros_(self.neighbour_weights.weight)
        nn.init.zeros_(self.implicit_neighbour_weights.weight)

    # |S|^-1/2 * sum over the entries S of every row of weight[link] * value (rows without entries -> 0)
    @staticmethod
    def normalised_link_sum(link_entries, weights: nn.Embedding, values: np.ndarray, n_rows: int) -> torch.Tensor:
        device = weights.weight.device
        rows = torch.as_tensor(link_entries.row, dtype=torch.long, device=device)
        link_indices = torch.as_tensor(np.rint(link_entries.data).astype(np.int64) - 1, device=device)
        weighted_values = weights(link_indices).squeeze(1) * torch.as_tensor(values, dtype=torch.float32, device=device)
        sums = torch.zeros(n_rows, device=device).index_add_(0, rows, weighted_values)
        counts = torch.zeros(n_rows, device=device).index_add_(0, rows, torch.ones_like(weighted_values))
        return sums / counts.clamp(min=1.0).sqrt()

    # the two neighbourhood sums of every rating (COO entries: row = the rating in the batch, column = the neighbour j)
    def neighbourhood_term(self, user_rows, item_rows):
        users, items = user_rows.cpu().numpy(), item_rows.cpu().numpy()
        links_of_asked_books = self.neighbour_links[items]

        # R^k(i;u): neighbours the user rated, each weighted by the user's residual on it
        rated_links = links_of_asked_books.multiply(self.rated_matrix[users]).tocoo()
        residuals = np.asarray(self.residual_matrix[users[rated_links.row], rated_links.col]).ravel()
        explicit_part = self.normalised_link_sum(rated_links, self.neighbour_weights, residuals, len(users))

        # N^k(i;u): neighbours in the user's implicit feedback
        known_links = links_of_asked_books.multiply(self.known_matrix[users]).tocoo()
        implicit_part = self.normalised_link_sum(known_links, self.implicit_neighbour_weights, np.ones(known_links.nnz), len(users))

        return explicit_part + implicit_part

    def forward(self, user_rows, item_rows):
        return super().forward(user_rows, item_rows) + self.neighbourhood_term(user_rows, item_rows)

    # SVD++ penalty + lambda_n * (sum of all squared w_ij, c_ij) / number of training ratings
    def l2_penalty(self, user_rows, item_rows, user_penalty_weights, item_penalty_weights,
                   user_bias_regularization, item_bias_regularization, factor_regularization):
        penalty = super().l2_penalty(user_rows, item_rows, user_penalty_weights, item_penalty_weights,
                                     user_bias_regularization, item_bias_regularization, factor_regularization)
        squared_neighbour_weights = (self.neighbour_weights.weight ** 2).sum() + (self.implicit_neighbour_weights.weight ** 2).sum()
        return penalty + self.neighbour_regularization * squared_neighbour_weights / self.n_training_ratings


def train_integrated_model(task: RatingTask, config: IntegratedModelConfig) -> ModelResult:

    def build_model(training_ratings: RatingData) -> IntegratedModel:
        torch.manual_seed(config.seed)  # identical initial factors for the fit and the refit
        user_rows, item_rows, ratings = training_ratings.user_rows, training_ratings.item_rows, training_ratings.ratings
        n_user_rows, n_item_rows = task.n_user_rows, task.n_item_rows

        # 1) closed-form baseline = starting biases and the fixed baseline of the residuals
        global_mean_rating = float(ratings.mean())
        initial_user_biases, initial_item_biases = fit_closed_form_biases(
            user_rows, item_rows, ratings, n_user_rows, n_item_rows, global_mean_rating,
            config.user_bias_regularization, config.item_bias_regularization)
        residuals = ratings - (global_mean_rating + initial_user_biases[user_rows] + initial_item_biases[item_rows])
        residual_matrix = sparse_matrix_from_entries(user_rows, item_rows, n_user_rows, n_item_rows, values=residuals)
        rated_matrix = sparse_matrix_from_entries(user_rows, item_rows, n_user_rows, n_item_rows)
        rated_matrix.data[:] = 1.0              # 0 / 1 even for a pair rated twice (summed in a sparse matrix)

        # 2) the k neighbours of every book (shrunk residual cosine as in the item kNN; very active users subsampled)
        similarity_positions = subsample_active_users(user_rows, config.max_ratings_per_user_for_similarity, config.seed)
        similarities = top_neighbour_similarities(
            sparse_matrix_from_entries(user_rows[similarity_positions], item_rows[similarity_positions], n_user_rows, n_item_rows,
                                       values=residuals[similarity_positions]),
            sparse_matrix_from_entries(user_rows[similarity_positions], item_rows[similarity_positions], n_user_rows, n_item_rows),
            config)
        neighbour_links = similarities.copy()
        neighbour_links.data = np.arange(1, similarities.nnz + 1, dtype=np.float32)

        # 3) N(u): the ratings it trains on, or every known pair of the training + test file (ratings never used)
        implicit_feedback_pairs = known_interaction_pairs(task) if config.use_test_pairs_as_implicit_feedback else training_ratings
        known_matrix = sparse_matrix_from_entries(implicit_feedback_pairs.user_rows, implicit_feedback_pairs.item_rows,
                                                  n_user_rows, n_item_rows)
        known_matrix.data[:] = 1.0

        return IntegratedModel(neighbour_links, residual_matrix, rated_matrix, known_matrix,
                               neighbour_regularization=config.neighbour_regularization, n_training_ratings=len(ratings),
                               n_user_rows=n_user_rows, n_item_rows=n_item_rows, n_factors=config.n_factors,
                               global_mean_rating=global_mean_rating, implicit_feedback_ratings=implicit_feedback_pairs,
                               initial_user_biases=initial_user_biases, initial_item_biases=initial_item_biases).to(DEVICE)

    # fit with early stopping -> validation predictions -> optional refit on 100 % -> test predictions
    return fit_refit_and_predict(task, build_model,
                                 max_epochs=config.max_epochs,
                                 early_stopping_patience=config.early_stopping_patience,
                                 refit_on_train_and_valid=config.refit_on_train_and_valid,
                                 learning_rate=config.learning_rate,
                                 user_bias_regularization=config.user_bias_regularization,
                                 item_bias_regularization=config.item_bias_regularization,
                                 factor_regularization=config.factor_regularization,
                                 ratings_per_batch=config.ratings_per_batch,
                                 seed=config.seed)


def store_integrated_model(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "integrated_model")


def load_integrated_model(task_name: str) -> ModelResult:
    return load_model_result(task_name, "integrated_model")
