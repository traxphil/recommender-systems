
import numpy as np
import torch
import torch.nn as nn

from lib.custom_types import ModelResult, RatingData
from lib.helper_functioins import sparse_matrix_from_entries
from lib.model_storage import load_model_result, save_model_result

# SVD   : r(u,i) = mu + b_u + b_i + q_i . p_u
# SVD++ : r(u,i) = mu + b_u + b_i + q_i . (p_u + |N(u)|^-1/2 * sum_{j in N(u)} y_j)

class LatentFactorModel(nn.Module):

    # no time terms -> lib/torch_trainer.py only passes (user_rows, item_rows); timeSVD++ (time_svdpp.py) sets True
    uses_days = False

    # Parameters:
    # user_biases               b_u     (U+1, 1)
    # item_biases               b_i     (I+1, 1)
    # user_factors              p_u     (U+1, n_factors)
    # item_factors              q_i     (I+1, n_factors)
    # implicit_item_factors     y_j     (I+1, n_factors)        // for SVD++
    # (not a parameter) normalised_interaction_matrix: N(u) as a (U+1, I+1) scipy CSR matrix   // for SVD++


    def __init__(self, n_user_rows, n_item_rows, n_factors, global_mean_rating, implicit_feedback_ratings: RatingData | None = None,
                 initial_user_biases=None, initial_item_biases=None, initial_factor_std=0.01):

        super().__init__()
        self.global_mean_rating = global_mean_rating

        # set up parameter tables
        self.user_biases = nn.Embedding(n_user_rows, 1, padding_idx=0)
        self.item_biases = nn.Embedding(n_item_rows, 1, padding_idx=0)
        self.user_factors = nn.Embedding(n_user_rows, n_factors, padding_idx=0)
        self.item_factors = nn.Embedding(n_item_rows, n_factors, padding_idx=0)

        # biases start at 0, or at given values (e.g. the closed-form bias model) -> training starts from a good baseline
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)
        with torch.no_grad():
            if initial_user_biases is not None:
                self.user_biases.weight[:, 0] = torch.as_tensor(initial_user_biases, dtype=torch.float32)
            if initial_item_biases is not None:
                self.item_biases.weight[:, 0] = torch.as_tensor(initial_item_biases, dtype=torch.float32)

        # factors start at small noise
        for factor_table in (self.user_factors, self.item_factors):
            nn.init.normal_(factor_table.weight, std=initial_factor_std)
            with torch.no_grad():
                factor_table.weight[0] = 0.0

        # SVD++ only: implicit item factors y_j & N(u) = the items of implicit_feedback_ratings (the ratings this model trains on)
        # as a user x item matrix with |N(u)|^-1/2 in every entry of row u -> row u times y = the implicit term of the formula
        self.uses_implicit_feedback = implicit_feedback_ratings is not None
        if self.uses_implicit_feedback:
            self.implicit_item_factors = nn.EmbeddingBag(n_item_rows, n_factors, mode="sum", include_last_offset=True)
            nn.init.normal_(self.implicit_item_factors.weight, std=initial_factor_std)

            user_rows, item_rows = implicit_feedback_ratings.user_rows, implicit_feedback_ratings.item_rows
            ratings_per_user = np.bincount(user_rows, minlength=n_user_rows)
            self.normalised_interaction_matrix = sparse_matrix_from_entries(user_rows, item_rows, n_user_rows, n_item_rows,
                                                                            values=1.0 / np.sqrt(ratings_per_user[user_rows]))


    # For SVD++: |N(u)|^-1/2 * sum_{j in N(u)} y_j for the user of every rating, computed once per distinct user
    # the CSR rows of these users are exactly the bags of EmbeddingBag: indices = the items, indptr = where each user's
    # items start, data = the weights |N(u)|^-1/2 (a user without ratings, e.g. row 0, has an empty row -> zero vector)
    def implicit_feedback_vectors(self, user_rows):

        distinct_users, distinct_index_per_rating = torch.unique(user_rows, return_inverse=True)
        interaction_rows = self.normalised_interaction_matrix[distinct_users.cpu().numpy()]

        device = self.implicit_item_factors.weight.device
        implicit_vector_per_user = self.implicit_item_factors(torch.as_tensor(interaction_rows.indices, device=device),
                                                              torch.as_tensor(interaction_rows.indptr, device=device),
                                                              per_sample_weights=torch.as_tensor(interaction_rows.data, device=device))

        return implicit_vector_per_user[distinct_index_per_rating]


    def forward(self, user_rows, item_rows):

        user_vectors = self.user_factors(user_rows)
        
        if self.uses_implicit_feedback:
            user_vectors = user_vectors + self.implicit_feedback_vectors(user_rows)

        user_item_affinity = (user_vectors * self.item_factors(item_rows)).sum(dim=1)

        biases = self.user_biases(user_rows).squeeze(1) + self.item_biases(item_rows).squeeze(1)

        return self.global_mean_rating + biases + user_item_affinity


    # lambda_bu * sum b_u^2 + lambda_bi * sum b_i^2 + lambda_f * (sum ||p_u||^2 + sum ||q_i||^2 [+ sum ||y_j||^2])
    # (PML Eq. 5.12 with the bias lambda split into a user and an item part)
    def l2_penalty(self, user_rows, item_rows, user_penalty_weights, item_penalty_weights,
                   user_bias_regularization, item_bias_regularization, factor_regularization):

        # squared norms of the parameters used by each rating
        user_bias_squared = self.user_biases(user_rows).squeeze(1) ** 2
        item_bias_squared = self.item_biases(item_rows).squeeze(1) ** 2
        user_factor_squared = (self.user_factors(user_rows) ** 2).sum(dim=1)
        item_factor_squared = (self.item_factors(item_rows) ** 2).sum(dim=1)

        if self.uses_implicit_feedback:
            # y_j is regularised like q_j: through the ratings of item j
            item_factor_squared = item_factor_squared + (self.implicit_item_factors.weight[item_rows] ** 2).sum(dim=1)

        # weight per rating, then average over the batch
        bias_penalty = (user_bias_regularization * user_penalty_weights * user_bias_squared
                        + item_bias_regularization * item_penalty_weights * item_bias_squared)
        factor_penalty = user_penalty_weights * user_factor_squared + item_penalty_weights * item_factor_squared

        return (bias_penalty + factor_regularization * factor_penalty).mean()


# SVD and SVD++ are the same class -> stored under their own names, so both can be part of the ensemble
def store_svd(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "svd")


def load_svd(task_name: str) -> ModelResult:
    return load_model_result(task_name, "svd")


def store_svdpp(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "svdpp")


def load_svdpp(task_name: str) -> ModelResult:
    return load_model_result(task_name, "svdpp")
