import numpy as np
import pandas as pd
import scipy.sparse as sp
import math
import time

from lib.custom_types import RatingData


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Indexing & Data Structuring
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# build the embedding IDs based on the order the IDs of the entries appear in the data
# INDEX: distinct str ids (csv) in order of first appearance -> position + 1 = embedding id
def number_by_first_appearance(ids_in_file_order) -> pd.Index:
    return pd.Index(pd.unique(pd.Series(ids_in_file_order)))

# mapping from str id to embedding rows
def ids_to_rows(ids, row_of_id: pd.Index) -> np.ndarray:
    # get_indexer gives -1 for ids that are not in the index -> + 1 = row 0 (unknowns to 0!)
    return (row_of_id.get_indexer(ids) + 1).astype(np.int64)

# the one way every sparse matrix of the project is built: one (row, column, value) entry per interaction
# built in COO format, stored as CSR in canonical form (duplicates summed, indices sorted)
# -> fast row slices (the rows of the asking users), sparse products and element lookups A[rows, columns]
# values: one per entry, or one number for all (default 1 = "has interacted"); entries with column -1 are left out
# Source: SciPy docs, scipy.sparse: COO is the recommended format for construction, CSR for arithmetic and row slicing
def sparse_matrix_from_entries(row_indices: np.ndarray, column_indices: np.ndarray, n_rows: int, n_columns: int,
                               values: np.ndarray | float = 1.0) -> sp.csr_matrix:
    values = np.broadcast_to(np.asarray(values, dtype=np.float32), np.shape(row_indices))
    has_column = column_indices >= 0
    return sp.coo_matrix((values[has_column], (row_indices[has_column], column_indices[has_column])),
                         shape=(n_rows, n_columns)).tocsr()

# the max_items most interacted-with items (with at least one interaction) get a column 0 .. n_kept_items - 1, all others -1
# (EASE / Mult-VAE: a dense items x items matrix or network layer only fits for the popular items)
def most_popular_item_columns(item_rows: np.ndarray, n_item_rows: int, max_items: int) -> tuple[np.ndarray, int]:
    interactions_per_item = np.bincount(item_rows, minlength=n_item_rows)
    kept_items = np.argsort(-interactions_per_item, kind="stable")[:max_items]
    kept_items = kept_items[interactions_per_item[kept_items] > 0]
    column_of_item = np.full(n_item_rows, -1, dtype=np.int64)
    column_of_item[kept_items] = np.arange(len(kept_items))
    return column_of_item, len(kept_items)

# the last sequence_length books of every user, oldest first, left-padded with 0 (interactions chronological per user)
# (SASRec's input windows, the transition model's context)
def recent_sequences(interactions: RatingData, n_user_rows: int, sequence_length: int) -> np.ndarray:
    user_rows, item_rows = interactions.user_rows, interactions.item_rows
    sequences = np.zeros((n_user_rows, sequence_length), dtype=np.int64)
    last_positions = np.flatnonzero(np.append(user_rows[1:] != user_rows[:-1], True))
    for steps_back in range(sequence_length):
        positions = last_positions - steps_back
        is_same_user = (positions >= 0) & (user_rows[np.maximum(positions, 0)] == user_rows[last_positions])
        sequences[user_rows[last_positions[is_same_user]], sequence_length - 1 - steps_back] = item_rows[positions[is_same_user]]
    return sequences


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Closed-form Bias Model
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# best b_u, b_i for r = mu + b_u + b_i under
#   sum(errors^2) + user_bias_regularization * sum b_u^2 + item_bias_regularization * sum b_i^2
# -> alternating exact updates: b = sum of residuals / (number of ratings + regularization of that side)
# one lambda per side: Koren's baseline estimates shrink b_i and b_u with different lambdas
# Source: Koren 2008, "Factorization Meets the Neighborhood" (KDD), Sec. 2.1 baseline estimates; PML Ch. 5
def fit_closed_form_biases(user_rows: np.ndarray, item_rows: np.ndarray, ratings: np.ndarray,
                           n_user_rows: int, n_item_rows: int, global_mean_rating: float,
                           user_bias_regularization: float, item_bias_regularization: float,
                           n_iterations: int = 10) -> tuple[np.ndarray, np.ndarray]:

    ratings_per_user = np.bincount(user_rows, minlength=n_user_rows)
    ratings_per_item = np.bincount(item_rows, minlength=n_item_rows)
    user_biases = np.zeros(n_user_rows)

    for _ in range(n_iterations):
        # item bias = shrunken mean of what is left after mu and the user bias
        item_residual_sums = np.bincount(item_rows, weights=ratings - global_mean_rating - user_biases[user_rows], minlength=n_item_rows)
        item_biases = item_residual_sums / (ratings_per_item + item_bias_regularization)

        # user bias = shrunken mean of what is left after mu and the item bias
        user_residual_sums = np.bincount(user_rows, weights=ratings - global_mean_rating - item_biases[item_rows], minlength=n_user_rows)
        user_biases = user_residual_sums / (ratings_per_user + user_bias_regularization)

    # rows without ratings (row 0 = unknown) get 0 / regularization = 0
    return user_biases, item_biases


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Errors, Math Operations
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# intersection over union
def jaccard_sim(set_a: set, set_b: set) -> float:
    intersection = len(set_a & set_b)
    union = len(set_a) + len(set_b) - intersection
    return intersection / union

def cosine_sim_set(set_a: set, set_b: set) -> float:
    pass

# np.ndarray
def cosine_sim(vec_a: dict, vec_b: dict) -> float:
    pass

def pearson_sim(set_a: set, set_b: set) -> float:
    pass

# mean squared error
def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2))

# root mean squared error
def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return math.sqrt(mse(predictions, targets))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Infrastructure & Data Fetching
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# time-stamped log print
def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)