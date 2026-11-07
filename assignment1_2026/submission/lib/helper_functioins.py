import numpy as np
import math
import time


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Indexing & Data Structuring
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# build the embedding IDs based on the order the IDs of the entries appear in the data
# DICT: mapping from str id (csv) tp embedding id
def number_by_first_appearance(ids_in_file_order) -> dict[str, int]:
    distinct_ids = dict.fromkeys(ids_in_file_order)
    return {user_or_item_id: row for row, user_or_item_id in enumerate(distinct_ids, start=1)}

# mapping from str id to embedding rows
def ids_to_rows(ids, row_of_id: dict[str, int]) -> np.ndarray:
    return np.array([row_of_id.get(user_or_item_id, 0) for user_or_item_id in ids], dtype=np.int64)

# obtain all entries belonging to a user
def group_by_user(user_rows: np.ndarray, values: np.ndarray, n_user_rows: int) -> list[np.ndarray]:
    sorted_values = values[np.argsort(user_rows, kind="stable")]
    values_per_user = np.bincount(user_rows, minlength=n_user_rows)
    return np.split(sorted_values, np.cumsum(values_per_user)[:-1])


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Closed-form Bias Model
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# best b_u, b_i for r = mu + b_u + b_i under sum(errors^2) + regularization * (sum b_u^2 + sum b_i^2)
# -> alternating exact updates: b = sum of residuals / (number of ratings + regularization)
# Source: Koren 2008, "Factorization Meets the Neighborhood" (KDD), Sec. 2.1 baseline estimates; PML Ch. 5
def fit_closed_form_biases(user_rows: np.ndarray, item_rows: np.ndarray, ratings: np.ndarray,
                           n_user_rows: int, n_item_rows: int, global_mean_rating: float,
                           regularization: float, n_iterations: int = 10) -> tuple[np.ndarray, np.ndarray]:

    ratings_per_user = np.bincount(user_rows, minlength=n_user_rows)
    ratings_per_item = np.bincount(item_rows, minlength=n_item_rows)
    user_biases = np.zeros(n_user_rows)

    for _ in range(n_iterations):
        # item bias = shrunken mean of what is left after mu and the user bias
        item_residual_sums = np.bincount(item_rows, weights=ratings - global_mean_rating - user_biases[user_rows], minlength=n_item_rows)
        item_biases = item_residual_sums / (ratings_per_item + regularization)

        # user bias = shrunken mean of what is left after mu and the item bias
        user_residual_sums = np.bincount(user_rows, weights=ratings - global_mean_rating - item_biases[item_rows], minlength=n_user_rows)
        user_biases = user_residual_sums / (ratings_per_user + regularization)

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