
from typing import Any, NamedTuple

import numpy as np

# Comments for Documentation: ------
# U = number of users in training file ~ 30k
# I = number of items in training file ~ 51k
# N = number of ratings in Rating Data ~ 1.67 Mil
# N_test = number of rows in the Test file
# K = index -> to N

# The users and items are indexed in the embedding vectors by their order of appearance


class RatingData(NamedTuple):
    # one rating per position k: user user_rows[k] -> gets the rating from ratings[k] for item item_rows[k]
    user_rows: np.ndarray                   # (N,)  int    values: 1-U          // which user gave rating k
    item_rows: np.ndarray                   # (N,)  int    values: 1-I          // which item was in rating k
    ratings: np.ndarray                     # (N,)  float  values: 0-5          // what was the score of rating k


class RatingTask(NamedTuple):
    training_data: RatingData               # random cut of the full data for training       
    validation_data: RatingData             # random cut of the full data for validation
    training_and_validation_data: RatingData  # entire data available for training & validation
    test_user_rows: np.ndarray              # (N_test,)  int    values: 0-U     // user for the test row, 0 for unknown!!
    test_item_rows: np.ndarray              # (N_test,)  int    values: 0-I     // item for the test row, 0 for unknown!!
    n_user_rows: int                        # U + 1 = rows of every user embedding table (row 0 = unknown)
    n_item_rows: int                        # I + 1 = rows of every item embedding table (row 0 = unknown)
    # N(u) for SVD++ is not stored here: the pipeline builds it from the ratings a model trains on (no test data)
    

# The return type of the model
class ModelResult(NamedTuple):
    model: Any                              # model of any kind
    history: list                           # one dict per epoch {"epoch", "train_mse", "valid_mse"}, epoch 0 = starting point / empty if not iterative
    valid_predictions: np.ndarray           # (N_valid,)    float   values: e.g. 0-5 for the rating task / standard interface for merging
    test_predictions: np.ndarray            # (N_test,)     float   values: e.g. 0-5