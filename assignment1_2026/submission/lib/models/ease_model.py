import numpy as np
import torch

from lib.config import DEVICE
from lib.custom_types import EaseConfig, ModelResult, PairChoiceTask, RatingData
from lib.model_storage import load_model_result, save_model_result
from lib.helper_functioins import log, most_popular_item_columns, sparse_matrix_from_entries
from lib.pair_choice import pair_choice_accuracy, train_pair_scorer

# EASE (Steck 2019, "Embarrassingly Shallow Autoencoders for Sparse Data", WWW): a linear autoencoder
#   score(u, i) = sum over the books j that u has read of B_ji        (= row of u in X times column i of B)
#   B = argmin ||X - X B||^2 + lambda ||B||^2  with diag(B) = 0   (zero diagonal: an item may not predict itself)
#   closed form: P = (X^T X + lambda I)^-1,  B_ij = -P_ij / P_jj (i != j)            (Steck 2019, Eq. 8)
# no training loop: one Gram matrix and one matrix inverse
# memory: B is a dense items x items matrix (51k books = 10.4 GB in float32, peak ~21 GB while inverting)
# -> use_gpu=False computes it in RAM; max_items keeps only the most popular items, the others score NaN
#
# time-weighted history (Task 4, history_time_decay_days > 0): books the user read close to the asked day count more
#   score(u, i, t) = sum_j d_uj(t) B_ji,   d_uj(t) = floor + (1 - floor) * exp(-|t - t_uj| / decay)


class EaseModel:

    def __init__(self, interactions: RatingData, n_user_rows: int, n_item_rows: int, config: EaseConfig):
        self.pairs_per_scoring_chunk = config.pairs_per_scoring_chunk
        self.history_time_decay_days = config.history_time_decay_days
        self.history_time_decay_floor = config.history_time_decay_floor
        # lib/pair_choice.py passes the day of every pair to time-weighted models
        self.uses_days = config.history_time_decay_days > 0
        self.compute_device = DEVICE if config.use_gpu else torch.device("cpu")

        # 1) the max_items most popular items (with at least one interaction) -> column in X and B, others -1
        self.column_of_item, n_kept_items = most_popular_item_columns(interactions.item_rows, n_item_rows, config.max_items)

        # 2) user x kept-item matrix of who read what, stored as day + 1 (day 0 would be dropped as a sparse zero);
        #    X = the same matrix with every entry 1 (interactions with items without a column are left out)
        self.history_day_matrix = sparse_matrix_from_entries(interactions.user_rows, self.column_of_item[interactions.item_rows],
                                                             n_user_rows, n_kept_items, values=interactions.days + 1.0)

        # 3) closed-form weights B
        self.item_weights = self.closed_form_weights(config)

    # X^T X, filled items_per_gram_block columns at a time (the sparse product of all columns at once can need several GB)
    def gram_matrix(self, items_per_block: int) -> np.ndarray:
        read_matrix = self.history_day_matrix.copy()
        read_matrix.data[:] = 1.0
        read_columns = read_matrix.tocsc()

        n_kept_items = read_matrix.shape[1]
        gram = np.empty((n_kept_items, n_kept_items), dtype=np.float32)
        for block_start in range(0, n_kept_items, items_per_block):
            block = slice(block_start, min(block_start + items_per_block, n_kept_items))
            gram[:, block] = (read_matrix.T @ read_columns[:, block]).toarray()
        return gram

    def closed_form_weights(self, config: EaseConfig) -> torch.Tensor:
        gram = torch.as_tensor(self.gram_matrix(config.items_per_gram_block), device=self.compute_device)
        gram.diagonal().add_(config.regularization)

        # X^T X + lambda I is symmetric positive definite -> Cholesky inverse (faster and more stable than a general inverse)
        cholesky_factor = torch.linalg.cholesky(gram)
        del gram
        weights = torch.cholesky_inverse(cholesky_factor)
        del cholesky_factor

        # B_ij = -P_ij / P_jj: every column j divided by -P_jj, then the diagonal set to 0
        weights /= -weights.diagonal().clone()
        weights.fill_diagonal_(0.0)
        return weights

    # sum of B_ji over the user's books j, gathered entry by entry (only the |history| entries of B a pair needs are read);
    # with days (and history_time_decay_days > 0) every book j is weighted by its time distance to the asked day
    def score(self, user_rows: np.ndarray, item_rows: np.ndarray, days: np.ndarray | None = None) -> np.ndarray:
        if self.item_weights is None:
            raise RuntimeError("the EASE weights are not stored with the model (GBs) -> the stored result holds its scores")
        weight_by_time = days is not None and self.history_time_decay_days > 0

        scores = np.full(len(user_rows), np.nan, dtype=np.float32)
        item_columns = self.column_of_item[item_rows]
        positions_with_weights = np.flatnonzero(item_columns >= 0)
        weights_device = self.item_weights.device

        for chunk_start in range(0, len(positions_with_weights), self.pairs_per_scoring_chunk):
            positions = positions_with_weights[chunk_start:chunk_start + self.pairs_per_scoring_chunk]

            # one entry per (pair, book in the user's history): COO row = the pair, column = the book
            histories = self.history_day_matrix[user_rows[positions]].tocoo()
            pair_of_entry = histories.row
            entry_weights = self.item_weights[torch.as_tensor(histories.col, device=weights_device),
                                              torch.as_tensor(item_columns[positions][pair_of_entry], device=weights_device)]

            if weight_by_time:
                days_apart = np.abs(days[positions][pair_of_entry] - (histories.data - 1.0))
                time_decay = self.history_time_decay_floor + (1.0 - self.history_time_decay_floor) * np.exp(-days_apart / self.history_time_decay_days)
                entry_weights = entry_weights * torch.as_tensor(time_decay, dtype=torch.float32, device=weights_device)

            chunk_scores = torch.zeros(len(positions), device=weights_device).index_add_(
                0, torch.as_tensor(pair_of_entry, device=weights_device), entry_weights)
            scores[positions] = chunk_scores.cpu().numpy()

        return scores

    # the dense weights are only needed while scoring -> dropped before the model is stored
    def release_item_weights(self):
        self.item_weights = None


def train_ease_model(task: PairChoiceTask, config: EaseConfig, model_name: str = "EASE") -> ModelResult:
    model_result = train_pair_scorer(task, lambda interactions: EaseModel(interactions, task.n_user_rows, task.n_item_rows, config), model_name)
    model_result.model.release_item_weights()
    return model_result


# plain and time-weighted EASE from the same weights B (one inverse per stage instead of two; Task 5: 50k books in RAM)
# -> (plain EASE result, time-weighted EASE result)
def train_plain_and_time_weighted_ease(task: PairChoiceTask, config: EaseConfig) -> tuple[ModelResult, ModelResult]:

    def pair_scores(model: EaseModel, pairs, weight_by_time: bool) -> np.ndarray:
        days = pairs.days if weight_by_time else None
        return np.column_stack([model.score(pairs.user_rows, pairs.first_item_rows, days),
                                model.score(pairs.user_rows, pairs.second_item_rows, days)]).astype(np.float32)

    # 1) weights from the training split -> validation scores of both variants
    model = EaseModel(task.training_data, task.n_user_rows, task.n_item_rows, config)
    valid_scores = {weight_by_time: pair_scores(model, task.validation_pairs, weight_by_time) for weight_by_time in (False, True)}
    for weight_by_time, name in ((False, "EASE"), (True, "time-weighted EASE")):
        log(f"{name} | validation accuracy {pair_choice_accuracy(valid_scores[weight_by_time], task.validation_pairs.first_is_positive):.4f}")

    # 2) weights from every interaction -> test scores of both variants
    del model
    model = EaseModel(task.training_and_validation_data, task.n_user_rows, task.n_item_rows, config)
    test_scores = {weight_by_time: pair_scores(model, task.test_pairs, weight_by_time) for weight_by_time in (False, True)}
    model.release_item_weights()

    return (ModelResult(model, [], valid_scores[False], test_scores[False]),
            ModelResult(model, [], valid_scores[True], test_scores[True]))


def store_ease_model(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "ease")


def load_ease_model(task_name: str) -> ModelResult:
    return load_model_result(task_name, "ease")


# Task 4: EASE with the time-weighted history, stored as its own sub-model
def store_time_ease_model(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "time_ease")


def load_time_ease_model(task_name: str) -> ModelResult:
    return load_model_result(task_name, "time_ease")
