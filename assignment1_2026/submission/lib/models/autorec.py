import copy

import numpy as np
import torch
import torch.nn as nn

from lib.config import DEVICE, RATING_MAX, RATING_MIN
from lib.custom_types import AutoRecConfig, ModelResult, RatingData, RatingTask
from lib.helper_functioins import fit_closed_form_biases, log, mse, sparse_matrix_from_entries
from lib.model_storage import load_model_result, save_model_result

# I-AutoRec (Sedhain et al. 2015, "AutoRec: Autoencoders Meet Collaborative Filtering", WWW), item-based autoencoder,
# here on the residuals of the closed-form bias model (like the item kNN):
#   b_ui    = mu + b_u + b_i                    closed-form baseline, fixed
#   x_i     = the residuals r_ui - b_ui book i received from every user (users who did not rate it: 0)    (U+1,)
#   h_i     = tanh(V x_i + b)                                                                             (hidden,)
#   r(u, i) = b_ui + (W h_i)_u                                                       (identity output layer, no bias)
#   loss    = mean over the hidden (u, i) of the batch of (r_ui - r(u, i))^2 + lambda / 2 (||W||^2 + ||V||^2) / N
#             (masked training, see train_one_epoch; N = number of training ratings -> lambda on the scale of the paper's sum loss)
# why the residuals: centred on mu only (as in the paper), the check on Task 1 stayed far behind the bias model (valid MSE
# 0.824 vs 0.763, hidden 200): a book with few ratings has an almost empty x_i, so its own bias was nearly lost
# two more changes to the paper, both for the same check: with sigmoid every book starts at h_i ~ 0.5, so all hidden units
# moved together and Adam overshot (worse than the bias model already on the training ratings); tanh starts at h_i ~ 0.
# no output bias c_u: it would be a second, unregularised user bias next to b_u
# masked training instead of the paper's full input: with every rating in its own input the network learned to copy it
# (training MSE 0.32 after 2 epochs, validation worse every epoch)
# the decoder starts at 0 -> epoch 0 = the bias model
# item-based because I-AutoRec beat the user-based version in the paper; non-linear -> its errors differ from the factor models
# one epoch = every book with ratings once, in random order, books_per_batch at a time


class AutoRecNetwork(nn.Module):

    def __init__(self, n_user_rows: int, hidden_dimension: int):
        super().__init__()
        self.encoder = nn.Linear(n_user_rows, hidden_dimension)
        self.decoder = nn.Linear(hidden_dimension, n_user_rows, bias=False)
        nn.init.zeros_(self.decoder.weight)     # output 0 = the baseline

    # (B, U+1) residuals of B books -> (B, U+1) predicted residuals of every user
    def forward(self, residuals):
        return self.decoder(torch.tanh(self.encoder(residuals)))

    def squared_weight_norm(self):
        return (self.encoder.weight ** 2).sum() + (self.decoder.weight ** 2).sum()


class AutoRecModel:

    def __init__(self, training_ratings: RatingData, n_user_rows: int, n_item_rows: int, config: AutoRecConfig):
        self.config = config
        user_rows, item_rows, ratings = training_ratings.user_rows, training_ratings.item_rows, training_ratings.ratings
        self.n_training_ratings = len(ratings)

        # 1) closed-form baseline mu + b_u + b_i (fixed) and the residuals it leaves
        self.global_mean_rating = float(ratings.mean())
        self.user_biases, self.item_biases = fit_closed_form_biases(
            user_rows, item_rows, ratings, n_user_rows, n_item_rows, self.global_mean_rating,
            config.user_bias_regularization, config.item_bias_regularization)
        residuals = ratings - self.baseline(user_rows, item_rows)

        # 2) book x user matrices of the residuals and of "rated" (a residual can be exactly 0)
        #    a pair rated twice is summed in a sparse matrix -> divide by the count (same entries, same order), then rated = 1
        self.residual_matrix = sparse_matrix_from_entries(item_rows, user_rows, n_item_rows, n_user_rows, values=residuals)
        self.rated_matrix = sparse_matrix_from_entries(item_rows, user_rows, n_item_rows, n_user_rows)
        self.residual_matrix.data /= self.rated_matrix.data
        self.rated_matrix.data[:] = 1.0
        self.items_with_ratings = np.flatnonzero(self.rated_matrix.getnnz(axis=1))

        torch.manual_seed(config.seed)  # identical initial weights for the fit and the refit
        self.network = AutoRecNetwork(n_user_rows, config.hidden_dimension).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.shuffle_generator = np.random.default_rng(config.seed)

    def baseline(self, user_rows: np.ndarray, item_rows: np.ndarray) -> np.ndarray:
        return (self.global_mean_rating + self.user_biases[user_rows] + self.item_biases[item_rows]).astype(np.float32)

    # (B, U+1) dense rows of a sparse book x user matrix on the device
    def dense_rows(self, matrix, item_rows: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(matrix[item_rows].toarray(), device=DEVICE)

    # one pass over every book with ratings, in random order -> mean squared error of the hidden (= predicted) training ratings
    # masked training: in every batch a random masked_fraction of the ratings is hidden from the input and only those are
    # predicted -> a rating never predicts itself (as for the validation / test pairs, which are never in the input)
    def train_one_epoch(self) -> float:
        self.network.train()
        squared_error_sum, n_hidden = 0.0, 0

        shuffled_items = self.shuffle_generator.permutation(self.items_with_ratings)
        for batch_start in range(0, len(shuffled_items), self.config.books_per_batch):
            batch_items = shuffled_items[batch_start:batch_start + self.config.books_per_batch]
            residuals = self.dense_rows(self.residual_matrix, batch_items)
            is_rated = self.dense_rows(self.rated_matrix, batch_items)
            is_hidden = (torch.rand_like(is_rated) < self.config.masked_fraction) * is_rated
            # visible ratings scaled up as in inverted dropout -> the input has the size it has at prediction time (all ratings)
            visible_residuals = residuals * (is_rated - is_hidden) / (1.0 - self.config.masked_fraction)

            squared_errors = ((self.network(visible_residuals) - residuals) ** 2) * is_hidden
            loss = (squared_errors.sum() / is_hidden.sum().clamp(min=1.0)
                    + self.config.regularization / 2 * self.network.squared_weight_norm() / self.n_training_ratings)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            squared_error_sum, n_hidden = squared_error_sum + squared_errors.sum().item(), n_hidden + int(is_hidden.sum().item())

        return squared_error_sum / n_hidden

    # predicted stars for every pair (user_rows[k], item_rows[k]): every asked book is encoded once
    def predict(self, user_rows: np.ndarray, item_rows: np.ndarray) -> np.ndarray:
        self.network.eval()
        predictions = np.empty(len(user_rows), dtype=np.float32)
        distinct_items, item_index_of_pair = np.unique(item_rows, return_inverse=True)

        with torch.no_grad():
            for batch_start in range(0, len(distinct_items), self.config.books_per_batch):
                batch_items = distinct_items[batch_start:batch_start + self.config.books_per_batch]
                outputs = self.network(self.dense_rows(self.residual_matrix, batch_items))
                batch_pairs = np.flatnonzero((item_index_of_pair >= batch_start) & (item_index_of_pair < batch_start + len(batch_items)))
                gathered = outputs[torch.as_tensor(item_index_of_pair[batch_pairs] - batch_start, device=DEVICE),
                                   torch.as_tensor(user_rows[batch_pairs], device=DEVICE)]
                predictions[batch_pairs] = gathered.cpu().numpy()

        return np.clip(self.baseline(user_rows, item_rows) + predictions, RATING_MIN, RATING_MAX).astype(np.float32)


def train_autorec(task: RatingTask, config: AutoRecConfig) -> ModelResult:
    validation = task.validation_data

    # 1) fit on the training split, early stopping on the validation MSE
    model = AutoRecModel(task.training_data, task.n_user_rows, task.n_item_rows, config)
    valid_mse = mse(model.predict(validation.user_rows, validation.item_rows), validation.ratings)
    log(f"AutoRec epoch   0 | valid MSE {valid_mse:.4f}  (starting point = bias model)")
    epoch_history, best_valid_mse, best_weights, epochs_without_improvement = [], float("inf"), None, 0
    for epoch in range(1, config.max_epochs + 1):
        train_mse = model.train_one_epoch()
        valid_mse = mse(model.predict(validation.user_rows, validation.item_rows), validation.ratings)
        epoch_history.append({"epoch": epoch, "train_mse": train_mse, "valid_mse": valid_mse})
        log(f"AutoRec epoch {epoch:3d} | train MSE {train_mse:.4f} | valid MSE {valid_mse:.4f}")

        if valid_mse < best_valid_mse:
            best_valid_mse, best_weights, epochs_without_improvement = valid_mse, copy.deepcopy(model.network.state_dict()), 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == config.early_stopping_patience:
                break

    model.network.load_state_dict(best_weights)
    best_epoch = min(epoch_history, key=lambda epoch_entry: epoch_entry["valid_mse"])["epoch"]
    log(f"AutoRec | best valid MSE {best_valid_mse:.4f} at epoch {best_epoch}")
    valid_predictions = model.predict(validation.user_rows, validation.item_rows)

    # 2) optional: refit from scratch on 100 % for the best number of epochs -> test predictions
    if config.refit_on_train_and_valid:
        del model
        model = AutoRecModel(task.training_and_validation_data, task.n_user_rows, task.n_item_rows, config)
        for epoch in range(1, best_epoch + 1):
            log(f"AutoRec refit epoch {epoch:3d} | train MSE {model.train_one_epoch():.4f}")
    test_predictions = model.predict(task.test_user_rows, task.test_item_rows)

    # the optimizer state is not needed after training (as large as the network twice)
    model.optimizer = None
    return ModelResult(model, epoch_history, valid_predictions, test_predictions)


def store_autorec(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "autorec")


def load_autorec(task_name: str) -> ModelResult:
    return load_model_result(task_name, "autorec")
