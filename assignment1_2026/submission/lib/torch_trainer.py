import copy
from typing import NamedTuple

import numpy as np
import torch

from lib.config import DEVICE, RATING_MAX, RATING_MIN
from lib.custom_types import RatingData
from lib.helper_functioins import mse, log


class MiniBatch(NamedTuple):
    user_rows: torch.Tensor             # (B,) user of each rating in the batch
    item_rows: torch.Tensor             # (B,) item of each rating
    ratings: torch.Tensor               # (B,) true stars
    user_penalty_weights: torch.Tensor  # (B,) 1 / #training ratings of that rating's user
    item_penalty_weights: torch.Tensor  # (B,) 1 / #training ratings of that rating's item


# Mini-batches of individual ratings in random order (standard SGD sampling):
# every user / item gets many small updates spread over the epoch instead of one big update per epoch
class RatingMiniBatches:

    def __init__(self, rating_data: RatingData, ratings_per_batch: int, seed: int):

        self.user_rows = torch.as_tensor(rating_data.user_rows)
        self.item_rows = torch.as_tensor(rating_data.item_rows)
        self.ratings = torch.as_tensor(rating_data.ratings)

        # 1 / count spreads each user's / item's L2 penalty over its ratings (see LatentFactorModel.l2_penalty)
        ratings_per_user = np.bincount(rating_data.user_rows)
        ratings_per_item = np.bincount(rating_data.item_rows)
        self.user_penalty_weights = torch.as_tensor(1.0 / np.maximum(ratings_per_user, 1), dtype=torch.float32)
        self.item_penalty_weights = torch.as_tensor(1.0 / np.maximum(ratings_per_item, 1), dtype=torch.float32)

        self.ratings_per_batch = ratings_per_batch
        self.shuffle_generator = torch.Generator().manual_seed(seed)

    def __iter__(self):

        # new random order every epoch, then cut into consecutive batches
        shuffled_positions = torch.randperm(len(self.ratings), generator=self.shuffle_generator)

        for batch_positions in shuffled_positions.split(self.ratings_per_batch):
            user_rows = self.user_rows[batch_positions]
            item_rows = self.item_rows[batch_positions]

            yield MiniBatch(user_rows.to(DEVICE),
                            item_rows.to(DEVICE),
                            self.ratings[batch_positions].to(DEVICE),
                            self.user_penalty_weights[user_rows].to(DEVICE),
                            self.item_penalty_weights[item_rows].to(DEVICE))


def train_one_epoch(model, optimizer, mini_batches: RatingMiniBatches, bias_regularization, factor_regularization) -> float:
    squared_error_sum, n_ratings = 0.0, 0

    for batch in mini_batches:

        # loss = mean squared error + L2 penalty (regularization)
        squared_errors = (model(batch.user_rows, batch.item_rows) - batch.ratings) ** 2
        penalty = model.l2_penalty(batch.user_rows, batch.item_rows, batch.user_penalty_weights, batch.item_penalty_weights, bias_regularization, factor_regularization)
        loss = squared_errors.mean() + penalty

        # gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        squared_error_sum += squared_errors.sum().item()
        n_ratings += len(batch.ratings)

    return squared_error_sum / n_ratings


# predicted stars for every pair (user_rows[k], item_rows[k]) -> clipped to rating range to make sure it makes sense
# in chunks, so predicting all ~1.7M training ratings does not need several GB of memory at once
def predict_ratings(model, user_rows: np.ndarray, item_rows: np.ndarray, chunk_size: int = 200_000) -> np.ndarray:

    model.eval()
    prediction_chunks = []
    with torch.no_grad():
        for chunk_start in range(0, len(user_rows), chunk_size):
            chunk_user_rows = torch.as_tensor(user_rows[chunk_start:chunk_start + chunk_size], device=DEVICE)
            chunk_item_rows = torch.as_tensor(item_rows[chunk_start:chunk_start + chunk_size], device=DEVICE)
            prediction_chunks.append(model(chunk_user_rows, chunk_item_rows).clamp(RATING_MIN, RATING_MAX).cpu().numpy())

    model.train()
    return np.concatenate(prediction_chunks)


# train MSE and valid MSE of the current weights
def evaluate(model, train_ratings: RatingData, valid_ratings: RatingData) -> tuple[float, float]:
    train_mse = mse(predict_ratings(model, train_ratings.user_rows, train_ratings.item_rows), train_ratings.ratings)
    valid_mse = mse(predict_ratings(model, valid_ratings.user_rows, valid_ratings.item_rows), valid_ratings.ratings)
    return train_mse, valid_mse


# make sure we have early stopping -> MAIN benefit over surprise and the reason I'm putting myself through this
def fit_with_early_stopping(model,
                            train_ratings: RatingData,
                            valid_ratings: RatingData,
                            learning_rate,
                            bias_regularization,
                            factor_regularization,
                            ratings_per_batch,
                            max_epochs,
                            early_stopping_patience,
                            seed) -> list[dict]:

    mini_batches = RatingMiniBatches(train_ratings, ratings_per_batch, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # epoch 0 = the starting point (closed-form biases, tiny factors) -> the final model is never worse than that
    train_mse, valid_mse = evaluate(model, train_ratings, valid_ratings)
    epoch_history = [{"epoch": 0, "train_mse": train_mse, "valid_mse": valid_mse}]
    log(f"epoch   0 | train MSE {train_mse:.4f} | valid MSE {valid_mse:.4f}  (starting point)")
    best_valid_mse, best_weights, epochs_without_improvement = valid_mse, copy.deepcopy(model.state_dict()), 0

    # one training pass + validation score
    for epoch in range(1, max_epochs + 1):

        train_one_epoch(model, optimizer, mini_batches, bias_regularization, factor_regularization)
        # both MSEs measured with the weights at the END of the epoch -> directly comparable
        train_mse, valid_mse = evaluate(model, train_ratings, valid_ratings)

        epoch_history.append({"epoch": epoch, "train_mse": train_mse, "valid_mse": valid_mse})

        log(f"epoch {epoch:3d} | train MSE {train_mse:.4f} | valid MSE {valid_mse:.4f}")

        # early stopping: keep the best weights, stop after `early_stopping_patience` epochs without improvement
        if valid_mse < best_valid_mse:
            best_valid_mse, best_weights, epochs_without_improvement = valid_mse, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_improvement += 1
            # go here if we're not improving / epoch patience is defined
            if epochs_without_improvement == early_stopping_patience:
                break

    model.load_state_dict(best_weights)
    best_epoch = min(epoch_history, key=lambda epoch_entry: epoch_entry["valid_mse"])["epoch"]
    log(f"best valid MSE {best_valid_mse:.4f} at epoch {best_epoch}")
    return epoch_history


# main training handler (fixed number of epochs) -> used for the final refit on training + validation data
def train_for_epochs(model, rating_data: RatingData, n_epochs, learning_rate, bias_regularization, factor_regularization, ratings_per_batch, seed):

    mini_batches = RatingMiniBatches(rating_data, ratings_per_batch, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, n_epochs + 1):
        train_mse = train_one_epoch(model, optimizer, mini_batches, bias_regularization, factor_regularization)
        log(f"refit epoch {epoch:3d} | train MSE {train_mse:.4f}")
