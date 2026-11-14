import copy
from typing import NamedTuple

import numpy as np
import torch

from lib.config import DEVICE, RATING_MAX, RATING_MIN
from lib.custom_types import ModelResult, RatingData, RatingTask
from lib.helper_functioins import mse, log


class MiniBatch(NamedTuple):
    user_rows: torch.Tensor             # (B,) user of each rating in the batch
    item_rows: torch.Tensor             # (B,) item of each rating
    ratings: torch.Tensor               # (B,) true stars
    user_penalty_weights: torch.Tensor  # (B,) 1 / #training ratings of that rating's user
    item_penalty_weights: torch.Tensor  # (B,) 1 / #training ratings of that rating's item
    days: torch.Tensor | None           # (B,) day of each rating / None if the data has no days
    timestamps: torch.Tensor | None     # (B,) unix timestamp of each rating / None if the data has no timestamps


# every model gets (user_rows, item_rows); temporal models also the time of every rating:
#   uses_days = True       -> the day (time-aware bias model, timeSVD++)
#   uses_timestamps = True -> the unix timestamp (time-aware factorization machine: the hour of day needs more than the day)
def model_inputs(model, user_rows, item_rows, days, timestamps) -> tuple:
    if model.uses_timestamps:
        if timestamps is None:
            raise ValueError(f"{type(model).__name__} needs the timestamp of every rating -> RatingData.timestamps / RatingTask.test_timestamps")
        return user_rows, item_rows, timestamps
    if not model.uses_days:
        return user_rows, item_rows
    if days is None:
        raise ValueError(f"{type(model).__name__} needs the day of every rating -> RatingData.days / RatingTask.test_days")
    return user_rows, item_rows, days


# Mini-batches of individual ratings in random order (standard SGD sampling):
# every user / item gets many small updates spread over the epoch instead of one big update per epoch
class RatingMiniBatches:

    def __init__(self, rating_data: RatingData, ratings_per_batch: int, seed: int):

        self.user_rows = torch.as_tensor(rating_data.user_rows)
        self.item_rows = torch.as_tensor(rating_data.item_rows)
        self.ratings = torch.as_tensor(rating_data.ratings)
        self.days = None if rating_data.days is None else torch.as_tensor(rating_data.days)
        self.timestamps = None if rating_data.timestamps is None else torch.as_tensor(rating_data.timestamps)

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
                            self.item_penalty_weights[item_rows].to(DEVICE),
                            None if self.days is None else self.days[batch_positions].to(DEVICE),
                            None if self.timestamps is None else self.timestamps[batch_positions].to(DEVICE))


def train_one_epoch(model, optimizer, mini_batches: RatingMiniBatches,
                    user_bias_regularization, item_bias_regularization, factor_regularization) -> float:
    squared_error_sum, n_ratings = 0.0, 0

    for batch in mini_batches:

        # loss = mean squared error + L2 penalty (regularization)
        inputs = model_inputs(model, batch.user_rows, batch.item_rows, batch.days, batch.timestamps)
        squared_errors = (model(*inputs) - batch.ratings) ** 2
        penalty = model.l2_penalty(*inputs, batch.user_penalty_weights, batch.item_penalty_weights,
                                   user_bias_regularization, item_bias_regularization, factor_regularization)
        loss = squared_errors.mean() + penalty

        # gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        squared_error_sum += squared_errors.sum().item()
        n_ratings += len(batch.ratings)

    return squared_error_sum / n_ratings


# predicted stars for every pair (user_rows[k], item_rows[k]) (at days[k] / timestamps[k]) -> clipped to rating range to make sure it makes sense
# in chunks, so predicting all ~1.7M training ratings does not need several GB of memory at once
def predict_ratings(model, user_rows: np.ndarray, item_rows: np.ndarray, days: np.ndarray | None = None,
                    timestamps: np.ndarray | None = None, chunk_size: int = 200_000) -> np.ndarray:

    # a model with a large cost per rating sets its own, smaller chunk (the rating transformer: attention over the history)
    chunk_size = getattr(model, "prediction_chunk_size", chunk_size)
    model.eval()
    prediction_chunks = []
    with torch.no_grad():
        for chunk_start in range(0, len(user_rows), chunk_size):
            chunk = slice(chunk_start, chunk_start + chunk_size)
            chunk_user_rows = torch.as_tensor(user_rows[chunk], device=DEVICE)
            chunk_item_rows = torch.as_tensor(item_rows[chunk], device=DEVICE)
            chunk_days = None if days is None else torch.as_tensor(days[chunk], device=DEVICE)
            chunk_timestamps = None if timestamps is None else torch.as_tensor(timestamps[chunk], device=DEVICE)
            inputs = model_inputs(model, chunk_user_rows, chunk_item_rows, chunk_days, chunk_timestamps)
            prediction_chunks.append(model(*inputs).clamp(RATING_MIN, RATING_MAX).cpu().numpy())

    model.train()
    return np.concatenate(prediction_chunks)


# train MSE and valid MSE of the current weights
def evaluate(model, train_ratings: RatingData, valid_ratings: RatingData) -> tuple[float, float]:
    train_mse = mse(predict_ratings(model, train_ratings.user_rows, train_ratings.item_rows, train_ratings.days,
                                    train_ratings.timestamps), train_ratings.ratings)
    valid_mse = mse(predict_ratings(model, valid_ratings.user_rows, valid_ratings.item_rows, valid_ratings.days,
                                    valid_ratings.timestamps), valid_ratings.ratings)
    return train_mse, valid_mse


# make sure we have early stopping -> MAIN benefit over surprise and the reason I'm putting myself through this
def fit_with_early_stopping(model,
                            train_ratings: RatingData,
                            valid_ratings: RatingData,
                            learning_rate,
                            user_bias_regularization,
                            item_bias_regularization,
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

        train_one_epoch(model, optimizer, mini_batches, user_bias_regularization, item_bias_regularization, factor_regularization)
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
def train_for_epochs(model, rating_data: RatingData, n_epochs, learning_rate,
                     user_bias_regularization, item_bias_regularization, factor_regularization, ratings_per_batch, seed):

    mini_batches = RatingMiniBatches(rating_data, ratings_per_batch, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, n_epochs + 1):
        train_mse = train_one_epoch(model, optimizer, mini_batches, user_bias_regularization, item_bias_regularization, factor_regularization)
        log(f"refit epoch {epoch:3d} | train MSE {train_mse:.4f}")


# the whole pipeline of a PyTorch sub-model: fit with early stopping -> validation predictions (blender input)
# -> optional refit on 100 % for the best number of epochs -> test predictions
# build_model(rating_data) creates a fresh model from the ratings it will train on (biases, N(u), time tables, ...)
def fit_refit_and_predict(task: RatingTask, build_model, max_epochs, early_stopping_patience, refit_on_train_and_valid,
                          **training_options) -> ModelResult:

    # 1) fit on the training data, early-stop on the validation data
    model = build_model(task.training_data)
    epoch_history = fit_with_early_stopping(model, task.training_data, task.validation_data,
                                            max_epochs=max_epochs,
                                            early_stopping_patience=early_stopping_patience,
                                            **training_options)
    validation = task.validation_data
    valid_predictions = predict_ratings(model, validation.user_rows, validation.item_rows, validation.days, validation.timestamps)

    # 2) optional: retrain from scratch on 100 % for the best number of epochs
    #    best epoch 0 = training never helped -> the refit keeps the starting point built from all data
    if refit_on_train_and_valid:
        best_epoch = min(epoch_history, key=lambda epoch_entry: epoch_entry["valid_mse"])["epoch"]
        model = build_model(task.training_and_validation_data)
        train_for_epochs(model, task.training_and_validation_data, n_epochs=best_epoch, **training_options)

    test_predictions = predict_ratings(model, task.test_user_rows, task.test_item_rows, task.test_days, task.test_timestamps)
    return ModelResult(model, epoch_history, valid_predictions, test_predictions)
