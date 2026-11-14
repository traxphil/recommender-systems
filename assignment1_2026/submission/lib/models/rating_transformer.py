import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from lib.config import DEVICE
from lib.custom_types import ModelResult, RatingData, RatingTask, RatingTransformerConfig
from lib.data_loading import known_interaction_pairs
from lib.helper_functioins import fit_closed_form_biases, sparse_matrix_from_entries
from lib.model_storage import load_model_result, save_model_result
from lib.torch_trainer import fit_refit_and_predict

# Rating transformer: self-attention over the books a user has rated, asked about one more book (a set transformer;
# Vaswani et al. 2017 encoder blocks, as in SASRec / BERT4Rec but without positions: Task 1 has no order)
#   history tokens  t_j = e_j + e_rating(r_uj)       book embedding + embedding of the stars the user gave it
#                   (rating "unknown" for known pairs without a rating, e.g. the test pairs, with use_test_pairs_in_history)
#   query token     t_q = e_i + e_query              the asked book
#   h               = transformer encoder over [t_q, t_j for the user's books]
#   r(u, i)         = mu + b_u + b_i + w . h_query
# the asked book attends to the user's books -> a learned, non-linear mix of item kNN (attention = learned book-to-book
# similarity) and SVD++ (the history as the user representation)
# the asked book is always hidden from the user's history -> a training rating can never be copied from its own token
# histories: at most max_history_length books per user (a random, seeded subset of longer histories)
# biases start at the closed-form bias model, the output layer at 0 -> epoch 0 = the bias baseline

N_STAR_VALUES = 6                           # ratings 0 .. 5
UNKNOWN_RATING_TOKEN = 6                    # a known pair whose rating the model may not see (validation / test pair)
QUERY_TOKEN = 7                             # marks the asked book


# (U+1, L) books and rating tokens of every user's history: a random subset of at most L pairs per user, 0 = padding
def user_histories(history_pairs: RatingData, n_user_rows: int, max_history_length: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    random_order = np.random.default_rng(seed).permutation(len(history_pairs.user_rows))
    user_rows, item_rows, ratings = (column[random_order] for column in history_pairs[:3])

    # position of every pair within its user's history (in the random order); pairs beyond L are left out
    position_in_history = pd.Series(user_rows).groupby(user_rows).cumcount().to_numpy()
    is_kept = position_in_history < max_history_length

    history_items = np.zeros((n_user_rows, max_history_length), dtype=np.int64)
    history_rating_tokens = np.zeros((n_user_rows, max_history_length), dtype=np.int64)
    history_items[user_rows[is_kept], position_in_history[is_kept]] = item_rows[is_kept]
    history_rating_tokens[user_rows[is_kept], position_in_history[is_kept]] = np.where(
        np.isnan(ratings[is_kept]), UNKNOWN_RATING_TOKEN, np.clip(np.round(np.nan_to_num(ratings[is_kept])), 0, N_STAR_VALUES - 1))
    return history_items, history_rating_tokens


# the pairs a model built from these ratings may show in the histories: the ratings themselves, plus (optionally) every
# other known pair of the training + test file with its rating hidden (validation pairs during the fit, the test pairs)
def history_pairs(training_ratings: RatingData, task: RatingTask, use_test_pairs_in_history: bool) -> RatingData:
    if not use_test_pairs_in_history:
        return training_ratings
    known_pairs = known_interaction_pairs(task)

    # the visible rating of every known pair: mean of its training ratings (a pair rated twice is summed in a sparse matrix)
    def looked_up(values) -> np.ndarray:
        matrix = sparse_matrix_from_entries(training_ratings.user_rows, training_ratings.item_rows,
                                            task.n_user_rows, task.n_item_rows, values)
        return np.asarray(matrix[known_pairs.user_rows, known_pairs.item_rows]).ravel()
    rating_sums, rating_counts = looked_up(training_ratings.ratings), looked_up(1.0)   # count 0 = no rating it may see
    visible_ratings = np.where(rating_counts > 0, rating_sums / np.maximum(rating_counts, 1.0), np.nan)
    return RatingData(known_pairs.user_rows, known_pairs.item_rows, visible_ratings)


class RatingTransformer(nn.Module):

    # no time terms -> lib/torch_trainer.py only passes (user_rows, item_rows)
    uses_days = False
    uses_timestamps = False
    # predict_ratings: every rating attends over a whole history -> 4096 ratings per chunk instead of 200 000 (GPU memory)
    prediction_chunk_size = 4096

    # Parameters:
    # item_embeddings       e_j         (I+1, d)
    # token_embeddings      e_rating    (6 star values + unknown + query, d)
    # encoder               n_blocks transformer encoder blocks (pre-norm)
    # output                w           (d -> 1), starts at 0
    # user_biases / item_biases  b_u / b_i  (closed-form start)
    # buffers: history_items / history_rating_tokens (U+1, L)

    def __init__(self, history_items: np.ndarray, history_rating_tokens: np.ndarray, n_user_rows: int, n_item_rows: int,
                 config: RatingTransformerConfig, global_mean_rating: float, initial_user_biases: np.ndarray,
                 initial_item_biases: np.ndarray):

        super().__init__()
        self.global_mean_rating = global_mean_rating
        dimension = config.embedding_dimension

        self.item_embeddings = nn.Embedding(n_item_rows, dimension, padding_idx=0)
        self.token_embeddings = nn.Embedding(QUERY_TOKEN + 1, dimension)
        encoder_block = nn.TransformerEncoderLayer(d_model=dimension, nhead=config.n_heads, dim_feedforward=4 * dimension,
                                                   dropout=config.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_block, num_layers=config.n_blocks, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(dimension)
        self.output = nn.Linear(dimension, 1)
        nn.init.normal_(self.item_embeddings.weight, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        # biases start at the closed-form bias model of the training ratings
        self.user_biases = nn.Embedding(n_user_rows, 1)
        self.item_biases = nn.Embedding(n_item_rows, 1)
        with torch.no_grad():
            self.user_biases.weight[:, 0] = torch.as_tensor(initial_user_biases, dtype=torch.float32)
            self.item_biases.weight[:, 0] = torch.as_tensor(initial_item_biases, dtype=torch.float32)

        self.register_buffer("history_items", torch.as_tensor(history_items))
        self.register_buffer("history_rating_tokens", torch.as_tensor(history_rating_tokens))

    def forward(self, user_rows, item_rows):

        # 1) the user's history; the asked book is always hidden (and padding = book 0)
        history_items = self.history_items[user_rows]
        is_hidden = (history_items == 0) | (history_items == item_rows.unsqueeze(1))
        history_tokens = self.item_embeddings(history_items) + self.token_embeddings(self.history_rating_tokens[user_rows])

        # 2) query token first, then the history; attention ignores the hidden positions
        query_tokens = self.item_embeddings(item_rows) + self.token_embeddings.weight[QUERY_TOKEN]
        tokens = torch.cat([query_tokens.unsqueeze(1), history_tokens], dim=1)
        ignored_positions = torch.cat([torch.zeros_like(is_hidden[:, :1]), is_hidden], dim=1)
        query_state = self.encoder(tokens, src_key_padding_mask=ignored_positions)[:, 0]

        # 3) rating = baseline + what the attention over the history adds
        biases = self.user_biases(user_rows).squeeze(1) + self.item_biases(item_rows).squeeze(1)
        return self.global_mean_rating + biases + self.output(self.output_norm(query_state)).squeeze(1)

    # biases as in LatentFactorModel.l2_penalty (spread over the ratings of a user / item); the asked book's embedding like
    # an item factor; the encoder is regularised by dropout
    def l2_penalty(self, user_rows, item_rows, user_penalty_weights, item_penalty_weights,
                   user_bias_regularization, item_bias_regularization, factor_regularization):
        user_bias_squared = self.user_biases(user_rows).squeeze(1) ** 2
        item_bias_squared = self.item_biases(item_rows).squeeze(1) ** 2
        item_embedding_squared = (self.item_embeddings(item_rows) ** 2).sum(dim=1)
        return (user_bias_regularization * user_penalty_weights * user_bias_squared
                + item_bias_regularization * item_penalty_weights * item_bias_squared
                + factor_regularization * item_penalty_weights * item_embedding_squared).mean()


def train_rating_transformer(task: RatingTask, config: RatingTransformerConfig) -> ModelResult:

    def build_model(training_ratings: RatingData) -> RatingTransformer:
        torch.manual_seed(config.seed)  # identical initial weights for the fit and the refit
        global_mean_rating = float(training_ratings.ratings.mean())
        initial_user_biases, initial_item_biases = fit_closed_form_biases(
            training_ratings.user_rows, training_ratings.item_rows, training_ratings.ratings,
            task.n_user_rows, task.n_item_rows, global_mean_rating, config.bias_regularization, config.bias_regularization)
        history_items, history_rating_tokens = user_histories(history_pairs(training_ratings, task, config.use_test_pairs_in_history),
                                                              task.n_user_rows, config.max_history_length, config.seed)
        return RatingTransformer(history_items, history_rating_tokens, task.n_user_rows, task.n_item_rows, config,
                                 global_mean_rating, initial_user_biases, initial_item_biases).to(DEVICE)

    # fit with early stopping -> validation predictions -> optional refit on 100 % -> test predictions
    # one bias lambda for both sides (as the factorization machine)
    return fit_refit_and_predict(task, build_model,
                                 max_epochs=config.max_epochs,
                                 early_stopping_patience=config.early_stopping_patience,
                                 refit_on_train_and_valid=config.refit_on_train_and_valid,
                                 learning_rate=config.learning_rate,
                                 user_bias_regularization=config.bias_regularization,
                                 item_bias_regularization=config.bias_regularization,
                                 factor_regularization=config.factor_regularization,
                                 ratings_per_batch=config.ratings_per_batch,
                                 seed=config.seed)


def store_rating_transformer(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "rating_transformer")


def load_rating_transformer(task_name: str) -> ModelResult:
    return load_model_result(task_name, "rating_transformer")
