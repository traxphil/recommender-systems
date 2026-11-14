import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config import DEVICE
from lib.custom_types import ChoicePairs, ModelResult, PairChoiceTask, RatingData, SasRecConfig
from lib.helper_functioins import log, recent_sequences
from lib.model_storage import load_model_result, save_model_result
from lib.pair_choice import pair_choice_accuracy

# SASRec (Kang & McAuley 2018, "Self-Attentive Sequential Recommendation", ICDM; PML Sec. 7.7.2):
#   input:  the user's last L books i_1 .. i_L  ->  h_l = sqrt(d) * e_{i_l} + p_l        (item + position embedding)
#   blocks: causal self-attention (position l only sees l' <= l) + point-wise feed-forward, residuals, layer norms
#   output: h_l predicts the book at position l + 1;  score(next = i) = h_L . e_i            (shared item embeddings)
#   loss:   -log sigmoid(h_l . e_{next}) - log(1 - sigmoid(h_l . e_{negative})) for every position l
# negatives are drawn in proportion to popularity (like the test negatives), the original draws them uniformly
# early stopping on the validation pair accuracy, then a refit on all interactions for the best number of epochs


class SelfAttentionBlock(nn.Module):

    def __init__(self, hidden_dimension: int, n_heads: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dimension)
        self.attention = nn.MultiheadAttention(hidden_dimension, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward_norm = nn.LayerNorm(hidden_dimension)
        self.feed_forward = nn.Sequential(nn.Linear(hidden_dimension, hidden_dimension), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(hidden_dimension, hidden_dimension), nn.Dropout(dropout))

    # layout of the authors' implementation: queries normalised, keys / values not; padding positions reset to 0
    def forward(self, hidden_states, causal_mask, is_book):
        queries = self.attention_norm(hidden_states)
        attended, _ = self.attention(queries, hidden_states, hidden_states, attn_mask=causal_mask, need_weights=False)
        hidden_states = self.feed_forward_norm(queries + attended)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        return hidden_states * is_book


class SasRecNetwork(nn.Module):

    def __init__(self, n_item_rows: int, config: SasRecConfig):
        super().__init__()
        self.embedding_scale = math.sqrt(config.hidden_dimension)
        self.item_embeddings = nn.Embedding(n_item_rows, config.hidden_dimension, padding_idx=0)   # row 0 = padding
        self.position_embeddings = nn.Embedding(config.max_sequence_length, config.hidden_dimension)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([SelfAttentionBlock(config.hidden_dimension, config.n_heads, config.dropout)
                                     for _ in range(config.n_blocks)])
        self.output_norm = nn.LayerNorm(config.hidden_dimension)

        # small initial embeddings (Xavier, as in the authors' code): with PyTorch's default N(0, 1) and the sqrt(d) scaling
        # the first logits are huge and training starts far off
        nn.init.xavier_normal_(self.item_embeddings.weight)
        nn.init.xavier_normal_(self.position_embeddings.weight)
        with torch.no_grad():
            self.item_embeddings.weight[0] = 0.0

        # True = not allowed: position l may not look at later positions
        self.register_buffer("causal_mask", torch.triu(torch.ones(config.max_sequence_length, config.max_sequence_length, dtype=torch.bool), 1))

    # (B, L) book rows, left-padded with 0 -> (B, L, d) hidden state per position
    def forward(self, item_sequences):
        is_book = (item_sequences != 0).unsqueeze(-1).float()
        positions = torch.arange(item_sequences.shape[1], device=item_sequences.device)
        hidden_states = self.item_embeddings(item_sequences) * self.embedding_scale + self.position_embeddings(positions)
        hidden_states = self.input_dropout(hidden_states) * is_book
        for block in self.blocks:
            hidden_states = block(hidden_states, self.causal_mask, is_book)
        return self.output_norm(hidden_states)


class SasRecModel:

    # no days -> lib/pair_choice.py only passes (user_rows, item_rows)
    uses_days = False

    def __init__(self, interactions: RatingData, n_user_rows: int, n_item_rows: int, config: SasRecConfig):
        self.config = config

        # one window per user: the last L + 1 books -> inputs = first L, targets = last L (shifted by one)
        self.training_windows = recent_sequences(interactions, n_user_rows, config.max_sequence_length + 1)
        self.users_with_history = np.flatnonzero(self.training_windows[:, -1] != 0)
        # negatives drawn like the test negatives: the book of a random interaction
        self.interaction_item_rows = interactions.item_rows

        torch.manual_seed(config.seed)  # identical initial weights for the fit and the refit
        self.network = SasRecNetwork(n_item_rows, config).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate, betas=(0.9, 0.98))
        self.random_generator = np.random.default_rng(config.seed)

    def train_one_epoch(self) -> float:
        self.network.train()
        loss_sum, n_batches = 0.0, 0
        shuffled_users = self.random_generator.permutation(self.users_with_history)

        for batch_start in range(0, len(shuffled_users), self.config.users_per_batch):
            windows = self.training_windows[shuffled_users[batch_start:batch_start + self.config.users_per_batch]]
            input_sequences = torch.as_tensor(windows[:, :-1], device=DEVICE)
            next_items = torch.as_tensor(windows[:, 1:], device=DEVICE)
            negative_items = torch.as_tensor(self.random_generator.choice(self.interaction_item_rows, size=next_items.shape), device=DEVICE)

            hidden_states = self.network(input_sequences)
            positive_logits = (hidden_states * self.network.item_embeddings(next_items)).sum(dim=-1)
            negative_logits = (hidden_states * self.network.item_embeddings(negative_items)).sum(dim=-1)

            # binary cross-entropy at every position that has an input book and a next book
            is_trained_position = (input_sequences != 0) & (next_items != 0)
            losses = F.softplus(-positive_logits) + F.softplus(negative_logits)     # -log sigma(x) = softplus(-x)
            loss = losses[is_trained_position].mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum, n_batches = loss_sum + loss.item(), n_batches + 1

        return loss_sum / n_batches

    # (P, 2): h_L of the user's sequence times the embeddings of both books (unknown books score NaN)
    def score_pairs(self, pairs: ChoicePairs) -> np.ndarray:
        self.network.eval()
        device = next(self.network.parameters()).device
        pair_scores = np.empty((len(pairs.user_rows), 2), dtype=np.float32)

        with torch.no_grad():
            for batch_start in range(0, len(pairs.user_rows), 4096):
                batch = slice(batch_start, batch_start + 4096)
                input_sequences = torch.as_tensor(self.training_windows[pairs.user_rows[batch], 1:], device=device)
                last_hidden_states = self.network(input_sequences)[:, -1, :]
                for side, item_rows in enumerate((pairs.first_item_rows[batch], pairs.second_item_rows[batch])):
                    item_vectors = self.network.item_embeddings(torch.as_tensor(item_rows, device=device))
                    pair_scores[batch, side] = (last_hidden_states * item_vectors).sum(dim=-1).cpu().numpy()

        pair_scores[pairs.first_item_rows == 0, 0] = np.nan
        pair_scores[pairs.second_item_rows == 0, 1] = np.nan
        return pair_scores


# refit_on_all_interactions=False: test scores from the training-split model (used by the sweeps: the refit cannot change
# the validation score and doubles the run time)
def train_sasrec(task: PairChoiceTask, config: SasRecConfig, refit_on_all_interactions: bool = True) -> ModelResult:
    validation_pairs = task.validation_pairs

    # 1) fit on the training split, early stopping on the validation pair accuracy (higher = better)
    model = SasRecModel(task.training_data, task.n_user_rows, task.n_item_rows, config)
    epoch_history, best_accuracy, best_weights, epochs_without_improvement = [], -1.0, None, 0
    for epoch in range(1, config.max_epochs + 1):
        train_loss = model.train_one_epoch()
        valid_accuracy = pair_choice_accuracy(model.score_pairs(validation_pairs), validation_pairs.first_is_positive)
        epoch_history.append({"epoch": epoch, "train_loss": train_loss, "valid_accuracy": valid_accuracy})
        log(f"SASRec epoch {epoch:3d} | train loss {train_loss:.4f} | validation accuracy {valid_accuracy:.4f}")

        if valid_accuracy > best_accuracy:
            best_accuracy, best_weights, epochs_without_improvement = valid_accuracy, copy.deepcopy(model.network.state_dict()), 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == config.early_stopping_patience:
                break

    model.network.load_state_dict(best_weights)
    best_epoch = max(epoch_history, key=lambda epoch_entry: epoch_entry["valid_accuracy"])["epoch"]
    log(f"SASRec | best validation accuracy {best_accuracy:.4f} at epoch {best_epoch}")
    valid_scores = model.score_pairs(validation_pairs)

    # 2) refit from scratch on every interaction for the best number of epochs -> test scores
    if refit_on_all_interactions:
        del model
        model = SasRecModel(task.training_and_validation_data, task.n_user_rows, task.n_item_rows, config)
        for epoch in range(1, best_epoch + 1):
            train_loss = model.train_one_epoch()
            log(f"SASRec refit epoch {epoch:3d} | train loss {train_loss:.4f}")
    test_scores = model.score_pairs(task.test_pairs)

    model.optimizer = None  # not needed after training
    return ModelResult(model, epoch_history, valid_scores, test_scores)


def store_sasrec(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "sasrec")


def load_sasrec(task_name: str) -> ModelResult:
    return load_model_result(task_name, "sasrec")
