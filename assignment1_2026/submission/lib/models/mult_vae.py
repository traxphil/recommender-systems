import copy

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config import DEVICE
from lib.custom_types import ChoicePairs, ModelResult, MultVaeConfig, PairChoiceTask, RatingData
from lib.helper_functioins import log
from lib.model_storage import load_model_result, save_model_result
from lib.pair_choice import pair_choice_accuracy

# Mult-VAE (Liang et al. 2018, "Variational Autoencoders for Collaborative Filtering", WWW):
#   encoder: x_u (0/1 history) -> L2-normalise -> dropout -> tanh(hidden) -> mean, log-variance of z_u (latent_dimension)
#   decoder: z_u -> tanh(hidden) -> one logit per item;  softmax over the items = where the user's next read goes
#   loss: -sum_i x_ui log softmax(logits)_i  +  beta * KL(q(z_u | x_u) || N(0, I)),  beta raised linearly to kl_weight_max
# non-linear and trained with dropout on the input -> learns to fill in held-out books (unlike EASE: linear, closed form)
# score(u, i) = logit of i (z_u = its mean, no sampling); both books of a pair belong to the same user -> logits compare directly
# early stopping on the validation pair accuracy, then a refit on all interactions for the best number of epochs


class MultVaeNetwork(nn.Module):

    def __init__(self, n_item_columns: int, config: MultVaeConfig):
        super().__init__()
        self.input_dropout = config.input_dropout
        self.latent_dimension = config.latent_dimension
        self.encoder_hidden = nn.Linear(n_item_columns, config.hidden_dimension)
        self.encoder_output = nn.Linear(config.hidden_dimension, 2 * config.latent_dimension)  # mean and log-variance
        self.decoder_hidden = nn.Linear(config.latent_dimension, config.hidden_dimension)
        self.decoder_output = nn.Linear(config.hidden_dimension, n_item_columns)

    # (B, items) 0/1 histories -> (B, items) logits, (B, latent) mean, (B, latent) log-variance
    def forward(self, read_rows):
        encoder_input = F.dropout(F.normalize(read_rows, dim=1), p=self.input_dropout, training=self.training)
        encoded = self.encoder_output(torch.tanh(self.encoder_hidden(encoder_input)))
        latent_mean, latent_log_variance = encoded[:, :self.latent_dimension], encoded[:, self.latent_dimension:]

        # reparameterisation trick in training, the mean when scoring
        latent = latent_mean + torch.randn_like(latent_mean) * torch.exp(0.5 * latent_log_variance) if self.training else latent_mean
        logits = self.decoder_output(torch.tanh(self.decoder_hidden(latent)))
        return logits, latent_mean, latent_log_variance


class MultVaeModel:

    # no time terms -> lib/pair_choice.py only passes (user_rows, item_rows)
    uses_days = False

    def __init__(self, interactions: RatingData, n_user_rows: int, n_item_rows: int, config: MultVaeConfig):
        self.config = config

        # 1) the max_items most popular items -> input / output column, others -1 (score NaN)
        interactions_per_item = np.bincount(interactions.item_rows, minlength=n_item_rows)
        kept_items = np.argsort(-interactions_per_item, kind="stable")[:config.max_items]
        kept_items = kept_items[interactions_per_item[kept_items] > 0]
        self.column_of_item = np.full(n_item_rows, -1, dtype=np.int64)
        self.column_of_item[kept_items] = np.arange(len(kept_items))

        # 2) user x kept-item 0/1 matrix of who read what = the network's input (and the reconstruction target)
        is_kept = self.column_of_item[interactions.item_rows] >= 0
        self.read_matrix = sp.csr_matrix((np.ones(is_kept.sum(), dtype=np.float32),
                                          (interactions.user_rows[is_kept], self.column_of_item[interactions.item_rows[is_kept]])),
                                         shape=(n_user_rows, len(kept_items)))
        self.users_with_history = np.flatnonzero(np.diff(self.read_matrix.indptr) > 0)

        torch.manual_seed(config.seed)  # identical initial weights for the fit and the refit
        self.network = MultVaeNetwork(len(kept_items), config).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.shuffle_generator = np.random.default_rng(config.seed)
        self.epochs_trained = 0

    def read_rows(self, user_rows: np.ndarray) -> torch.Tensor:
        device = next(self.network.parameters()).device
        return torch.as_tensor(self.read_matrix[user_rows].toarray(), device=device)

    # one pass over all users with a history, in random order
    def train_one_epoch(self) -> float:
        self.network.train()
        self.epochs_trained += 1
        kl_weight = self.config.kl_weight_max * min(1.0, self.epochs_trained / self.config.kl_annealing_epochs)
        loss_sum, n_batches = 0.0, 0

        shuffled_users = self.shuffle_generator.permutation(self.users_with_history)
        for batch_start in range(0, len(shuffled_users), self.config.users_per_batch):
            read_rows = self.read_rows(shuffled_users[batch_start:batch_start + self.config.users_per_batch])
            logits, latent_mean, latent_log_variance = self.network(read_rows)

            # multinomial log-likelihood of the user's books + KL divergence of q(z | x) from N(0, I)
            reconstruction_loss = -(F.log_softmax(logits, dim=1) * read_rows).sum(dim=1).mean()
            kl_divergence = (-0.5 * (1 + latent_log_variance - latent_mean ** 2 - latent_log_variance.exp()).sum(dim=1)).mean()
            loss = reconstruction_loss + kl_weight * kl_divergence

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum, n_batches = loss_sum + loss.item(), n_batches + 1

        return loss_sum / n_batches

    # (P, 2) logits of the first and the second item of every pair; the users' logits are computed once per user batch
    def score_pairs(self, pairs: ChoicePairs) -> np.ndarray:
        self.network.eval()
        pair_scores = np.full((len(pairs.user_rows), 2), np.nan, dtype=np.float32)
        item_columns = np.column_stack([self.column_of_item[pairs.first_item_rows], self.column_of_item[pairs.second_item_rows]])

        # pairs sorted by user -> every user batch covers one contiguous block of pairs
        pair_order = np.argsort(pairs.user_rows, kind="stable")
        sorted_users = pairs.user_rows[pair_order]
        distinct_users, first_pair_of_user = np.unique(sorted_users, return_index=True)
        first_pair_of_user = np.append(first_pair_of_user, len(sorted_users))

        with torch.no_grad():
            for batch_start in range(0, len(distinct_users), self.config.users_per_batch):
                batch_users = distinct_users[batch_start:batch_start + self.config.users_per_batch]
                logits, _, _ = self.network(self.read_rows(batch_users))

                # gather the needed logits on the device (a full batch of logits is users x items)
                batch_pairs = pair_order[first_pair_of_user[batch_start]:first_pair_of_user[batch_start + len(batch_users)]]
                row_in_batch = np.searchsorted(batch_users, pairs.user_rows[batch_pairs])
                for side in (0, 1):
                    has_column = item_columns[batch_pairs, side] >= 0
                    gathered = logits[torch.as_tensor(row_in_batch[has_column], device=logits.device),
                                      torch.as_tensor(item_columns[batch_pairs[has_column], side], device=logits.device)]
                    pair_scores[batch_pairs[has_column], side] = gathered.cpu().numpy()

        return pair_scores


def train_mult_vae(task: PairChoiceTask, config: MultVaeConfig) -> ModelResult:
    validation_pairs = task.validation_pairs

    # 1) fit on the training split, early stopping on the validation pair accuracy (higher = better)
    model = MultVaeModel(task.training_data, task.n_user_rows, task.n_item_rows, config)
    epoch_history, best_accuracy, best_weights, epochs_without_improvement = [], -1.0, None, 0
    for epoch in range(1, config.max_epochs + 1):
        train_loss = model.train_one_epoch()
        valid_accuracy = pair_choice_accuracy(model.score_pairs(validation_pairs), validation_pairs.first_is_positive)
        epoch_history.append({"epoch": epoch, "train_loss": train_loss, "valid_accuracy": valid_accuracy})
        log(f"Mult-VAE epoch {epoch:3d} | train loss {train_loss:.2f} | validation accuracy {valid_accuracy:.4f}")

        if valid_accuracy > best_accuracy:
            best_accuracy, best_weights, epochs_without_improvement = valid_accuracy, copy.deepcopy(model.network.state_dict()), 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == config.early_stopping_patience:
                break

    model.network.load_state_dict(best_weights)
    best_epoch = max(epoch_history, key=lambda epoch_entry: epoch_entry["valid_accuracy"])["epoch"]
    log(f"Mult-VAE | best validation accuracy {best_accuracy:.4f} at epoch {best_epoch}")
    valid_scores = model.score_pairs(validation_pairs)

    # 2) refit from scratch on every interaction for the best number of epochs -> test scores
    del model
    model = MultVaeModel(task.training_and_validation_data, task.n_user_rows, task.n_item_rows, config)
    for epoch in range(1, best_epoch + 1):
        train_loss = model.train_one_epoch()
        log(f"Mult-VAE refit epoch {epoch:3d} | train loss {train_loss:.2f}")
    test_scores = model.score_pairs(task.test_pairs)

    # the optimizer state is not needed after training (as large as the network twice)
    model.optimizer = None
    return ModelResult(model, epoch_history, valid_scores, test_scores)


def store_mult_vae(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "mult_vae")


def load_mult_vae(task_name: str) -> ModelResult:
    return load_model_result(task_name, "mult_vae")
