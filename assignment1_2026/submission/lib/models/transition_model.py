import numpy as np
import scipy.sparse as sp

from lib.custom_types import ModelResult, PairChoiceTask, RatingData, TransitionConfig
from lib.model_storage import load_model_result, save_model_result
from lib.pair_choice import train_pair_scorer

# Which book follows which (PML Sec. 7.4, Eq. 7.7: p(user u interacts with item i | previous item j)):
#   C_ji      = sum over all users' sequences of step_decay^(d-1) whenever i comes d = 1 .. steps_ahead interactions after j
#   P(i | j)  = C_ji / (sum_i' C_ji' + shrinkage)                       (rare predecessors j are shrunk towards 0)
#   score(u, i) = sum over the user's last books j_1 (most recent), j_2, ... of context_decay^(m-1) * P(i | j_m)
# a plain (non-factorised) Markov chain: catches series and "read together" books; the factorised versions
# (FMC / FPMC, PML Sec. 7.5.1) share statistical strength between items, which the other sub-models already do
# expects the interactions in chronological order per user (lib/data_loading.read_chronological_interactions)


class TransitionModel:

    # no days -> lib/pair_choice.py only passes (user_rows, item_rows)
    uses_days = False

    def __init__(self, interactions: RatingData, n_user_rows: int, n_item_rows: int, config: TransitionConfig):
        self.n_item_rows = n_item_rows
        user_rows, item_rows = interactions.user_rows, interactions.item_rows

        # 1) transition counts j -> i for i up to steps_ahead interactions after j (same user only)
        from_items, to_items, weights = [], [], []
        for steps in range(1, config.steps_ahead + 1):
            is_same_user = user_rows[steps:] == user_rows[:-steps]
            from_items.append(item_rows[:-steps][is_same_user])
            to_items.append(item_rows[steps:][is_same_user])
            weights.append(np.full(is_same_user.sum(), config.step_decay ** (steps - 1)))
        transition_counts = sp.csr_matrix((np.concatenate(weights), (np.concatenate(from_items), np.concatenate(to_items))),
                                          shape=(n_item_rows, n_item_rows))
        transition_counts.sum_duplicates()

        # 2) shrunk transition probabilities, stored as sorted keys from * n_item_rows + to for fast lookups
        outgoing_counts = np.asarray(transition_counts.sum(axis=1)).ravel()
        transition_probabilities = (sp.diags(1.0 / (outgoing_counts + config.shrinkage)) @ transition_counts).tocoo()
        self.transition_keys = transition_probabilities.row.astype(np.int64) * n_item_rows + transition_probabilities.col
        key_order = np.argsort(self.transition_keys)
        self.transition_keys = self.transition_keys[key_order]
        self.transition_probabilities = transition_probabilities.data[key_order].astype(np.float32)

        # 3) the user's last context_length books, most recent first (0 = no book)
        self.recent_items = np.zeros((n_user_rows, config.context_length), dtype=np.int64)
        last_positions = np.flatnonzero(np.append(user_rows[1:] != user_rows[:-1], True))
        for steps_back in range(config.context_length):
            positions = last_positions - steps_back
            is_same_user = (positions >= 0) & (user_rows[np.maximum(positions, 0)] == user_rows[last_positions])
            self.recent_items[user_rows[last_positions[is_same_user]], steps_back] = item_rows[positions[is_same_user]]
        self.context_weights = config.context_decay ** np.arange(config.context_length)

    # P(to | from) for many pairs at once (0 where j -> i never happened)
    def transition_probability(self, from_item_rows: np.ndarray, to_item_rows: np.ndarray) -> np.ndarray:
        query_keys = from_item_rows.astype(np.int64) * self.n_item_rows + to_item_rows
        positions = np.minimum(np.searchsorted(self.transition_keys, query_keys), len(self.transition_keys) - 1)
        return np.where(self.transition_keys[positions] == query_keys, self.transition_probabilities[positions], 0.0)

    def score(self, user_rows: np.ndarray, item_rows: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(user_rows), dtype=np.float64)
        for steps_back, context_weight in enumerate(self.context_weights):
            scores += context_weight * self.transition_probability(self.recent_items[user_rows, steps_back], item_rows)
        return scores


def train_transition_model(task: PairChoiceTask, config: TransitionConfig) -> ModelResult:
    return train_pair_scorer(task, lambda interactions: TransitionModel(interactions, task.n_user_rows, task.n_item_rows, config),
                             "transition")


def store_transition_model(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "transition")


def load_transition_model(task_name: str) -> ModelResult:
    return load_model_result(task_name, "transition")
