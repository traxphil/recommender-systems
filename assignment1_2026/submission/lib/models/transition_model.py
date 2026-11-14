import numpy as np

from lib.custom_types import ModelResult, PairChoiceTask, RatingData, TransitionConfig
from lib.helper_functioins import recent_sequences, sparse_matrix_from_entries
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
        user_rows, item_rows = interactions.user_rows, interactions.item_rows

        # 1) transition counts j -> i for i up to steps_ahead interactions after j (same user only)
        from_items, to_items, weights = [], [], []
        for steps in range(1, config.steps_ahead + 1):
            is_same_user = user_rows[steps:] == user_rows[:-steps]
            from_items.append(item_rows[:-steps][is_same_user])
            to_items.append(item_rows[steps:][is_same_user])
            weights.append(np.full(is_same_user.sum(), config.step_decay ** (steps - 1)))
        transition_counts = sparse_matrix_from_entries(np.concatenate(from_items), np.concatenate(to_items), n_item_rows, n_item_rows,
                                                       values=np.concatenate(weights))

        # 2) shrunk transition probabilities: row j of the counts divided by (its outgoing count + shrinkage)
        #    .tocsr() -> canonical CSR again, which keeps the element lookups P[from, to] fast
        outgoing_counts = np.asarray(transition_counts.sum(axis=1)).ravel()
        self.transition_probabilities = transition_counts.multiply(1.0 / (outgoing_counts + config.shrinkage)[:, None]).tocsr()

        # 3) the user's last context_length books, most recent first (0 = no book)
        self.recent_items = recent_sequences(interactions, n_user_rows, config.context_length)[:, ::-1]
        self.context_weights = config.context_decay ** np.arange(config.context_length)

    # P(to | from) for many pairs at once (0 where j -> i never happened)
    def transition_probability(self, from_item_rows: np.ndarray, to_item_rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.transition_probabilities[from_item_rows, to_item_rows]).ravel()

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
