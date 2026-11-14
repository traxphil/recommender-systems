import numpy as np

from lib.custom_types import ChoicePairs, ModelResult, PairChoiceTask
from lib.helper_functioins import log

# Shared pieces of the interaction models (Tasks 2 / 4):
# every sub-model scores (user, item) -> higher = more likely that the user read the item;
# a pair is decided by comparing the scores of its two items


# (P, 2): score of the first and of the second item of every pair
# time-weighted models (uses_days = True) also get the day of every pair
def score_both_items(model, pairs: ChoicePairs) -> np.ndarray:
    day_arguments = (pairs.days,) if model.uses_days else ()
    return np.column_stack([model.score(pairs.user_rows, pairs.first_item_rows, *day_arguments),
                            model.score(pairs.user_rows, pairs.second_item_rows, *day_arguments)]).astype(np.float32)


# share of pairs where the higher score belongs to the real item; equal or missing (NaN) scores count as a coin flip
def pair_choice_accuracy(pair_scores: np.ndarray, first_is_positive: np.ndarray) -> float:
    first_scores, second_scores = pair_scores[:, 0], pair_scores[:, 1]
    is_decided = (first_scores != second_scores) & ~np.isnan(first_scores) & ~np.isnan(second_scores)
    is_correct = (first_scores > second_scores) == first_is_positive
    return float(np.mean(np.where(is_decided, is_correct, 0.5)))


# the pipeline of every sub-model: built on the training split -> scores of the validation pairs (blender input)
# -> rebuilt on every interaction of the training file -> scores of the test pairs
# build_model(interactions) creates the model from the interactions it may look at
def train_pair_scorer(task: PairChoiceTask, build_model, model_name: str) -> ModelResult:
    model = build_model(task.training_data)
    valid_scores = score_both_items(model, task.validation_pairs)
    log(f"{model_name} | validation accuracy {pair_choice_accuracy(valid_scores, task.validation_pairs.first_is_positive):.4f}")

    del model  # free its memory (EASE: a dense items x items matrix) before the second build
    model = build_model(task.training_and_validation_data)
    test_scores = score_both_items(model, task.test_pairs)
    return ModelResult(model, [], valid_scores, test_scores)
