from typing import NamedTuple

import numpy as np
import pandas as pd

from lib.config import FIRST_PLAUSIBLE_TIMESTAMP, LAST_PLAUSIBLE_TIMESTAMP
from lib.custom_types import ChoicePairs, PairChoiceTask, RatingData, RatingTask, TimeGapTask
from lib.helper_functioins import ids_to_rows, number_by_first_appearance

SECONDS_PER_DAY = 86_400

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# read & write
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# read the CSV file into columns as trings
def read_csv_columns(csv_path) -> pd.DataFrame:

    # every column as str, empty fields stay "" instead of NaN / blank lines are skipped
    return pd.read_csv(csv_path, dtype=str, keep_default_na=False)

# write prediction results into the file
def write_prediction_file(test_csv_path, output_csv_path, predictions_in_test_order):

    test_columns = read_csv_columns(test_csv_path)

    assert len(test_columns) == len(predictions_in_test_order), \
        f"{len(test_columns)} test rows vs {len(predictions_in_test_order)} predictions"

    # the prediction always goes into the last column of the test file / header and all other fields stay as they are
    test_columns[test_columns.columns[-1]] = predictions_in_test_order
    test_columns.to_csv(output_csv_path, index=False)

# Tasks 2 / 4: the "which" column gets the id of the chosen item (item1 where first_item_wins, else item2)
def write_choice_prediction_file(test_csv_path, output_csv_path, first_item_wins):
    test_columns = read_csv_columns(test_csv_path)
    chosen_item_ids = np.where(first_item_wins, test_columns["item1"], test_columns["item2"])
    write_prediction_file(test_csv_path, output_csv_path, chosen_item_ids)

# unix timestamps (str column or integer array) -> int64 seconds, impossible timestamps clipped to the plausible range
def plausible_timestamps(timestamp_column) -> np.ndarray:
    seconds = np.asarray(pd.to_numeric(timestamp_column), dtype=np.int64)
    return np.clip(seconds, FIRST_PLAUSIBLE_TIMESTAMP, LAST_PLAUSIBLE_TIMESTAMP)

# unix timestamps (str column or integer array) -> whole days since 2000-01-01, impossible timestamps clipped to the plausible range
def timestamps_to_days(timestamp_column) -> np.ndarray:
    return (plausible_timestamps(timestamp_column) - FIRST_PLAUSIBLE_TIMESTAMP) // SECONDS_PER_DAY


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Data loading Task 1 / Task 3
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# create the prediction task problem for task 1 (test: user, item) and task 3 (test: user, item, timestamp)
def load_rating_prediction_task(training_csv_path, test_csv_path, validation_fraction: float = 0.1, split_seed: int = 0) -> RatingTask:

    # read both files as string columns (training: user, item, timestamp, rating | test: user, item (, timestamp))
    training_columns = read_csv_columns(training_csv_path)
    test_columns = read_csv_columns(test_csv_path)

    # numbering the users and items by their first appearance for the embedding index / id (unknowns to 0!)
    row_of_user = number_by_first_appearance(training_columns["user"])
    row_of_item = number_by_first_appearance(training_columns["item"])

    # populating the embeddings
    user_rows = ids_to_rows(training_columns["user"], row_of_user)
    item_rows = ids_to_rows(training_columns["item"], row_of_item)
    ratings = np.array(training_columns["rating"], dtype=np.float32)
    test_user_rows = ids_to_rows(test_columns["user"], row_of_user)
    test_item_rows = ids_to_rows(test_columns["item"], row_of_item)
    n_user_rows = len(row_of_user) + 1  # U users + row 0
    n_item_rows = len(row_of_item) + 1  # I items + row 0

    # time of every rating (day + timestamp) only if the test file has it too (Task 3): a model can only use time it also
    # gets for the test rows -> Task 1 (test file without timestamps): no time data at all, no Task 1 model can use it
    has_time = "timestamp" in test_columns.columns
    timestamps = plausible_timestamps(training_columns["timestamp"]) if has_time else None
    test_timestamps = plausible_timestamps(test_columns["timestamp"]) if has_time else None
    all_ratings = RatingData(user_rows, item_rows, ratings,
                             days=timestamps_to_days(timestamps) if has_time else None, timestamps=timestamps)

    # training and validation split / randomized with a seed
    shuffled_indices = np.random.default_rng(split_seed).permutation(len(ratings))
    n_valid = int(len(ratings) * validation_fraction)
    valid_indices, train_indices = shuffled_indices[:n_valid], shuffled_indices[n_valid:]

    # create rating task instance
    return RatingTask(
        training_data=all_ratings.at_positions(train_indices),
        validation_data=all_ratings.at_positions(valid_indices),
        training_and_validation_data=all_ratings,
        test_user_rows=test_user_rows,
        test_item_rows=test_item_rows,
        n_user_rows=n_user_rows,
        n_item_rows=n_item_rows,
        test_days=timestamps_to_days(test_timestamps) if has_time else None,
        test_timestamps=test_timestamps,
    )



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Data loading Task 2 / Task 4 (which of two items did the user read?)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# validation pairs built like the test pairs: a held-out interaction (positive) vs. the item of a random other interaction
# (negative, so drawn in proportion to popularity, regardless of time; never the positive item itself), in random order
# -> checked on the test files: popularity of both pair items and "item not seen yet at t" (Task 4) match this sampling
def load_pair_choice_task(training_csv_path, test_csv_path, validation_fraction: float = 0.1, split_seed: int = 0) -> PairChoiceTask:

    # read both files as string columns (training: user, item, timestamp, rating | test: user, (timestamp,) item1, item2, which)
    training_columns = read_csv_columns(training_csv_path)
    test_columns = read_csv_columns(test_csv_path)

    # numbering the users and items by their first appearance for the embedding index / id (unknowns to 0!)
    row_of_user = number_by_first_appearance(training_columns["user"])
    row_of_item = number_by_first_appearance(training_columns["item"])

    user_rows = ids_to_rows(training_columns["user"], row_of_user)
    item_rows = ids_to_rows(training_columns["item"], row_of_item)
    ratings = np.array(training_columns["rating"], dtype=np.float32)
    days = timestamps_to_days(training_columns["timestamp"])
    all_interactions = RatingData(user_rows, item_rows, ratings, days)

    # validation positives = a random cut of the interactions / the models of the validation stage never see them
    random_generator = np.random.default_rng(split_seed)
    shuffled_indices = random_generator.permutation(len(user_rows))
    n_valid = int(len(user_rows) * validation_fraction)
    valid_indices, train_indices = shuffled_indices[:n_valid], shuffled_indices[n_valid:]

    # negatives = items of random training-split interactions, drawn again where they hit the positive item
    positive_item_rows = item_rows[valid_indices]
    negative_item_rows = item_rows[random_generator.choice(train_indices, n_valid)]
    while np.any(is_same_item := negative_item_rows == positive_item_rows):
        negative_item_rows[is_same_item] = item_rows[random_generator.choice(train_indices, is_same_item.sum())]

    # random order inside every pair -> first_is_positive is 50 / 50
    first_is_positive = random_generator.random(n_valid) < 0.5
    validation_pairs = ChoicePairs(
        user_rows=user_rows[valid_indices],
        first_item_rows=np.where(first_is_positive, positive_item_rows, negative_item_rows),
        second_item_rows=np.where(first_is_positive, negative_item_rows, positive_item_rows),
        days=days[valid_indices],
        first_is_positive=first_is_positive,
    )

    test_pairs = ChoicePairs(
        user_rows=ids_to_rows(test_columns["user"], row_of_user),
        first_item_rows=ids_to_rows(test_columns["item1"], row_of_item),
        second_item_rows=ids_to_rows(test_columns["item2"], row_of_item),
        days=timestamps_to_days(test_columns["timestamp"]) if "timestamp" in test_columns.columns else None,
        first_is_positive=None,
    )

    return PairChoiceTask(
        training_data=RatingData(user_rows[train_indices], item_rows[train_indices], ratings[train_indices], days[train_indices]),
        training_and_validation_data=all_interactions,
        validation_pairs=validation_pairs,
        test_pairs=test_pairs,
        n_user_rows=len(row_of_user) + 1,
        n_item_rows=len(row_of_item) + 1,
    )


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Tasks 5 / 6: interactions in the order they happened
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class ChronologicalInteractions(NamedTuple):
    interactions: RatingData                # sorted by user, then time (ties keep the file order)
    seconds: np.ndarray                     # (N,) raw unix timestamps in the same order
    is_last_of_user: np.ndarray             # (N,) bool: the user's most recent interaction
    row_of_user: pd.Index
    row_of_item: pd.Index


def read_chronological_interactions(training_csv_path) -> ChronologicalInteractions:
    training_columns = read_csv_columns(training_csv_path)
    row_of_user = number_by_first_appearance(training_columns["user"])
    row_of_item = number_by_first_appearance(training_columns["item"])

    user_rows = ids_to_rows(training_columns["user"], row_of_user)
    item_rows = ids_to_rows(training_columns["item"], row_of_item)
    ratings = np.array(training_columns["rating"], dtype=np.float32)
    seconds = pd.to_numeric(training_columns["timestamp"]).to_numpy(dtype=np.int64)

    # np.lexsort sorts by the last key first and is stable -> by user, then time, ties in file order
    chronological_order = np.lexsort((seconds, user_rows))
    user_rows, item_rows, ratings, seconds = (values[chronological_order] for values in (user_rows, item_rows, ratings, seconds))
    is_last_of_user = np.append(user_rows[1:] != user_rows[:-1], True)

    return ChronologicalInteractions(RatingData(user_rows, item_rows, ratings, timestamps_to_days(seconds)),
                                     seconds, is_last_of_user, row_of_user, row_of_item)


# Task 5: the test asks for the book right after each user's last training interaction
# validation pairs the same way ("leave last out"): each user's last training interaction (positive) vs. the book of a random
# other interaction (negative, popularity-weighted as in Tasks 2 / 4), in random order
# the test file has no timestamp -> the day of a pair = the day of the user's last known interaction ("now")
def load_next_item_task(training_csv_path, test_csv_path, split_seed: int = 0) -> PairChoiceTask:

    chronological = read_chronological_interactions(training_csv_path)
    test_columns = read_csv_columns(test_csv_path)
    all_interactions = chronological.interactions
    user_rows, item_rows, days = all_interactions.user_rows, all_interactions.item_rows, all_interactions.days

    # validation positives = each user's last interaction (users with a single interaction keep it for training)
    last_positions = np.flatnonzero(chronological.is_last_of_user)
    has_earlier_interaction = (last_positions > 0) & (user_rows[np.maximum(last_positions - 1, 0)] == user_rows[last_positions])
    valid_indices = last_positions[has_earlier_interaction]
    is_training = np.ones(len(user_rows), dtype=bool)
    is_training[valid_indices] = False
    train_indices = np.flatnonzero(is_training)

    # negatives = items of random training-split interactions, drawn again where they hit the positive item
    random_generator = np.random.default_rng(split_seed)
    positive_item_rows = item_rows[valid_indices]
    negative_item_rows = item_rows[random_generator.choice(train_indices, len(valid_indices))]
    while np.any(is_same_item := negative_item_rows == positive_item_rows):
        negative_item_rows[is_same_item] = item_rows[random_generator.choice(train_indices, is_same_item.sum())]

    first_is_positive = random_generator.random(len(valid_indices)) < 0.5
    validation_pairs = ChoicePairs(
        user_rows=user_rows[valid_indices],
        first_item_rows=np.where(first_is_positive, positive_item_rows, negative_item_rows),
        second_item_rows=np.where(first_is_positive, negative_item_rows, positive_item_rows),
        days=days[valid_indices - 1],               # "now" = the user's previous (last remaining) interaction
        first_is_positive=first_is_positive,
    )

    # test pairs: "now" = the day of the user's last training interaction
    last_day_of_user = np.zeros(len(chronological.row_of_user) + 1, dtype=np.int64)
    last_day_of_user[user_rows[last_positions]] = days[last_positions]
    test_user_rows = ids_to_rows(test_columns["user"], chronological.row_of_user)
    test_pairs = ChoicePairs(
        user_rows=test_user_rows,
        first_item_rows=ids_to_rows(test_columns["item1"], chronological.row_of_item),
        second_item_rows=ids_to_rows(test_columns["item2"], chronological.row_of_item),
        days=last_day_of_user[test_user_rows],
        first_is_positive=None,
    )

    return PairChoiceTask(
        training_data=RatingData(user_rows[train_indices], item_rows[train_indices], all_interactions.ratings[train_indices], days[train_indices]),
        training_and_validation_data=all_interactions,
        validation_pairs=validation_pairs,
        test_pairs=test_pairs,
        n_user_rows=len(chronological.row_of_user) + 1,
        n_item_rows=len(chronological.row_of_item) + 1,
    )


# Task 6: the test asks for the time between each user's last training interaction and the next one (book given)
# validation rows the same way: each user's last training interaction is held out, the target is its gap to the previous one
def load_time_gap_task(training_csv_path, test_csv_path) -> TimeGapTask:

    chronological = read_chronological_interactions(training_csv_path)
    test_columns = read_csv_columns(test_csv_path)
    all_interactions, seconds = chronological.interactions, chronological.seconds
    user_rows, item_rows = all_interactions.user_rows, all_interactions.item_rows

    last_positions = np.flatnonzero(chronological.is_last_of_user)
    has_earlier_interaction = (last_positions > 0) & (user_rows[np.maximum(last_positions - 1, 0)] == user_rows[last_positions])
    valid_indices = last_positions[has_earlier_interaction]
    is_training = np.ones(len(user_rows), dtype=bool)
    is_training[valid_indices] = False

    return TimeGapTask(
        training_data=RatingData(user_rows[is_training], item_rows[is_training], all_interactions.ratings[is_training], all_interactions.days[is_training]),
        training_seconds=seconds[is_training],
        training_and_validation_data=all_interactions,
        training_and_validation_seconds=seconds,
        validation_user_rows=user_rows[valid_indices],
        validation_item_rows=item_rows[valid_indices],
        validation_gaps=(seconds[valid_indices] - seconds[valid_indices - 1]).astype(np.float64),
        test_user_rows=ids_to_rows(test_columns["user"], chronological.row_of_user),
        test_item_rows=ids_to_rows(test_columns["item"], chronological.row_of_item),
        n_user_rows=len(chronological.row_of_user) + 1,
        n_item_rows=len(chronological.row_of_item) + 1,
    )
