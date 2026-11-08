import numpy as np
import pandas as pd

from lib.config import FIRST_PLAUSIBLE_TIMESTAMP, LAST_PLAUSIBLE_TIMESTAMP
from lib.custom_types import RatingData
from lib.temporal_features import ItemTimeline

# Features of Task 6 (time until the user's next interaction), computed from reference interactions in chronological order:
# the training split for the validation rows (each user's last interaction held out), all interactions for the test rows
#   - the user's rhythm = autoregression on the previous gaps (PML Sec. 7.1): last 3 gaps, median / mean / spread of all gaps
#   - when the user's last interaction happened (calendar time, hour, weekday)
#   - the next book (given in the test file): popularity, when it first / typically appears relative to the user's last
#     interaction, and how much of its readership comes after that day
#     (19 % of the Task 6 test books were first read by anyone only after the user's last interaction -> a lower bound for the gap)
# gaps enter as g = log(1 + seconds): the gaps span seconds to years

SECONDS_PER_DAY = 86_400
DAYS_PER_YEAR = 365.25

# the columns the linear autoregression uses (the in-book model); LightGBM gets every column
AUTOREGRESSIVE_FEATURES = ["last_log_gap", "second_last_log_gap", "third_last_log_gap", "median_log_gap", "mean_log_gap",
                           "log_interactions"]


def gap_features(reference: RatingData, reference_seconds: np.ndarray, user_rows: np.ndarray, item_rows: np.ndarray,
                 n_user_rows: int, n_item_rows: int) -> pd.DataFrame:

    # 1) gaps between consecutive interactions of the same user (reference is chronological per user)
    is_same_user_as_previous = np.append(False, reference.user_rows[1:] == reference.user_rows[:-1])
    gap_positions = np.flatnonzero(is_same_user_as_previous)
    log_gaps = np.log1p(np.maximum(reference_seconds[gap_positions] - reference_seconds[gap_positions - 1], 0))
    gaps_by_user = pd.DataFrame({"user_row": reference.user_rows[gap_positions], "log_gap": log_gaps})
    gap_statistics = gaps_by_user.groupby("user_row")["log_gap"].agg(["median", "mean", "std", "min", "max"]).reindex(np.arange(n_user_rows))
    share_under_one_day = (gaps_by_user["log_gap"] < np.log1p(SECONDS_PER_DAY)).groupby(gaps_by_user["user_row"]).mean().reindex(np.arange(n_user_rows))

    # 2) the user's last three gaps (NaN where the history is shorter)
    last_gaps = np.full((n_user_rows, 3), np.nan)
    last_gap_positions = np.flatnonzero(np.append(gaps_by_user["user_row"].to_numpy()[1:] != gaps_by_user["user_row"].to_numpy()[:-1], True))
    for steps_back in range(3):
        positions = last_gap_positions - steps_back
        is_same_user = (positions >= 0) & (gaps_by_user["user_row"].to_numpy()[np.maximum(positions, 0)] == gaps_by_user["user_row"].to_numpy()[last_gap_positions])
        last_gaps[gaps_by_user["user_row"].to_numpy()[last_gap_positions[is_same_user]], steps_back] = log_gaps[positions[is_same_user]]

    # 3) the user's number of interactions, first and last timestamp
    interactions_per_user = np.bincount(reference.user_rows, minlength=n_user_rows)
    last_seconds_of_user = np.full(n_user_rows, np.nan)
    first_seconds_of_user = np.full(n_user_rows, np.nan)
    is_last_of_user = np.append(reference.user_rows[1:] != reference.user_rows[:-1], True)
    is_first_of_user = np.append(True, reference.user_rows[1:] != reference.user_rows[:-1])
    last_seconds_of_user[reference.user_rows[is_last_of_user]] = reference_seconds[is_last_of_user]
    first_seconds_of_user[reference.user_rows[is_first_of_user]] = reference_seconds[is_first_of_user]
    last_day_of_user = np.zeros(n_user_rows, dtype=np.int64)
    last_day_of_user[reference.user_rows[is_last_of_user]] = reference.days[is_last_of_user]

    # 4) the next book relative to the user's last day: popularity, first / median day, share of readers after that day
    interactions_per_item = np.bincount(reference.item_rows, minlength=n_item_rows)
    item_days = pd.Series(reference.days).groupby(reference.item_rows)
    first_day_of_item = item_days.min().reindex(np.arange(n_item_rows)).to_numpy(dtype=np.float64)
    median_day_of_item = item_days.median().reindex(np.arange(n_item_rows)).to_numpy(dtype=np.float64)
    users_last_day = last_day_of_user[user_rows]
    item_interactions_until_day = ItemTimeline(reference.item_rows, reference.days).counts_before(item_rows, users_last_day + 1)
    item_popularity = interactions_per_item[item_rows]

    last_seconds = last_seconds_of_user[user_rows]
    # calendar features from the clipped timestamp (a few timestamps lie in the years 25 / 3645)
    last_calendar_seconds = np.clip(np.nan_to_num(last_seconds, nan=FIRST_PLAUSIBLE_TIMESTAMP), FIRST_PLAUSIBLE_TIMESTAMP, LAST_PLAUSIBLE_TIMESTAMP)
    last_datetime = pd.to_datetime(last_calendar_seconds, unit="s")
    return pd.DataFrame({
        "last_log_gap": last_gaps[user_rows, 0],
        "second_last_log_gap": last_gaps[user_rows, 1],
        "third_last_log_gap": last_gaps[user_rows, 2],
        "median_log_gap": gap_statistics["median"].to_numpy()[user_rows],
        "mean_log_gap": gap_statistics["mean"].to_numpy()[user_rows],
        "std_log_gap": gap_statistics["std"].to_numpy()[user_rows],
        "min_log_gap": gap_statistics["min"].to_numpy()[user_rows],
        "max_log_gap": gap_statistics["max"].to_numpy()[user_rows],
        "share_gaps_under_one_day": share_under_one_day.to_numpy()[user_rows],
        "log_interactions": np.log1p(interactions_per_user[user_rows]),
        "log_active_span": np.log1p(np.maximum(last_seconds - first_seconds_of_user[user_rows], 0)),
        "last_interaction_year": last_calendar_seconds / (SECONDS_PER_DAY * DAYS_PER_YEAR) + 1970,
        "last_interaction_hour": last_datetime.hour.to_numpy(),
        "last_interaction_weekday": last_datetime.weekday.to_numpy(),
        "log_book_popularity": np.log1p(item_popularity),
        "book_first_day_minus_last_day": first_day_of_item[item_rows] - users_last_day,
        "book_median_day_minus_last_day": median_day_of_item[item_rows] - users_last_day,
        "book_share_read_after_last_day": np.where(item_popularity > 0, 1.0 - item_interactions_until_day / np.maximum(item_popularity, 1), np.nan),
    })


def mean_absolute_error_seconds(predicted_gaps: np.ndarray, true_gaps: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted_gaps - true_gaps)))
