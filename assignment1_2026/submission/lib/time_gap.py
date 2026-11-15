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
#   - how soon after the user's last interaction the other readers of the next book read it (first / 10 % / 25 % / half of
#     the later readers, share within a week / month / year) and how long before it the last earlier reader read it
#   - book-to-book timing: readers of both the user's last book and the next book -> how many, how often the next book came
#     second, the median time between the two
#     (checked 15 Nov, out-of-fold MAE averaged over 3 fold splits: 83.62 days -> 81.7 with both groups)
# gaps enter as g = log(1 + seconds): the gaps span seconds to years

SECONDS_PER_DAY = 86_400
DAYS_PER_YEAR = 365.25
KEY_STRIDE = np.int64(1) << 37              # > every span of timestamps (year 25 ... 3645) -> one sortable key per (book, second)

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
        **next_book_reader_timing(reference.item_rows, reference_seconds, item_rows, last_seconds),
        **book_to_book_timing(reference, reference_seconds, user_rows, item_rows, n_user_rows),
    })


# when the other readers of the next book read it, relative to the user's last interaction t (log(1 + seconds))
#   one sorted key per interaction: book * KEY_STRIDE + (seconds - earliest second) -> position of t inside the book's
#   readers by one binary search for every row at once
def next_book_reader_timing(reference_item_rows: np.ndarray, reference_seconds: np.ndarray, item_rows: np.ndarray,
                            last_seconds: np.ndarray) -> dict:
    order = np.lexsort((reference_seconds, reference_item_rows))
    sorted_items = reference_item_rows[order].astype(np.int64)
    earliest_second = int(reference_seconds.min())
    sorted_offsets = reference_seconds[order].astype(np.int64) - earliest_second
    keys = sorted_items * KEY_STRIDE + sorted_offsets
    item_rows = item_rows.astype(np.int64)
    has_last = ~np.isnan(last_seconds)
    last_offsets = np.nan_to_num(last_seconds, nan=earliest_second) - earliest_second

    # first position in the book's readers after t + seconds_after
    def position_after(seconds_after: float) -> np.ndarray:
        offsets = np.clip(last_offsets + seconds_after, 0, KEY_STRIDE - 1).astype(np.int64)
        return np.searchsorted(keys, item_rows * KEY_STRIDE + offsets, side="right")

    segment_starts = np.searchsorted(keys, item_rows * KEY_STRIDE, side="left")
    segment_ends = np.searchsorted(keys, (item_rows + 1) * KEY_STRIDE, side="left")
    n_readers = segment_ends - segment_starts
    first_after = position_after(0.0)
    n_after = segment_ends - first_after
    has_after = has_last & (n_after > 0)
    has_before = has_last & (first_after > segment_starts)

    # quantile of the later readers' times (linear interpolation like np.quantile), as seconds after t
    def later_reader_quantile(share: float) -> np.ndarray:
        index = share * np.maximum(n_after - 1, 0)
        lower = np.minimum(first_after + np.floor(index).astype(np.int64), len(keys) - 1)
        upper = np.minimum(first_after + np.ceil(index).astype(np.int64), len(keys) - 1)
        offsets = sorted_offsets[lower] + (index - np.floor(index)) * (sorted_offsets[upper] - sorted_offsets[lower])
        return np.where(has_after, np.log1p(np.maximum(offsets - last_offsets, 0)), np.nan)

    last_before_offsets = sorted_offsets[np.maximum(first_after - 1, 0)]
    features = {
        "book_first_reader_after": later_reader_quantile(0.0),
        "book_readers_10_percent_after": later_reader_quantile(0.1),
        "book_readers_25_percent_after": later_reader_quantile(0.25),
        "book_readers_median_after": later_reader_quantile(0.5),
        "book_last_reader_before": np.where(has_before, np.log1p(np.maximum(last_offsets - last_before_offsets, 0)), np.nan),
    }
    for name, days in (("week", 7), ("month", 30), ("year", 365)):
        n_within = position_after(days * SECONDS_PER_DAY) - first_after
        features[f"book_share_read_within_{name}_after"] = np.where(has_last & (n_readers > 0), n_within / np.maximum(n_readers, 1), np.nan)
    return features


# readers of both the user's last book and the next book (the user excluded): how many, how often the next book came
# second, and the median log(1 + seconds) from the last book to the next one among those who read it second
# (first read per reader and book; one set intersection per row)
def book_to_book_timing(reference: RatingData, reference_seconds: np.ndarray, user_rows: np.ndarray, item_rows: np.ndarray,
                        n_user_rows: int) -> dict:
    is_last_of_user = np.append(reference.user_rows[1:] != reference.user_rows[:-1], True)   # reference is chronological per user
    last_item_of_user = np.zeros(n_user_rows, dtype=np.int64)
    last_item_of_user[reference.user_rows[is_last_of_user]] = reference.item_rows[is_last_of_user]

    first_reads = (pd.DataFrame({"item": reference.item_rows, "user": reference.user_rows, "seconds": reference_seconds})
                   .sort_values(["item", "user", "seconds"]).drop_duplicates(["item", "user"]))
    readers, read_seconds = first_reads["user"].to_numpy(), first_reads["seconds"].to_numpy(dtype=np.float64)
    item_of_read = first_reads["item"].to_numpy()
    n_items = int(max(item_of_read.max(), item_rows.max())) + 2
    segment_starts = np.searchsorted(item_of_read, np.arange(n_items), side="left")
    segment_ends = np.searchsorted(item_of_read, np.arange(n_items), side="right")

    n_co_readers = np.zeros(len(user_rows))
    share_next_second = np.full(len(user_rows), np.nan)
    median_log_gap = np.full(len(user_rows), np.nan)
    for row, (user, next_item) in enumerate(zip(user_rows, item_rows)):
        last_item = last_item_of_user[user]
        if segment_ends[last_item] == segment_starts[last_item] or segment_ends[next_item] == segment_starts[next_item]:
            continue
        last_slice = slice(segment_starts[last_item], segment_ends[last_item])
        next_slice = slice(segment_starts[next_item], segment_ends[next_item])
        common, last_positions, next_positions = np.intersect1d(readers[last_slice], readers[next_slice], assume_unique=True,
                                                                return_indices=True)
        is_other = common != user
        if not is_other.any():
            continue
        seconds_between = read_seconds[next_slice][next_positions[is_other]] - read_seconds[last_slice][last_positions[is_other]]
        n_co_readers[row] = is_other.sum()
        share_next_second[row] = np.mean(seconds_between > 0)
        if (seconds_between > 0).any():
            median_log_gap[row] = np.median(np.log1p(seconds_between[seconds_between > 0]))
    return {"log_co_readers_of_last_and_next_book": np.log1p(n_co_readers),
            "co_readers_share_next_book_second": share_next_second,
            "co_readers_median_log_gap": median_log_gap}


def mean_absolute_error_seconds(predicted_gaps: np.ndarray, true_gaps: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted_gaps - true_gaps)))
