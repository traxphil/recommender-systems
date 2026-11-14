import numpy as np
import pandas as pd

from lib.config import FIRST_PLAUSIBLE_TIMESTAMP, LAST_PLAUSIBLE_TIMESTAMP
from lib.custom_types import RatingTask

SECONDS_PER_HOUR = 3_600


# temporal models need the day of every test row -> only tasks whose test file has timestamps (Task 3), not Task 1
def require_test_days(task: RatingTask, model_name: str) -> None:
    if task.test_days is None:
        raise ValueError(f"{model_name} needs the day of every test row -> the test file has no timestamp column")


# the same for models that need the full timestamp (calendar fields: hour of day, weekday, month, year)
def require_test_timestamps(task: RatingTask, model_name: str) -> None:
    if task.test_timestamps is None:
        raise ValueError(f"{model_name} needs the timestamp of every test row -> the test file has no timestamp column")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Calendar fields of a timestamp (one-hot features of the time-aware factorization machine)
# all in UTC: the users' time zones are unknown
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# number of values of every calendar field; the year field covers first_year .. last_year (-> first_and_last_year)
CALENDAR_FIELD_SIZES = {"hour_of_day": 24, "weekday": 7, "month": 12}


# first and last calendar year among the timestamps (e.g. of the training ratings)
def first_and_last_year(timestamps: np.ndarray) -> tuple[int, int]:
    years = pd.to_datetime(timestamps, unit="s").year
    return int(years.min()), int(years.max())


# (N, fields): the row of every timestamp within each requested field
#   hour_of_day 0 .. 23 | weekday 0 = Monday .. 6 = Sunday | month 0 = January .. 11 = December
#   year 0 = first_year .. last_year - first_year (years outside the range are clipped into it)
def calendar_field_rows(timestamps: np.ndarray, field_names: tuple[str, ...], first_year: int, last_year: int) -> np.ndarray:
    moments = pd.to_datetime(timestamps, unit="s")
    row_of_field = {
        "hour_of_day": moments.hour,
        "weekday": moments.day_of_week,
        "month": moments.month - 1,
        "year": np.clip(moments.year, first_year, last_year) - first_year,
    }
    return np.column_stack([np.asarray(row_of_field[name], dtype=np.int64) for name in field_names])


# calendar_field_rows of every hour of the plausible timestamp range (the loaded timestamps are clipped into it)
# -> a model looks the fields of a timestamp up as table[(timestamp - FIRST_PLAUSIBLE_TIMESTAMP) // SECONDS_PER_HOUR],
#    which also works on the GPU (no calendar arithmetic there)
def calendar_field_rows_per_hour(field_names: tuple[str, ...], first_year: int, last_year: int) -> np.ndarray:
    hour_starts = np.arange(FIRST_PLAUSIBLE_TIMESTAMP, LAST_PLAUSIBLE_TIMESTAMP + 1, SECONDS_PER_HOUR)
    return calendar_field_rows(hour_starts, field_names, first_year, last_year)


# Counting a user's ratings around a day without loops over users:
# every reference rating gets the key user * KEY_STRIDE + day -> sorted once, every window query = two binary searches
# KEY_STRIDE > largest day + largest window -> the windows of two different users can never overlap
KEY_STRIDE = 1 << 20


class UserTimeline:

    # reference ratings = the ratings a model may look at (training split for validation rows, whole file for test rows)
    def __init__(self, user_rows: np.ndarray, days: np.ndarray, is_zero: np.ndarray | None = None):
        keys = user_rows.astype(np.int64) * KEY_STRIDE + days
        order = np.argsort(keys, kind="stable")
        self.sorted_keys = keys[order]

        # zeros_before_position[p] = number of zeros among the first p sorted ratings -> zeros in a window by subtraction
        sorted_is_zero = np.zeros(len(order)) if is_zero is None else np.asarray(is_zero, dtype=np.float64)[order]
        self.zeros_before_position = np.concatenate([[0.0], np.cumsum(sorted_is_zero)])

    # how many reference ratings (and zeros) the user has within [day - window_days, day + window_days]
    def window_counts(self, user_rows: np.ndarray, days: np.ndarray, window_days: int) -> tuple[np.ndarray, np.ndarray]:
        query_keys = user_rows.astype(np.int64) * KEY_STRIDE + days
        window_start = np.searchsorted(self.sorted_keys, query_keys - window_days, side="left")
        window_end = np.searchsorted(self.sorted_keys, query_keys + window_days, side="right")
        rating_counts = (window_end - window_start).astype(np.float64)
        zero_counts = self.zeros_before_position[window_end] - self.zeros_before_position[window_start]
        return rating_counts, zero_counts

    # how many reference ratings the user gave before this day ("experience" at the time of the rating)
    def counts_before(self, user_rows: np.ndarray, days: np.ndarray) -> np.ndarray:
        user_start_keys = user_rows.astype(np.int64) * KEY_STRIDE
        first_position_of_user = np.searchsorted(self.sorted_keys, user_start_keys, side="left")
        first_position_on_day = np.searchsorted(self.sorted_keys, user_start_keys + days, side="left")
        return (first_position_on_day - first_position_of_user).astype(np.float64)


# the same counting keyed by item rows: ItemTimeline(item_rows, days).window_counts(item_rows, days, w)
# = interactions with the item within +- w days (Tasks 2 / 4: popularity of a book around the asked day)
ItemTimeline = UserTimeline


# first day on which every user / item row appears in the reference ratings; rows without ratings -> NaN
def first_day_per_row(rows: np.ndarray, days: np.ndarray, n_rows: int) -> np.ndarray:
    first_days = np.full(n_rows, np.inf)
    np.minimum.at(first_days, rows, days.astype(np.float64))
    return np.where(np.isfinite(first_days), first_days, np.nan)
