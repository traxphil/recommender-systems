import numpy as np
from sklearn.model_selection import KFold

from lib.custom_types import ChoicePairs, ModelResult, PairBlenderConfig, PairChoiceTask, RatingData
from lib.helper_functioins import log, sparse_matrix_from_entries
from lib.model_storage import load_model_result, save_model_result
from lib.pair_choice import pair_choice_accuracy
from lib.temporal_features import ItemTimeline, first_day_per_row

# Pair blender (Tasks 2 / 4 / 5): LightGBM learns P(first item is the real one) from both items' sub-model scores + meta-features
# - trained on the validation pairs, where every sub-model scores pairs it never saw (as the rating blender of Tasks 1 / 3)
# - every pair is used in both orders (first <-> second, label flipped) and predicted as the average of both orders
#   -> the answer does not depend on which item the file calls item1
# - scored out-of-fold: each validation pair is decided by a blender that never saw it

DAYS_PER_YEAR = 365.25
KEY_STRIDE = np.int64(1) << 32              # > every span of days -> one sortable key per (item, day)
READER_WINDOWS_DAYS = (7, 30, 90, 365)
SEQUENCE_WINDOW = 5                         # "within 5 books after"
SPAN_MARGIN_DAYS = 180                      # Task 2: the user's active span widened by half a year
WINDOW_PAIRS_PER_CHUNK = 20_000             # Task 2 reading window: pairs per sparse product (memory)


# per-group quantiles of values (lower quantile, NaN for groups without values): (n_groups,) array per quantile
def group_quantiles(group_rows: np.ndarray, values: np.ndarray, n_groups: int, quantiles: tuple[float, ...]) -> list[np.ndarray]:
    order = np.lexsort((values, group_rows))
    sorted_values = values[order].astype(np.float64)
    counts = np.bincount(group_rows, minlength=n_groups)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    result = []
    for quantile in quantiles:
        positions = starts + np.floor(quantile * np.maximum(counts - 1, 0)).astype(np.int64)
        result.append(np.where(counts > 0, sorted_values[np.minimum(positions, len(sorted_values) - 1)], np.nan))
    return result


class PairFeatureBuilder:

    # reference = the interactions the sub-models had for these pairs (training split for validation pairs, all for test pairs)
    def __init__(self, reference: RatingData, n_user_rows: int, n_item_rows: int, config: PairBlenderConfig):
        self.config = config
        # user x item 0/1 matrix of who read what -> "did the user already read it?" by element lookup
        self.read_matrix = sparse_matrix_from_entries(reference.user_rows, reference.item_rows, n_user_rows, n_item_rows)
        self.interactions_per_user = np.bincount(reference.user_rows, minlength=n_user_rows)

        if config.use_time_features:
            self.item_timeline = ItemTimeline(reference.item_rows, reference.days)
            self.first_day_of_item = first_day_per_row(reference.item_rows, reference.days, n_item_rows)
        if getattr(config, "use_reader_timing_features", False):    # getattr: configs stored before the field existed
            # one sorted key per interaction: item * KEY_STRIDE + (day - earliest day) -> readers of an item before / after a
            # day by binary search for every pair at once
            order = np.lexsort((reference.days, reference.item_rows))
            self.earliest_day = int(reference.days.min())
            self.sorted_days = reference.days[order].astype(np.int64)
            self.reader_keys = reference.item_rows[order].astype(np.int64) * KEY_STRIDE + (self.sorted_days - self.earliest_day)
            self.readers_per_item = np.bincount(reference.item_rows, minlength=n_item_rows)
        if getattr(config, "use_sequence_features", False):
            self.build_sequence_counts(reference, n_user_rows, n_item_rows)
        if getattr(config, "use_user_span_features", False):
            self.build_user_spans(reference, n_user_rows, n_item_rows)
        if getattr(config, "use_reading_window_features", False):
            self.build_reading_window(reference, n_item_rows)

    def item_feature_names(self) -> list[str]:
        time_feature_names = ["log_interactions_around_t", "item_age_years_at_t"] if self.config.use_time_features else []
        if getattr(self.config, "use_reader_timing_features", False):
            time_feature_names += [f"log_readers_{days}_days_after_t" for days in READER_WINDOWS_DAYS] + [
                "log_readers_30_days_before_t", "share_of_readers_after_t", "log_days_to_first_reader_after_t",
                "log_days_since_last_reader_before_t"]
        if getattr(self.config, "use_sequence_features", False):
            time_feature_names += ["log_next_after_last_book", f"log_within_{SEQUENCE_WINDOW}_after_last_book",
                                   "share_of_last_book_followers", f"log_within_{SEQUENCE_WINDOW}_after_2nd_3rd_last",
                                   f"log_within_{SEQUENCE_WINDOW}_before_last_book"]
        if getattr(self.config, "use_user_span_features", False):
            time_feature_names += ["item_first_minus_user_last_years", "item_median_minus_user_median_years",
                                   "item_10_percent_minus_user_last_years", "item_first_minus_user_first_years",
                                   "share_of_item_reads_in_user_span", f"share_of_item_reads_in_user_span_{SPAN_MARGIN_DAYS}_days"]
        if getattr(self.config, "use_reading_window_features", False):
            time_feature_names += [f"log_read_within_{SEQUENCE_WINDOW}_of_user_books",
                                   f"log_read_within_{SEQUENCE_WINDOW}_of_user_books_per_sqrt_readers"]
        return list(self.config.combined_models) + ["already_read"] + time_feature_names

    # Task 2: the user's first / median / last day and the item's first / 10 % / median day of the reference interactions +
    # every item's reads as sorted keys item * KEY_STRIDE + (day - earliest day) -> reads inside a span by binary search
    def build_user_spans(self, reference: RatingData, n_user_rows: int, n_item_rows: int):
        days = reference.days.astype(np.float64)
        self.user_first_day, self.user_median_day, self.user_last_day = group_quantiles(reference.user_rows, days, n_user_rows, (0.0, 0.5, 1.0))
        self.item_first_day, self.item_10_percent_day, self.item_median_day = group_quantiles(reference.item_rows, days, n_item_rows, (0.0, 0.1, 0.5))
        self.span_earliest_day = int(reference.days.min())
        self.span_keys = np.sort(reference.item_rows.astype(np.int64) * KEY_STRIDE + (reference.days.astype(np.int64) - self.span_earliest_day))
        self.reads_per_item = np.bincount(reference.item_rows, minlength=n_item_rows)

    def share_of_reads_between(self, item_rows: np.ndarray, first_days: np.ndarray, last_days: np.ndarray) -> np.ndarray:
        to_offset = lambda day: np.clip(np.nan_to_num(day - self.span_earliest_day, nan=-1), -1, KEY_STRIDE - 1).astype(np.int64)
        item_keys = item_rows.astype(np.int64) * KEY_STRIDE
        reads = (np.searchsorted(self.span_keys, item_keys + to_offset(last_days), side="right")
                 - np.searchsorted(self.span_keys, item_keys + to_offset(first_days), side="left"))
        n_reads = self.reads_per_item[item_rows]
        return np.where((n_reads > 0) & ~np.isnan(first_days), reads / np.maximum(n_reads, 1), np.nan)

    def user_span_columns(self, user_rows: np.ndarray, item_rows: np.ndarray) -> list[np.ndarray]:
        user_first, user_median, user_last = self.user_first_day[user_rows], self.user_median_day[user_rows], self.user_last_day[user_rows]
        return [(self.item_first_day[item_rows] - user_last) / DAYS_PER_YEAR,
                (self.item_median_day[item_rows] - user_median) / DAYS_PER_YEAR,
                (self.item_10_percent_day[item_rows] - user_last) / DAYS_PER_YEAR,
                (self.item_first_day[item_rows] - user_first) / DAYS_PER_YEAR,
                self.share_of_reads_between(item_rows, user_first, user_last),
                self.share_of_reads_between(item_rows, user_first - SPAN_MARGIN_DAYS, user_last + SPAN_MARGIN_DAYS)]

    # Task 2: item x item counts of "read within 5 books of each other" (every user's reference interactions in time order)
    def build_reading_window(self, reference: RatingData, n_item_rows: int):
        order = np.lexsort((reference.days, reference.user_rows))
        users, items = reference.user_rows[order], reference.item_rows[order]
        rows, columns = [], []
        for distance in range(1, SEQUENCE_WINDOW + 1):
            same_user = users[:-distance] == users[distance:]
            earlier, later = items[:-distance][same_user], items[distance:][same_user]
            rows += [earlier, later]
            columns += [later, earlier]
        self.window_counts = sparse_matrix_from_entries(np.concatenate(rows), np.concatenate(columns), n_item_rows, n_item_rows)
        self.has_read = self.read_matrix.copy()
        self.has_read.data[:] = 1.0
        self.readers_of_item = np.bincount(reference.item_rows, minlength=n_item_rows)

    # sum over the user's books j of window_counts[j, item] (symmetric), in chunks of pairs
    def reading_window_columns(self, user_rows: np.ndarray, item_rows: np.ndarray) -> list[np.ndarray]:
        totals = np.zeros(len(user_rows))
        for start in range(0, len(user_rows), WINDOW_PAIRS_PER_CHUNK):
            chunk = slice(start, start + WINDOW_PAIRS_PER_CHUNK)
            totals[chunk] = np.asarray(self.has_read[user_rows[chunk]].multiply(self.window_counts[item_rows[chunk]]).sum(axis=1)).ravel()
        return [np.log1p(totals), np.log1p(totals / np.sqrt(np.maximum(self.readers_of_item[item_rows], 1)))]

    # reference rows are in reading order within each user (lib/data_loading.py) -> item x item counts of "read d books later"
    def build_sequence_counts(self, reference: RatingData, n_user_rows: int, n_item_rows: int):
        users, items = reference.user_rows, reference.item_rows
        next_rows, next_columns, within_rows, within_columns = [], [], [], []
        for distance in range(1, SEQUENCE_WINDOW + 1):
            same_user = users[:-distance] == users[distance:]
            earlier, later = items[:-distance][same_user], items[distance:][same_user]
            within_rows.append(earlier)
            within_columns.append(later)
            if distance == 1:
                next_rows, next_columns = earlier, later
        within_rows, within_columns = np.concatenate(within_rows), np.concatenate(within_columns)
        self.next_counts = sparse_matrix_from_entries(next_rows, next_columns, n_item_rows, n_item_rows).tocsr()
        self.within_counts = sparse_matrix_from_entries(within_rows, within_columns, n_item_rows, n_item_rows).tocsr()
        self.followers_per_item = np.asarray(self.within_counts.sum(axis=1)).ravel()
        # the user's last three books (0 = none)
        is_last = np.append(users[1:] != users[:-1], True)
        last_positions = np.flatnonzero(is_last)
        self.last_books = np.zeros((3, n_user_rows), dtype=np.int64)
        for back in range(3):
            positions = last_positions - back
            valid = (positions >= 0) & (users[np.maximum(positions, 0)] == users[last_positions])
            self.last_books[back, users[last_positions[valid]]] = items[positions[valid]]

    def sequence_columns(self, user_rows: np.ndarray, item_rows: np.ndarray) -> list[np.ndarray]:
        lookup = lambda matrix, rows, columns: np.asarray(matrix[rows, columns]).ravel().astype(np.float64)
        last, second, third = self.last_books[0][user_rows], self.last_books[1][user_rows], self.last_books[2][user_rows]
        after_last = lookup(self.within_counts, last, item_rows)
        return [np.log1p(lookup(self.next_counts, last, item_rows)),
                np.log1p(after_last),
                after_last / np.maximum(self.followers_per_item[last], 1),
                np.log1p(lookup(self.within_counts, second, item_rows) + lookup(self.within_counts, third, item_rows)),
                np.log1p(lookup(self.within_counts, item_rows, last))]

    # readers of each item after t (+ an offset in days): position of the first interaction later than t + offset_days
    def reader_position_after(self, item_rows: np.ndarray, days: np.ndarray, offset_days: int) -> np.ndarray:
        offsets = np.clip(days.astype(np.int64) + offset_days - self.earliest_day, -1, KEY_STRIDE - 1)
        return np.searchsorted(self.reader_keys, item_rows.astype(np.int64) * KEY_STRIDE + offsets, side="right")

    def reader_timing_columns(self, item_rows: np.ndarray, days: np.ndarray) -> list[np.ndarray]:
        item_rows = item_rows.astype(np.int64)
        segment_starts = np.searchsorted(self.reader_keys, item_rows * KEY_STRIDE, side="left")
        segment_ends = np.searchsorted(self.reader_keys, (item_rows + 1) * KEY_STRIDE, side="left")
        first_after = self.reader_position_after(item_rows, days, 0)
        columns = [np.log1p(self.reader_position_after(item_rows, days, window) - first_after) for window in READER_WINDOWS_DAYS]
        columns.append(np.log1p(first_after - self.reader_position_after(item_rows, days, -30)))
        n_readers = self.readers_per_item[item_rows]
        columns.append(np.where(n_readers > 0, (segment_ends - first_after) / np.maximum(n_readers, 1), np.nan))
        has_after, has_before = first_after < segment_ends, first_after > segment_starts
        last = len(self.sorted_days) - 1
        columns.append(np.where(has_after, np.log1p(self.sorted_days[np.minimum(first_after, last)] - days), np.nan))
        columns.append(np.where(has_before, np.log1p(days - self.sorted_days[np.maximum(first_after - 1, 0)]), np.nan))
        return columns

    def feature_names(self) -> list[str]:
        user_names = ["log_user_interactions"] + (["user_span_years"] if getattr(self.config, "use_user_span_features", False) else [])
        return [f"{name}_{part}" for part in ("difference", "first", "second") for name in self.item_feature_names()] + user_names

    # (P, item features) for one item of every pair:
    #   sub-model scores | did the user already read it (the real item never is: it was held out) |
    #   Task 4: log(1 + interactions with the item within +- window of t), item age at t (negative = first read only after t)
    def item_side(self, scores_of_side: list[np.ndarray], user_rows: np.ndarray, item_rows: np.ndarray, days: np.ndarray | None) -> np.ndarray:
        already_read = np.asarray(self.read_matrix[user_rows, item_rows]).ravel() > 0
        columns = list(scores_of_side) + [already_read]

        if self.config.use_time_features:
            if days is None:
                raise ValueError("time features need the day of every pair -> the test file has no timestamp column")
            interactions_around_day, _ = self.item_timeline.window_counts(item_rows, days, self.config.activity_window_days)
            columns += [np.log1p(interactions_around_day), (days - self.first_day_of_item[item_rows]) / DAYS_PER_YEAR]
        if getattr(self.config, "use_reader_timing_features", False):
            columns += self.reader_timing_columns(item_rows, days)
        if getattr(self.config, "use_sequence_features", False):
            columns += self.sequence_columns(user_rows, item_rows)
        if getattr(self.config, "use_user_span_features", False):
            columns += self.user_span_columns(user_rows, item_rows)
        if getattr(self.config, "use_reading_window_features", False):
            columns += self.reading_window_columns(user_rows, item_rows)

        return np.column_stack(columns).astype(np.float32)

    # features of every pair in the given order and in the swapped order: [first - second, first, second, user activity]
    def pair_features(self, pair_scores_by_model: dict[str, np.ndarray], pairs: ChoicePairs) -> tuple[np.ndarray, np.ndarray]:
        first_side = self.item_side([pair_scores_by_model[name][:, 0] for name in self.config.combined_models],
                                    pairs.user_rows, pairs.first_item_rows, pairs.days)
        second_side = self.item_side([pair_scores_by_model[name][:, 1] for name in self.config.combined_models],
                                     pairs.user_rows, pairs.second_item_rows, pairs.days)
        user_activity = np.log1p(self.interactions_per_user[pairs.user_rows])[:, None].astype(np.float32)
        if getattr(self.config, "use_user_span_features", False):
            span_years = (self.user_last_day[pairs.user_rows] - self.user_first_day[pairs.user_rows]) / DAYS_PER_YEAR
            user_activity = np.column_stack([user_activity, span_years.astype(np.float32)])

        in_given_order = np.column_stack([first_side - second_side, first_side, second_side, user_activity])
        in_swapped_order = np.column_stack([second_side - first_side, second_side, first_side, user_activity])
        return in_given_order, in_swapped_order


class PairBlender:

    def __init__(self, config: PairBlenderConfig, classifier):
        self.config = config
        self.classifier = classifier

    # average of "first is real" in the given order and "second is real" in the swapped order
    def first_item_probability(self, features: np.ndarray, swapped_features: np.ndarray) -> np.ndarray:
        return first_item_probability(self.classifier, features, swapped_features)


def make_classifier(config: PairBlenderConfig):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise ImportError("LightGBM is not installed in this environment -> pip install lightgbm") from error
    return LGBMClassifier(n_estimators=config.lightgbm_n_estimators, learning_rate=config.lightgbm_learning_rate,
                          num_leaves=config.lightgbm_num_leaves, min_child_samples=config.lightgbm_min_child_samples,
                          random_state=config.seed, verbose=-1)


# every pair in both orders: (features, label) and (swapped features, 1 - label)
def fit_on_both_orders(config: PairBlenderConfig, features, swapped_features, first_is_positive):
    labels = first_is_positive.astype(np.int64)
    return make_classifier(config).fit(np.vstack([features, swapped_features]), np.concatenate([labels, 1 - labels]))


def first_item_probability(classifier, features, swapped_features) -> np.ndarray:
    return 0.5 * (classifier.predict_proba(features)[:, 1] + 1.0 - classifier.predict_proba(swapped_features)[:, 1])


def train_pair_blender(task: PairChoiceTask, model_results: dict[str, ModelResult], config: PairBlenderConfig) -> ModelResult:

    missing_models = [model_name for model_name in config.combined_models if model_name not in model_results]
    if missing_models:
        raise KeyError(f"pair blender needs the sub-models {missing_models} -> train or load them first")

    # 1) inputs: validation pair scores (sub-models built on the training split) / test pair scores (built on everything)
    validation_feature_builder = PairFeatureBuilder(task.training_data, task.n_user_rows, task.n_item_rows, config)
    test_feature_builder = PairFeatureBuilder(task.training_and_validation_data, task.n_user_rows, task.n_item_rows, config)
    valid_features, valid_swapped_features = validation_feature_builder.pair_features(
        {name: result.valid_predictions for name, result in model_results.items()}, task.validation_pairs)
    test_features, test_swapped_features = test_feature_builder.pair_features(
        {name: result.test_predictions for name, result in model_results.items()}, task.test_pairs)
    first_is_positive = task.validation_pairs.first_is_positive

    # 2) honest score of the blend: out-of-fold P(first is real) on the validation pairs
    out_of_fold_probabilities = np.empty(len(first_is_positive), dtype=np.float32)
    folds = list(KFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed).split(valid_features))
    # several pairs with the same user and positive (Task 5 with more negatives per positive) -> whole users per fold, so no
    # positive is both learned from and scored
    pairs = task.validation_pairs
    positive_item_rows = np.where(pairs.first_is_positive, pairs.first_item_rows, pairs.second_item_rows)
    if len(np.unique(pairs.user_rows.astype(np.int64) * (int(positive_item_rows.max()) + 1) + positive_item_rows)) < len(positive_item_rows):
        fold_of_user = np.random.default_rng(config.seed).integers(0, config.n_folds, int(pairs.user_rows.max()) + 1)
        fold_of_pair = fold_of_user[pairs.user_rows]
        folds = [(np.flatnonzero(fold_of_pair != fold), np.flatnonzero(fold_of_pair == fold)) for fold in range(config.n_folds)]
    for fold_train_positions, fold_held_out_positions in folds:
        classifier = fit_on_both_orders(config, valid_features[fold_train_positions], valid_swapped_features[fold_train_positions],
                                        first_is_positive[fold_train_positions])
        out_of_fold_probabilities[fold_held_out_positions] = first_item_probability(
            classifier, valid_features[fold_held_out_positions], valid_swapped_features[fold_held_out_positions])

    for model_name in config.combined_models:
        log(f"{model_name:<22} validation accuracy {pair_choice_accuracy(model_results[model_name].valid_predictions, first_is_positive):.4f}")
    blend_accuracy = pair_choice_accuracy(np.column_stack([out_of_fold_probabilities, 1.0 - out_of_fold_probabilities]), first_is_positive)
    log(f"pair blend {'':<11} validation accuracy {blend_accuracy:.4f}  (out-of-fold, {config.n_folds} folds)")

    # 3) final blender on all validation pairs -> test P(first is real)
    blender = PairBlender(config, fit_on_both_orders(config, valid_features, valid_swapped_features, first_is_positive))
    test_probabilities = blender.first_item_probability(test_features, test_swapped_features).astype(np.float32)

    # what the blend relies on: LightGBM split counts
    importances = sorted(zip(validation_feature_builder.feature_names(), blender.classifier.feature_importances_), key=lambda entry: -entry[1])
    log("pair blend inputs: " + ", ".join(f"{name} {value}" for name, value in importances))

    return ModelResult(blender, [], out_of_fold_probabilities, test_probabilities)


def store_pair_blender(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "pair_blender")


def load_pair_blender(task_name: str) -> ModelResult:
    return load_model_result(task_name, "pair_blender")
