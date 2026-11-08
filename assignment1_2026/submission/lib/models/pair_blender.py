import numpy as np
from sklearn.model_selection import KFold

from lib.custom_types import ChoicePairs, ModelResult, PairBlenderConfig, PairChoiceTask, RatingData
from lib.helper_functioins import log
from lib.model_storage import load_model_result, save_model_result
from lib.pair_choice import has_interacted, interaction_keys, pair_choice_accuracy
from lib.temporal_features import ItemTimeline, first_day_per_row

# Pair blender (Tasks 2 / 4): LightGBM learns P(first item is the real one) from both items' sub-model scores + meta-features
# - trained on the validation pairs, where every sub-model scores pairs it never saw (as the rating blender of Tasks 1 / 3)
# - every pair is used in both orders (first <-> second, label flipped) and predicted as the average of both orders
#   -> the answer does not depend on which item the file calls item1
# - scored out-of-fold: each validation pair is decided by a blender that never saw it

DAYS_PER_YEAR = 365.25


class PairFeatureBuilder:

    # reference = the interactions the sub-models had for these pairs (training split for validation pairs, all for test pairs)
    def __init__(self, reference: RatingData, n_user_rows: int, n_item_rows: int, config: PairBlenderConfig):
        self.config = config
        self.n_item_rows = n_item_rows
        self.sorted_interaction_keys = interaction_keys(reference, n_item_rows)
        self.interactions_per_user = np.bincount(reference.user_rows, minlength=n_user_rows)

        if config.use_time_features:
            self.item_timeline = ItemTimeline(reference.item_rows, reference.days)
            self.first_day_of_item = first_day_per_row(reference.item_rows, reference.days, n_item_rows)

    def item_feature_names(self) -> list[str]:
        time_feature_names = ["log_interactions_around_t", "item_age_years_at_t"] if self.config.use_time_features else []
        return list(self.config.combined_models) + ["already_read"] + time_feature_names

    def feature_names(self) -> list[str]:
        return [f"{name}_{part}" for part in ("difference", "first", "second") for name in self.item_feature_names()] + ["log_user_interactions"]

    # (P, item features) for one item of every pair:
    #   sub-model scores | did the user already read it (the real item never is: it was held out) |
    #   Task 4: log(1 + interactions with the item within +- window of t), item age at t (negative = first read only after t)
    def item_side(self, scores_of_side: list[np.ndarray], user_rows: np.ndarray, item_rows: np.ndarray, days: np.ndarray | None) -> np.ndarray:
        columns = list(scores_of_side) + [has_interacted(self.sorted_interaction_keys, user_rows, item_rows, self.n_item_rows)]

        if self.config.use_time_features:
            if days is None:
                raise ValueError("time features need the day of every pair -> the test file has no timestamp column")
            interactions_around_day, _ = self.item_timeline.window_counts(item_rows, days, self.config.activity_window_days)
            columns += [np.log1p(interactions_around_day), (days - self.first_day_of_item[item_rows]) / DAYS_PER_YEAR]

        return np.column_stack(columns).astype(np.float32)

    # features of every pair in the given order and in the swapped order: [first - second, first, second, user activity]
    def pair_features(self, pair_scores_by_model: dict[str, np.ndarray], pairs: ChoicePairs) -> tuple[np.ndarray, np.ndarray]:
        first_side = self.item_side([pair_scores_by_model[name][:, 0] for name in self.config.combined_models],
                                    pairs.user_rows, pairs.first_item_rows, pairs.days)
        second_side = self.item_side([pair_scores_by_model[name][:, 1] for name in self.config.combined_models],
                                     pairs.user_rows, pairs.second_item_rows, pairs.days)
        user_activity = np.log1p(self.interactions_per_user[pairs.user_rows])[:, None].astype(np.float32)

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
    folds = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    for fold_train_positions, fold_held_out_positions in folds.split(valid_features):
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
