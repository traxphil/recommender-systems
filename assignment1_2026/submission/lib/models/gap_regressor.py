import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib.custom_types import GapRegressorConfig, ModelResult, RatingData, TimeGapTask
from lib.helper_functioins import log
from lib.model_storage import load_model_result, save_model_result
from lib.time_gap import AUTOREGRESSIVE_FEATURES, gap_features, mean_absolute_error_seconds

# Task 6 models (metric: mean absolute error in seconds -> the best prediction is the conditional median, PML Ex. 2.4):
#   baseline:       median of the user's previous gaps (the assignment's baseline)
#   linear_median:  log(1 + gap) = theta_0 + sum_k theta_k x_k, fitted with an absolute loss (median regression) on the
#                   autoregressive features (PML Sec. 7.1 autoregression + Sec. 2.2.2 MAE)
#   lightgbm:       gradient-boosted trees with an absolute (L1) loss on log(1 + gap), every feature of lib/time_gap.py
# trained on one held-out gap per user (validation rows) + optionally the user's earlier gaps (n_earlier_gaps_as_training_rows),
# scored out-of-fold on the validation rows, predictions = exp(prediction) - 1
# (the median survives the monotone back-transformation, the mean would not)


class MedianGapBaseline:

    def __init__(self, gap_seconds_by_user: pd.Series, overall_median_gap: float):
        self.median_gap_of_user = gap_seconds_by_user
        self.overall_median_gap = overall_median_gap

    def predict(self, user_rows: np.ndarray) -> np.ndarray:
        return self.median_gap_of_user.reindex(user_rows).fillna(self.overall_median_gap).to_numpy(dtype=np.float64)


def fit_median_gap_baseline(reference_user_rows: np.ndarray, reference_seconds: np.ndarray) -> MedianGapBaseline:
    is_same_user_as_previous = np.append(False, reference_user_rows[1:] == reference_user_rows[:-1])
    gap_positions = np.flatnonzero(is_same_user_as_previous)
    gaps = pd.Series(reference_seconds[gap_positions] - reference_seconds[gap_positions - 1], dtype=np.float64)
    return MedianGapBaseline(gaps.groupby(reference_user_rows[gap_positions]).median(), float(gaps.median()))


def train_median_gap_baseline(task: TimeGapTask) -> ModelResult:
    model = fit_median_gap_baseline(task.training_data.user_rows, task.training_seconds)
    valid_predictions = model.predict(task.validation_user_rows)
    log(f"median gap baseline | validation MAE {mean_absolute_error_seconds(valid_predictions, task.validation_gaps):,.0f} s "
        f"({mean_absolute_error_seconds(valid_predictions, task.validation_gaps) / 86_400:.1f} days)")

    model = fit_median_gap_baseline(task.training_and_validation_data.user_rows, task.training_and_validation_seconds)
    return ModelResult(model, [], valid_predictions, model.predict(task.test_user_rows))


class GapRegressor:

    def __init__(self, config: GapRegressorConfig, feature_names: list[str], regressor):
        self.config = config
        self.feature_names = feature_names
        self.regressor = regressor

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.expm1(np.maximum(self.regressor.predict(prepared_features(features, self.config)), 0.0))


# linear model: the autoregressive columns, missing earlier gaps filled with the user's median gap
# LightGBM: every column, missing values stay NaN (LightGBM routes them itself)
def prepared_features(features: pd.DataFrame, config: GapRegressorConfig):
    if config.method == "linear_median":
        autoregressive = features[AUTOREGRESSIVE_FEATURES].copy()
        for column in ("last_log_gap", "second_last_log_gap", "third_last_log_gap", "mean_log_gap"):
            autoregressive[column] = autoregressive[column].fillna(autoregressive["median_log_gap"])
        return autoregressive.fillna(0.0).to_numpy(dtype=np.float64)
    return features.to_numpy(dtype=np.float32)


# bagging: several regressors with different seeds / random row and feature subsets, predictions averaged (log scale)
class BaggedRegressor:

    def __init__(self, regressors: list):
        self.regressors = regressors

    def fit(self, inputs, targets):
        for regressor in self.regressors:
            regressor.fit(inputs, targets)
        return self

    def predict(self, inputs) -> np.ndarray:
        return np.mean([regressor.predict(inputs) for regressor in self.regressors], axis=0)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.mean([regressor.feature_importances_ for regressor in self.regressors], axis=0)


def make_regressor(config: GapRegressorConfig):
    if config.method == "linear_median":
        # median (0.5-quantile) regression = least absolute deviations, no penalty
        return make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs"))
    if config.method == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise ImportError("LightGBM is not installed in this environment -> pip install lightgbm") from error
        bagging_fraction, feature_fraction = getattr(config, "bagging_fraction", 1.0), getattr(config, "feature_fraction", 1.0)
        make_lightgbm = lambda seed: LGBMRegressor(
            objective="l1", n_estimators=config.lightgbm_n_estimators, learning_rate=config.lightgbm_learning_rate,
            num_leaves=config.lightgbm_num_leaves, min_child_samples=config.lightgbm_min_child_samples,
            subsample=bagging_fraction, subsample_freq=1 if bagging_fraction < 1.0 else 0, colsample_bytree=feature_fraction,
            random_state=seed, verbose=-1)
        n_bagged_models = getattr(config, "n_bagged_models", 1)
        if n_bagged_models > 1:
            return BaggedRegressor([make_lightgbm(config.seed + model_number) for model_number in range(n_bagged_models)])
        return make_lightgbm(config.seed)
    raise ValueError(f"unknown gap regressor method {config.method!r} -> 'linear_median' or 'lightgbm'")


# extra training rows: cut k = every user's last k training interactions removed -> the first removed one is the "next"
# interaction, its gap to the one before is the target, the features come from what is left (as for the validation rows)
def earlier_gap_rows(task: TimeGapTask, n_cuts: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    reference, seconds = task.training_data, task.training_seconds
    users = reference.user_rows                                 # chronological per user
    position_from_end = (pd.Series(np.arange(len(users))).groupby(users).rank(ascending=False).to_numpy().astype(np.int64) - 1)
    features, targets, user_rows = [], [], []
    for cut in range(1, n_cuts + 1):
        keep = position_from_end >= cut
        next_positions = np.flatnonzero(position_from_end == cut - 1)
        has_previous = (next_positions > 0) & (users[np.maximum(next_positions - 1, 0)] == users[next_positions])
        next_positions = next_positions[has_previous]
        cut_reference = RatingData(users[keep], reference.item_rows[keep], reference.ratings[keep], reference.days[keep])
        features.append(gap_features(cut_reference, seconds[keep], users[next_positions], reference.item_rows[next_positions],
                                     task.n_user_rows, task.n_item_rows))
        targets.append(np.log1p(np.maximum(seconds[next_positions] - seconds[next_positions - 1], 0)))
        user_rows.append(users[next_positions])
        features[-1]["rows_from_end"] = cut
    return pd.concat(features, ignore_index=True), np.concatenate(targets), np.concatenate(user_rows)


def train_gap_regressor(task: TimeGapTask, config: GapRegressorConfig) -> ModelResult:

    # 1) features: validation rows from the training split, test rows from all interactions
    valid_features = gap_features(task.training_data, task.training_seconds, task.validation_user_rows, task.validation_item_rows,
                                  task.n_user_rows, task.n_item_rows)
    test_features = gap_features(task.training_and_validation_data, task.training_and_validation_seconds, task.test_user_rows,
                                 task.test_item_rows, task.n_user_rows, task.n_item_rows)
    use_rows_from_end = getattr(config, "use_rows_from_end_feature", False)
    if use_rows_from_end:
        valid_features["rows_from_end"] = 0
        test_features["rows_from_end"] = 0
    valid_targets = np.log1p(np.maximum(task.validation_gaps, 0))
    valid_inputs = prepared_features(valid_features, config)
    if config.n_earlier_gaps_as_training_rows > 0:
        extra_features, extra_targets, extra_user_rows = earlier_gap_rows(task, config.n_earlier_gaps_as_training_rows)
        if not use_rows_from_end:
            extra_features = extra_features.drop(columns="rows_from_end")
        extra_inputs = prepared_features(extra_features, config)
    else:
        extra_inputs, extra_targets, extra_user_rows = valid_inputs[:0], valid_targets[:0], task.validation_user_rows[:0]

    # 2) honest score: out-of-fold predictions on the validation rows
    #    an extra row joins a fold's training part only if its user's validation row is not held out in that fold
    fold_of_user = np.full(task.n_user_rows, -1)
    out_of_fold_log_gaps = np.empty(len(valid_targets))
    folds = list(KFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed).split(valid_inputs))
    for fold, (_, fold_held_out_positions) in enumerate(folds):
        fold_of_user[task.validation_user_rows[fold_held_out_positions]] = fold
    for fold, (fold_train_positions, fold_held_out_positions) in enumerate(folds):
        extra_positions = np.flatnonzero(fold_of_user[extra_user_rows] != fold)
        fold_regressor = make_regressor(config).fit(np.concatenate([valid_inputs[fold_train_positions], extra_inputs[extra_positions]]),
                                                    np.concatenate([valid_targets[fold_train_positions], extra_targets[extra_positions]]))
        out_of_fold_log_gaps[fold_held_out_positions] = fold_regressor.predict(valid_inputs[fold_held_out_positions])
    out_of_fold_gaps = np.expm1(np.maximum(out_of_fold_log_gaps, 0.0))
    validation_mae = mean_absolute_error_seconds(out_of_fold_gaps, task.validation_gaps)
    log(f"{config.method} gap model | validation MAE {validation_mae:,.0f} s ({validation_mae / 86_400:.1f} days)  (out-of-fold, {config.n_folds} folds)")

    # 3) final model on all validation rows (+ the extra rows) -> test predictions
    model = GapRegressor(config, list(valid_features.columns),
                         make_regressor(config).fit(np.concatenate([valid_inputs, extra_inputs]), np.concatenate([valid_targets, extra_targets])))
    if config.method == "lightgbm":
        importances = sorted(zip(model.feature_names, model.regressor.feature_importances_), key=lambda entry: -entry[1])
        log("gap model inputs: " + ", ".join(f"{name} {value}" for name, value in importances))

    return ModelResult(model, [], out_of_fold_gaps.astype(np.float64), model.predict(test_features))


def store_gap_model(model_result: ModelResult, task_name: str, model_name: str):
    return save_model_result(model_result, task_name, model_name)


def load_gap_model(task_name: str, model_name: str) -> ModelResult:
    return load_model_result(task_name, model_name)
