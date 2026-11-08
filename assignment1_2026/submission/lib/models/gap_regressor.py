import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib.custom_types import GapRegressorConfig, ModelResult, TimeGapTask
from lib.helper_functioins import log
from lib.model_storage import load_model_result, save_model_result
from lib.time_gap import AUTOREGRESSIVE_FEATURES, gap_features, mean_absolute_error_seconds

# Task 6 models (metric: mean absolute error in seconds -> the best prediction is the conditional median, PML Ex. 2.4):
#   baseline:       median of the user's previous gaps (the assignment's baseline)
#   linear_median:  log(1 + gap) = theta_0 + sum_k theta_k x_k, fitted with an absolute loss (median regression) on the
#                   autoregressive features (PML Sec. 7.1 autoregression + Sec. 2.2.2 MAE)
#   lightgbm:       gradient-boosted trees with an absolute (L1) loss on log(1 + gap), every feature of lib/time_gap.py
# trained on one held-out gap per user (validation rows), scored out-of-fold, predictions = exp(prediction) - 1
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


def make_regressor(config: GapRegressorConfig):
    if config.method == "linear_median":
        # median (0.5-quantile) regression = least absolute deviations, no penalty
        return make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs"))
    if config.method == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise ImportError("LightGBM is not installed in this environment -> pip install lightgbm") from error
        return LGBMRegressor(objective="l1", n_estimators=config.lightgbm_n_estimators, learning_rate=config.lightgbm_learning_rate,
                             num_leaves=config.lightgbm_num_leaves, min_child_samples=config.lightgbm_min_child_samples,
                             random_state=config.seed, verbose=-1)
    raise ValueError(f"unknown gap regressor method {config.method!r} -> 'linear_median' or 'lightgbm'")


def train_gap_regressor(task: TimeGapTask, config: GapRegressorConfig) -> ModelResult:

    # 1) features: validation rows from the training split, test rows from all interactions
    valid_features = gap_features(task.training_data, task.training_seconds, task.validation_user_rows, task.validation_item_rows,
                                  task.n_user_rows, task.n_item_rows)
    test_features = gap_features(task.training_and_validation_data, task.training_and_validation_seconds, task.test_user_rows,
                                 task.test_item_rows, task.n_user_rows, task.n_item_rows)
    valid_targets = np.log1p(np.maximum(task.validation_gaps, 0))
    valid_inputs = prepared_features(valid_features, config)

    # 2) honest score: out-of-fold predictions on the validation rows
    out_of_fold_log_gaps = np.empty(len(valid_targets))
    for fold_train_positions, fold_held_out_positions in KFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed).split(valid_inputs):
        fold_regressor = make_regressor(config).fit(valid_inputs[fold_train_positions], valid_targets[fold_train_positions])
        out_of_fold_log_gaps[fold_held_out_positions] = fold_regressor.predict(valid_inputs[fold_held_out_positions])
    out_of_fold_gaps = np.expm1(np.maximum(out_of_fold_log_gaps, 0.0))
    validation_mae = mean_absolute_error_seconds(out_of_fold_gaps, task.validation_gaps)
    log(f"{config.method} gap model | validation MAE {validation_mae:,.0f} s ({validation_mae / 86_400:.1f} days)  (out-of-fold, {config.n_folds} folds)")

    # 3) final model on all validation rows -> test predictions
    model = GapRegressor(config, list(valid_features.columns), make_regressor(config).fit(valid_inputs, valid_targets))
    if config.method == "lightgbm":
        importances = sorted(zip(model.feature_names, model.regressor.feature_importances_), key=lambda entry: -entry[1])
        log("gap model inputs: " + ", ".join(f"{name} {value}" for name, value in importances))

    return ModelResult(model, [], out_of_fold_gaps.astype(np.float64), model.predict(test_features))


def store_gap_model(model_result: ModelResult, task_name: str, model_name: str):
    return save_model_result(model_result, task_name, model_name)


def load_gap_model(task_name: str, model_name: str) -> ModelResult:
    return load_model_result(task_name, model_name)
