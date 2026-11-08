import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

from lib.config import RATING_MAX, RATING_MIN
from lib.custom_types import BlenderConfig, ModelResult, RatingData, RatingTask
from lib.helper_functioins import log, mse
from lib.model_storage import load_model_result, save_model_result
from lib.temporal_features import UserTimeline, first_day_per_row

# Stacking: a small model learns how to combine the sub-models' predictions
# - trained on the validation split, where every sub-model predicts ratings it never trained on
# - applied to the test predictions of the same sub-models refit on 100 % (as the Netflix Prize "probe" -> "quiz" blends)
# - scored out-of-fold: each validation rating is predicted by a blender that never saw it

# sub-models whose "predictions" are probabilities instead of ratings (no rating MSE printed for them)
PROBABILITY_SUB_MODELS = {"zero_model", "time_zero_model"}

DAYS_PER_YEAR = 365.25


class Blender:

    def __init__(self, config: BlenderConfig, regressor):
        self.config = config
        self.regressor = regressor

    def predict(self, blend_features: np.ndarray) -> np.ndarray:
        return np.clip(self.regressor.predict(blend_features), RATING_MIN, RATING_MAX).astype(np.float32)


# when a rating happens relative to the reference ratings (a row's own rating is never among them -> nothing to leave out):
#   log(1 + #ratings of the user on that day / within +- activity_window_days)  -> bursts (bulk shelving, active phases)
#   log(1 + #ratings of the user before that day)                               -> experience ("acquired tastes", PML Sec. 7.3.3)
#   item age in years (time since the item's first rating)                      -> new books are rated differently
#   years since 2000                                                            -> slow drift of the overall level
# -> the blender can learn e.g. "trust the user-period biases when the user rated a lot around t"
def time_features(user_rows: np.ndarray, item_rows: np.ndarray, days: np.ndarray | None, reference: RatingData,
                  n_item_rows: int, activity_window_days: int) -> list[np.ndarray]:

    if days is None:
        raise ValueError("time features need the day of every row -> the test file has no timestamp column")

    user_timeline = UserTimeline(reference.user_rows, reference.days)
    same_day_ratings, _ = user_timeline.window_counts(user_rows, days, 0)
    window_ratings, _ = user_timeline.window_counts(user_rows, days, activity_window_days)
    earlier_ratings = user_timeline.counts_before(user_rows, days)

    # unknown item (no reference ratings) -> age 0
    first_day_of_item = first_day_per_row(reference.item_rows, reference.days, n_item_rows)[item_rows]
    item_age_in_years = np.nan_to_num((days - first_day_of_item) / DAYS_PER_YEAR)

    return [np.log1p(same_day_ratings), np.log1p(window_ratings), np.log1p(earlier_ratings), item_age_in_years, days / DAYS_PER_YEAR]


# (N, inputs): one column per blended sub-model + optionally log(1 + #ratings) of user and item + optionally time features
# count_reference = the ratings the sub-models had for these rows (training split for validation rows, 100 % for test rows)
def blend_features(predictions_by_model: dict[str, np.ndarray], user_rows: np.ndarray, item_rows: np.ndarray, days: np.ndarray | None,
                   count_reference: RatingData, n_user_rows: int, n_item_rows: int, config: BlenderConfig) -> np.ndarray:

    feature_columns = [predictions_by_model[model_name] for model_name in config.blended_models]

    if config.use_count_features:
        ratings_per_user = np.bincount(count_reference.user_rows, minlength=n_user_rows)
        ratings_per_item = np.bincount(count_reference.item_rows, minlength=n_item_rows)
        feature_columns += [np.log1p(ratings_per_user[user_rows]), np.log1p(ratings_per_item[item_rows])]

    if config.use_time_features:
        feature_columns += time_features(user_rows, item_rows, days, count_reference, n_item_rows, config.activity_window_days)

    return np.column_stack(feature_columns).astype(np.float32)


def blend_feature_names(config: BlenderConfig) -> list[str]:
    count_feature_names = ["log_user_ratings", "log_item_ratings"] if config.use_count_features else []
    time_feature_names = (["log_same_day_ratings", "log_window_ratings", "log_earlier_ratings", "item_age_years", "years_since_2000"]
                          if config.use_time_features else [])
    return list(config.blended_models) + count_feature_names + time_feature_names


def make_regressor(config: BlenderConfig):
    if config.method == "ridge":
        return Ridge(alpha=config.ridge_alpha)

    if config.method == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise ImportError("LightGBM is not installed in this environment -> pip install lightgbm") from error
        return LGBMRegressor(n_estimators=config.lightgbm_n_estimators, learning_rate=config.lightgbm_learning_rate,
                             num_leaves=config.lightgbm_num_leaves, min_child_samples=config.lightgbm_min_child_samples,
                             random_state=config.seed, verbose=-1)

    raise ValueError(f"unknown blender method {config.method!r} (expected 'ridge' or 'lightgbm')")


def train_blender(task: RatingTask, model_results: dict[str, ModelResult], config: BlenderConfig) -> ModelResult:

    missing_models = [model_name for model_name in config.blended_models if model_name not in model_results]
    if missing_models:
        raise KeyError(f"blender needs the sub-models {missing_models} -> train or load them first")

    # 1) inputs: validation predictions (sub-models fit on the training split) / test predictions (sub-models refit on 100 %)
    validation, valid_ratings = task.validation_data, task.validation_data.ratings
    valid_features = blend_features({name: result.valid_predictions for name, result in model_results.items()},
                                    validation.user_rows, validation.item_rows, validation.days, task.training_data,
                                    task.n_user_rows, task.n_item_rows, config)
    test_features = blend_features({name: result.test_predictions for name, result in model_results.items()},
                                   task.test_user_rows, task.test_item_rows, task.test_days, task.training_and_validation_data,
                                   task.n_user_rows, task.n_item_rows, config)

    # 2) honest score of the blend: out-of-fold predictions on the validation rows
    folds = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    out_of_fold_predictions = np.clip(cross_val_predict(make_regressor(config), valid_features, valid_ratings, cv=folds),
                                      RATING_MIN, RATING_MAX).astype(np.float32)

    for model_name in config.blended_models:
        if model_name not in PROBABILITY_SUB_MODELS:
            log(f"{model_name:<22} valid MSE {mse(model_results[model_name].valid_predictions, valid_ratings):.4f}")
    log(f"{config.method} blend {'':<14} valid MSE {mse(out_of_fold_predictions, valid_ratings):.4f}  (out-of-fold, {config.n_folds} folds)")

    # 3) final blender on all validation rows -> test predictions
    blender = Blender(config, make_regressor(config).fit(valid_features, valid_ratings))
    test_predictions = blender.predict(test_features)

    # what the blend relies on: ridge weights / LightGBM split counts
    importances = blender.regressor.coef_ if config.method == "ridge" else blender.regressor.feature_importances_
    log(f"{config.method} blend inputs: " + ", ".join(f"{name} {value:.3g}" for name, value in zip(blend_feature_names(config), importances)))

    return ModelResult(blender, [], out_of_fold_predictions, test_predictions)


def store_blender(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, f"{model_result.model.config.method}_blender")


def load_blender(task_name: str, method: str) -> ModelResult:
    return load_model_result(task_name, f"{method}_blender")
