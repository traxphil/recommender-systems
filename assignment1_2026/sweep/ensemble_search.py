"""
Ensemble search: which of the stored sub-models should a blend combine, and with which blender? (run after training)
Used by the task<N>_ensemble_sweep.ipynb notebooks in this folder.

- uses only the stored predictions in trained_models/<task>/ -> nothing is retrained
- every score is out-of-fold on the validation rows, the same way the submitted blend is scored, averaged over several
  fold seeds (= different splits of the validation rows into folds):
    Tasks 1 / 3 (rating blender, lib/models/blender.py):     MSE of the validation ratings      (lower is better)
    Tasks 2 / 4 / 5 (pair blender, lib/models/pair_blender.py): accuracy of the validation pairs (higher is better)
- internally every scorer gives one loss per validation row (squared error / 1 - "pair decided correctly"); two compositions
  are compared row by row on the same validation rows: change = mean of the loss differences, +- its standard error
  -> a change within ~2 standard errors is noise
- reports: output/<task>_ensemble/ensemble_<start time>.json (own folder: the hyperparameter overview reads output/<task>/)
"""

import json
import sys
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, cross_val_predict

# this file lives in assignment1_2026/sweep, the model code in assignment1_2026/submission
SWEEP_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = SWEEP_DIR.parent / "submission"
OUTPUT_DIR = SWEEP_DIR / "output"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from lib.config import RATING_MAX, RATING_MIN                    # noqa: E402  (needs the path above)
from lib.custom_types import ModelResult                         # noqa: E402
from lib.helper_functioins import mse                            # noqa: E402
from lib.model_storage import load_model_result, trained_model_path  # noqa: E402
from lib.models.blender import PROBABILITY_SUB_MODELS, blend_features, make_regressor  # noqa: E402
from lib.models.pair_blender import PairFeatureBuilder, first_item_probability, fit_on_both_orders  # noqa: E402
from lib.pair_choice import pair_choice_accuracy                 # noqa: E402


# the stored results of the sub-models in trained_models/<task>/; models that are not stored yet are left out (printed)
def load_stored_sub_models(task_name: str, model_names) -> dict[str, ModelResult]:
    stored_names = [name for name in model_names if trained_model_path(task_name, name).exists()]
    missing_names = [name for name in model_names if name not in stored_names]
    if missing_names:
        print(f"not stored in trained_models/{task_name} yet -> left out: {', '.join(missing_names)}")
    return {name: load_model_result(task_name, name) for name in stored_names}


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Scorers: one loss per validation row for a blender config (cached: the stages ask for the same blends again)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class BlendScorer:
    models_field: str                       # the config field that lists the combined sub-models
    metric_name: str
    higher_is_better: bool

    def __init__(self, task, sub_model_results: dict[str, ModelResult], fold_seeds=(0, 1, 2)):
        self.task = task
        self.fold_seeds = tuple(fold_seeds)
        self.valid_predictions = {name: result.valid_predictions for name, result in sub_model_results.items()}
        self.cached_losses = {}

    def losses(self, config) -> np.ndarray:
        key = repr(config)
        if key not in self.cached_losses:
            self.cached_losses[key] = np.mean([self.losses_for_fold_seed(config, fold_seed) for fold_seed in self.fold_seeds], axis=0)
        return self.cached_losses[key]

    def with_models(self, config, model_names):
        return replace(config, **{self.models_field: tuple(model_names)})

    # the metric from a mean loss, and a loss change as a metric change ("+" = better for accuracy, worse for MSE)
    def score(self, mean_loss: float) -> float:
        return 1.0 - mean_loss if self.higher_is_better else mean_loss

    def describe(self, mean_loss: float) -> str:
        return f"{self.metric_name} {self.score(mean_loss):.5f}"

    def describe_change(self, change: dict) -> str:
        metric_change = -change["change"] if self.higher_is_better else change["change"]
        return f"{self.metric_name} {metric_change:+.5f} +- {change['standard_error']:.5f}"


# Tasks 1 / 3: squared error of every validation rating (rating blender: ridge or LightGBM regression)
class RatingBlendScorer(BlendScorer):
    models_field, metric_name, higher_is_better = "blended_models", "MSE", False

    def losses_for_fold_seed(self, config, fold_seed: int) -> np.ndarray:
        validation, ratings = self.task.validation_data, self.task.validation_data.ratings
        features = blend_features(self.valid_predictions, validation.user_rows, validation.item_rows, validation.days,
                                  self.task.training_data, self.task.n_user_rows, self.task.n_item_rows, config)
        folds = KFold(n_splits=config.n_folds, shuffle=True, random_state=fold_seed)
        predictions = cross_val_predict(make_regressor(config), features, ratings, cv=folds)
        return (np.clip(predictions, RATING_MIN, RATING_MAX) - ratings) ** 2

    # every rating model on its own (the zero models predict P(0), not a rating)
    def single_model_scores(self) -> dict[str, float]:
        return {name: mse(predictions, self.task.validation_data.ratings)
                for name, predictions in self.valid_predictions.items() if name not in PROBABILITY_SUB_MODELS}


# Tasks 2 / 4 / 5: 1 - "pair decided correctly" of every validation pair (pair blender: LightGBM classifier, both orders)
class PairBlendScorer(BlendScorer):
    models_field, metric_name, higher_is_better = "combined_models", "accuracy", True

    def losses_for_fold_seed(self, config, fold_seed: int) -> np.ndarray:
        pairs = self.task.validation_pairs
        feature_builder = PairFeatureBuilder(self.task.training_data, self.task.n_user_rows, self.task.n_item_rows, config)
        features, swapped_features = feature_builder.pair_features(self.valid_predictions, pairs)
        probabilities = np.empty(len(pairs.first_is_positive))
        folds = KFold(n_splits=config.n_folds, shuffle=True, random_state=fold_seed)
        for fold_train_positions, fold_held_out_positions in folds.split(features):
            classifier = fit_on_both_orders(config, features[fold_train_positions], swapped_features[fold_train_positions],
                                            pairs.first_is_positive[fold_train_positions])
            probabilities[fold_held_out_positions] = first_item_probability(
                classifier, features[fold_held_out_positions], swapped_features[fold_held_out_positions])
        # as pair_choice_accuracy: an undecided pair (P = 0.5) counts as a coin flip
        is_correct = np.where(probabilities == 0.5, 0.5, (probabilities > 0.5) == pairs.first_is_positive)
        return 1.0 - is_correct

    def single_model_scores(self) -> dict[str, float]:
        return {name: pair_choice_accuracy(scores, self.task.validation_pairs.first_is_positive)
                for name, scores in self.valid_predictions.items()}


# change of the loss from the reference blend to the candidate blend (same validation rows) +- its standard error
def paired_change(reference_losses: np.ndarray, candidate_losses: np.ndarray) -> dict:
    differences = candidate_losses - reference_losses
    return {"change": float(differences.mean()), "standard_error": float(differences.std(ddof=1) / np.sqrt(len(differences)))}


# models that may only be dropped together (one unit, e.g. bias model + item kNN: the rating blend uses their difference);
# every other model is a unit on its own; units in the order of model_names, a group only with its stored members
def blend_units(model_names, linked_models=()) -> list[tuple[str, ...]]:
    units = []
    for name in model_names:
        group = next((tuple(member for member in linked if member in model_names) for linked in linked_models if name in linked), (name,))
        if group not in units:
            units.append(group)
    return units


def unit_names(units) -> list[str]:
    return [name for unit in units for name in unit]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# The stages of the search (one notebook cell each)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# 1) every sub-model on its own + the blend of all stored models (best first)
def single_model_table(scorer: BlendScorer, config, model_names) -> dict:
    single_scores = scorer.single_model_scores()
    all_models_loss = float(scorer.losses(scorer.with_models(config, model_names)).mean())
    for name, value in sorted(single_scores.items(), key=lambda item: item[1], reverse=scorer.higher_is_better):
        print(f"{name:<34} {scorer.metric_name} {value:.5f}")
    print(f"{'blend of all ' + str(len(model_names)) + ' (out-of-fold)':<34} {scorer.describe(all_models_loss)}")
    return {"metric": scorer.metric_name, "single_models": single_scores, "all_models": scorer.score(all_models_loss)}


# 2) the blend without each unit -> how much the unit adds on top of all the others (most useful first)
def drop_one_table(scorer: BlendScorer, config, units) -> list[dict]:
    if len(units) < 2:
        print("only one unit stored -> nothing to drop")
        return []
    all_losses = scorer.losses(scorer.with_models(config, unit_names(units)))
    rows = [{"without": list(unit),
             **paired_change(all_losses, scorer.losses(scorer.with_models(config, unit_names([u for u in units if u != unit]))))}
            for unit in units]
    rows.sort(key=lambda row: row["change"], reverse=True)
    for row in rows:
        print(f"without {' + '.join(row['without']):<40} {scorer.describe_change(row)}")
    return rows


# 3) drop the least useful unit again and again, as long as the blend stays within `tolerance` of the blend of all units
#    (compared with the blend of all, not with the previous step -> small losses cannot add up; tolerance in MSE / accuracy)
def backward_elimination(scorer: BlendScorer, config, units, tolerance: float) -> dict:
    kept_units = list(units)
    all_losses = scorer.losses(scorer.with_models(config, unit_names(kept_units)))
    path = [{"removed": None, "kept": unit_names(kept_units), "score": scorer.score(float(all_losses.mean())),
             "change": 0.0, "standard_error": 0.0}]
    print(f"all {len(kept_units)} units {'':<32} {scorer.describe(float(all_losses.mean()))}")

    while len(kept_units) > 1:
        candidates = []
        for unit in kept_units:
            losses = scorer.losses(scorer.with_models(config, unit_names([u for u in kept_units if u != unit])))
            candidates.append((float(losses.mean()), unit, losses))
        candidate_loss, unit, losses = min(candidates, key=lambda candidate: candidate[0])
        change = paired_change(all_losses, losses)
        if change["change"] > tolerance:
            print(f"stop: dropping {' + '.join(unit)} next would cost {scorer.describe_change(change)} (more than {tolerance})")
            break
        kept_units.remove(unit)
        path.append({"removed": list(unit), "kept": unit_names(kept_units), "score": scorer.score(candidate_loss), **change})
        print(f"drop {' + '.join(unit):<36} {scorer.describe(candidate_loss)} | vs. all {scorer.describe_change(change)}")

    selected_models = tuple(unit_names(kept_units))
    print(f"-> {len(selected_models)} models: {selected_models}")
    return {"tolerance": tolerance, "path": path, "selected_models": selected_models}


# 4) blender settings on the selected models, each compared with the reference (the first entry, e.g. the submitted blender)
#    recommended: the best one if it beats the reference by more than 2 standard errors, otherwise the reference
def blender_comparison(scorer: BlendScorer, blender_candidates: dict, selected_models) -> dict:
    configs = {name: scorer.with_models(config, selected_models) for name, config in blender_candidates.items()}
    losses = {name: scorer.losses(config) for name, config in configs.items()}
    reference_name = next(iter(configs))
    rows = [{"blender": name, "score": scorer.score(float(losses[name].mean())), **paired_change(losses[reference_name], losses[name])}
            for name in configs]
    for row in rows:
        print(f"{row['blender']:<40} {scorer.metric_name} {row['score']:.5f} | vs. {reference_name}: {scorer.describe_change(row)}")

    best = min(rows, key=lambda row: row["change"])
    recommended_name = best["blender"] if best["change"] < -2 * best["standard_error"] else reference_name
    print(f"-> recommended blender: {recommended_name}")
    return {"rows": rows, "reference": reference_name, "recommended": recommended_name,
            "recommended_config": asdict(configs[recommended_name])}


# 5) what to copy into lib/config.py: the flags dict (True = in the blend; a model that is not stored keeps its current flag)
#    + the recommended blender as the closest config of lib/config.py (lib_configs: name -> config) and its changed settings
def print_recommendation(scorer: BlendScorer, blender_candidates: dict, elimination: dict, blenders: dict,
                         flags_name: str, current_flags: dict, stored_models, lib_configs: dict) -> dict:
    recommended = blender_candidates[blenders["recommended"]]

    def changed_settings(lib_config) -> dict:
        return {field.name: getattr(recommended, field.name) for field in fields(recommended)
                if field.name != scorer.models_field and getattr(recommended, field.name) != getattr(lib_config, field.name)}

    flags = {name: (name in elimination["selected_models"]) if name in stored_models else current_flag
             for name, current_flag in current_flags.items()}
    print(f"{flags_name} = {{")
    for name, is_blended in flags.items():
        print(f'    "{name}": {is_blended},' + ("" if name in stored_models else "   # not stored -> not checked, current flag kept"))
    print("}")

    closest_name = min(lib_configs, key=lambda name: len(changed_settings(lib_configs[name])))
    changes = changed_settings(lib_configs[closest_name])
    print(f"blender: {blenders['recommended']} = {closest_name}" + (f" with {changes}" if changes else " as it is"))
    return {"flags": flags, "blender": blenders["recommended"], "lib_config": closest_name, "changed_settings": changes}


# 6) the report file
def save_ensemble_report(task_name: str, report: dict) -> Path:
    output_path = OUTPUT_DIR / f"{task_name}_ensemble" / f"ensemble_{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"report -> {output_path}")
    return output_path
