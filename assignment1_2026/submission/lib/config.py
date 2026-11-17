from pathlib import Path

import torch

from lib.custom_types import (AutoRecConfig, BayesianFactorizationMachineConfig, BiasModelConfig, BlenderConfig, EaseConfig,
                              FactorizationMachineConfig, GapRegressorConfig, IntegratedModelConfig, ItemCosineConfig,
                              ItemKnnConfig, LatentFactorConfig, MultVaeConfig, PairBlenderConfig, RatingTransformerConfig,
                              SasRecConfig, TemporalBiasConfig, TimeBiasModelConfig, TimeFactorizationMachineConfig,
                              TimeItemKnnConfig, TimeSVDppConfig, TimeZeroModelConfig, TransitionConfig, ZeroModelConfig)


# #########################################################################################
# Global
# #########################################################################################

# Rating Range
RATING_MIN, RATING_MAX = 0.0, 5.0
# Tasks 1 / 3: rating models train on the ratings > 0 only (the leaderboard leaves the zeros out, lecture 6 Nov) - see
# rating_model_task in assignment1.py; the zero models keep the zeros
TRAIN_RATING_MODELS_WITHOUT_ZERO_RATINGS = True

# GPU if GPU is available / otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training file -> 90 % training / 10 % validation; all sub-models of a task share this split (the blender needs it)
VALIDATION_FRACTION = 0.1
SPLIT_SEED = 0

# stored sub-models and blenders: assignment1_2026/trained_models/<task>/<model>.pt
TRAINED_MODELS_DIR = Path(__file__).resolve().parents[2] / "trained_models"

# timestamps -> day numbers (day 0 = 2000-01-01); the files contain a few impossible timestamps (years 25, 1972, 3645)
# -> 0.05 % of the rows lie outside 2000 - 2017 and are clipped to this range
FIRST_PLAUSIBLE_TIMESTAMP = 946_684_800     # 2000-01-01 00:00 UTC
LAST_PLAUSIBLE_TIMESTAMP = 1_514_764_800    # 2018-01-01 00:00 UTC


# #########################################################################################
# Task 1 - rating prediction from (user, item) only
# #########################################################################################

# ---- shared baseline mu + b_u + b_i of the interlinked models ----------------------------
# the bias model and the item kNN's baseline must be the same fit: the blend combines them as (kNN - bias model)
# = the kNN's neighbourhood correction (ridge weights bias model -3.05 / item kNN +3.34), which only works if both
# baselines are identical (with 1 / 5 for the bias model and 3 / 3 inside the kNN the ridge blend fell from 0.7302 to 0.7368)
# -> defined once here, used by TASK1_BIAS_MODEL and TASK1_ITEM_KNN; after a change retrain both + the blend
# SVD / SVD++ / FM keep their own bias lambdas: their biases are trained further together with the factors anyway
# one lambda per side: closed-form grid on the validation split (user x item in 0.5 ... 25): 3 / 3 -> 0.7647, 1 / 5 -> 0.7628
# matches the empirical-Bayes estimate lambda = noise variance / bias variance: users ~1.8, items ~6.5
# (users' biases vary much more than the books' -> less shrinkage for b_u)
# sweep 8 Nov (user 0.5 ... 3 x item 3 ... 24 x rounds 10 ... 100): 1 / 5 best for every number of rounds;
# 10 rounds are not converged: 10 -> 0.76284, 20 -> 0.76269, 50 -> 0.762655, 100 -> 0.762654 (< 1 s per fit)
TASK1_SHARED_BASELINE_BIASES = BiasModelConfig(
    user_bias_regularization=1.0,
    item_bias_regularization=5.0,
    n_iterations=50,
)

# ---- bias model: mu + b_u + b_i in closed form ------------------------------------------
TASK1_BIAS_MODEL = TASK1_SHARED_BASELINE_BIASES

# ---- SVD -------------------------------------------------------------------------------
# best SVD of the sweep (archive/assignment1_2026_queue_sweep/sweep_summary.md): mean valid MSE 0.7498 over validation splits 0 / 1 / 2
# bias lambdas, sweep 8 Nov (split seed 0): 1 / 5 -> 0.74856, 1 / 3 -> 0.74896, 3 / 5 -> 0.74975, 3 / 3 -> 0.75021
TASK1_SVD = LatentFactorConfig(
    use_implicit_feedback=False,
    use_test_pairs_as_implicit_feedback=False,
    n_factors=120,
    learning_rate=2.5e-4,
    user_bias_regularization=1.0,           # own lambdas (independent of TASK1_SHARED_BASELINE_BIASES)
    item_bias_regularization=5.0,
    factor_regularization=8.0,
    ratings_per_batch=512,
    max_epochs=1000,
    early_stopping_patience=6,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- SVD++ -----------------------------------------------------------------------------
# best SVD++ of the sweep: mean valid MSE 0.7432 over validation splits 0 / 1 / 2
TASK1_SVDPP = LatentFactorConfig(
    use_implicit_feedback=True,
    use_test_pairs_as_implicit_feedback=True,     # every known pair, 0-star ratings included (assignment1.rating_model_task)
                                                  # 15 Nov without the 0-star ratings: 0.57210 -> 0.57193 (with them, 14 Nov: no gain)
    n_factors=500,
    learning_rate=2.5e-4,
    user_bias_regularization=1.0,           # own lambdas (independent of TASK1_SHARED_BASELINE_BIASES)
    item_bias_regularization=5.0,
    factor_regularization=8.0,
    ratings_per_batch=512,
    max_epochs=1000,
    early_stopping_patience=6,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- item-item kNN on the bias model's residuals ----------------------------------------
# sweep 8 Nov (still with the old 3 / 3 baseline): 25 -> 0.75658, 50 -> 0.75636, 100 -> 0.75615 neighbours (~20 s per fit)
TASK1_ITEM_KNN = ItemKnnConfig(
    n_neighbors=100,
    similarity_shrinkage=100.0,             # Koren 2010 uses ~100 for Netflix
    prediction_shrinkage=1.0,
    baseline_biases=TASK1_SHARED_BASELINE_BIASES,   # = the bias model's baseline (blend: kNN - bias model)
    max_ratings_per_user_for_similarity=300,  # caps the 899 most active users -> 1.3e8 instead of 4.3e8 item pairs
    items_per_similarity_block=2000,
    seed=0,
)

# ---- zero model: P(rating = 0) -----------------------------------------------------------
TASK1_ZERO_MODEL = ZeroModelConfig(
    user_prior_strength=5.0,
    item_prior_strength=20.0,
    inverse_regularization=1.0,
    max_iterations=1000,
)

# ---- factorization machine (user, item, activity buckets) -------------------------------
# starting point = the SVD settings above; no time data (the Task 1 test file has no timestamps -> load_task1 keeps none)
# sweep 14 Nov, learning rate x batch: 5e-4 / 256 -> 0.75149 (2.5e-4 / 256 0.75275, 1e-3 / 256 0.75495;
# batch 512: 2.5e-4 0.75527, 5e-4 0.75209, 1e-3 0.75462)
TASK1_FACTORIZATION_MACHINE = FactorizationMachineConfig(
    n_factors=120,
    n_activity_buckets=10,
    learning_rate=5e-4,
    bias_regularization=3.0,
    factor_regularization=8.0,
    ratings_per_batch=256,
    max_epochs=1000,
    early_stopping_patience=6,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- Bayesian factorization machine (Gibbs sampling, CPU, needs `pip install myfm`) -------
# no learning rate / lambda / early stopping to tune; more kept draws = a smoother posterior mean
# CPU check 14 Nov (rank 32, 20 + 80 draws, no refit): valid MSE 0.7255 (SVD++ k = 32 in the same check: 0.7506)
# time per draw (CPU check): rank 32 ~6 s, rank 64 ~12 s -> rank 32 to start, the sweep tries larger ranks
# sweep 14 Nov (20 + 180 draws): rank 32 0.7272, 64 0.72337, 128 0.72022 (~35 min per fit on the PC)
# 15 Nov without the 0-star ratings (MSE on the validation ratings > 0): 20 + 180 draws 0.54871, 30 + 380 draws 0.55708 (worse);
# + every known pair as implicit feedback 0.54395 (jobs/cpu_07; Task 1 blend 0.54058 -> 0.53768), at rank 192 0.54084
# (blend 0.53601; ~2 h per fit on the PC); 16 Nov 2 chains (seeds 0 + 1 averaged, jobs/cpu_08) 0.53796 (blend 0.53391)
TASK1_BAYESIAN_FACTORIZATION_MACHINE = BayesianFactorizationMachineConfig(
    rank=192,
    n_burn_in_iterations=20,
    n_kept_samples=180,
    use_implicit_feedback=True,
    use_test_pairs_as_implicit_feedback=True,
    refit_on_train_and_valid=True,
    seed=0,
    n_chains=2,
)

# ---- rating transformer (attention over the user's rated books) --------------------------
# starting point (not tuned): d = 64, 2 blocks, at most 200 books per user history
# CPU check 14 Nov (d 32, 1 block, L 50, no refit): 0.7647 (epoch 0 = bias model) -> 0.7595 at epoch 3, still falling slowly
TASK1_RATING_TRANSFORMER = RatingTransformerConfig(
    embedding_dimension=64,
    n_blocks=2,
    n_heads=2,
    dropout=0.1,
    max_history_length=200,
    use_test_pairs_in_history=False,
    learning_rate=1e-3,
    bias_regularization=3.0,
    factor_regularization=1.0,
    ratings_per_batch=1024,
    max_epochs=1000,
    early_stopping_patience=3,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- I-AutoRec (item-based autoencoder) ----------------------------------------------------
# hidden 500 as in Sedhain et al. 2015; models the residuals of the closed-form bias model, masked training
# CPU checks 14 Nov (hidden 200, no refit): learning rate 1e-3 -> worse than the bias model from epoch 1 on (lambda 1 and 100);
# 1e-4 with lambda 16 (~ the factor penalty of SVD) -> valid MSE 0.7506 at epoch 15 (SVD++ k = 32 in the same check: 0.7506)
# sweep 14 Nov: learning rate 3e-5 0.74950 (1e-4: 0.75110; 1.5e-5: 0.74898, same), hidden 250 + masked 25 % 0.74900
TASK1_AUTOREC = AutoRecConfig(
    hidden_dimension=250,
    regularization=16.0,
    user_bias_regularization=1.0,           # own lambdas (independent of TASK1_SHARED_BASELINE_BIASES)
    item_bias_regularization=5.0,
    masked_fraction=0.25,
    learning_rate=3e-5,
    books_per_batch=256,
    max_epochs=1000,
    early_stopping_patience=5,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- integrated model (SVD++ + learned neighbour weights, Koren 2008) ----------------------
# starting point = the tuned TASK1_SVDPP (k, learning rate, lambdas, batch) + the neighbourhoods of TASK1_ITEM_KNN
# CPU check 14 Nov (k = 32, lr 2e-3, batch 16384, 50 neighbours, no refit): valid MSE 0.7366 at epoch 3
# (SVD++ with the same k / learning rate / batch: 0.7506)
# neighbour lookups (CPU, 100 neighbours, measured 14 Nov): ~6 ms per batch of 512 -> ~20 s per epoch
# sweep 14 Nov: neighbour penalty 0.1 (1.0: +0.0003), factor penalty 24 (8: 0.7362, 16: 0.7331, 32: 0.7337),
# factors at penalty 24: 120 0.73246, 250 0.73267, 500 0.73290 -> 120 (fastest)
TASK1_INTEGRATED_MODEL = IntegratedModelConfig(
    n_factors=120,
    use_test_pairs_as_implicit_feedback=True,     # every known pair, 0-star ratings included (assignment1.rating_model_task)
                                                  # 15 Nov: 0.56463 -> 0.56343; Task 1 blend 0.54116 -> 0.54058 with these two
    n_neighbors=100,
    similarity_shrinkage=100.0,
    max_ratings_per_user_for_similarity=300,
    items_per_similarity_block=2000,
    learning_rate=2.5e-4,
    user_bias_regularization=1.0,
    item_bias_regularization=5.0,
    factor_regularization=24.0,
    neighbour_regularization=0.1,
    ratings_per_batch=512,
    max_epochs=1000,
    early_stopping_patience=6,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- blenders ----------------------------------------------------------------------------
# which sub-models the blend combines: True = in the blend, False = left out (still trained and stored -> can be switched back on)
# ensemble sweep 14 Nov (sweep/task1_ensemble_sweep.ipynb, all ten models trained on the GPU), out-of-fold MSE of the ridge:
# blend of all ten 0.71118 (before the newer models: 0.7302); dropping the zero model, FM, AutoRec and SVD costs 0.00016 -> 0.71134
# (without them: Bayesian FM +0.0137, bias model + item kNN +0.0037, integrated model +0.0026, SVD++ +0.0013, transformer +0.0009)
# 14 Nov night: all ten switched on again (every sub-model in the blend for now; the ensemble sweep decides later)
TASK1_BLEND_MODEL_FLAGS = {
    "bias_model": True,
    "svd": True,
    "svdpp": True,
    "item_knn": True,                       # bias model and item kNN only together: the blend uses their difference
    "zero_model": True,
    "factorization_machine": True,
    "bayesian_factorization_machine": True,
    "rating_transformer": True,
    "autorec": True,
    "integrated_model": True,
}
TASK1_BLENDED_MODELS = tuple(model_name for model_name, is_blended in TASK1_BLEND_MODEL_FLAGS.items() if is_blended)

# learned weighted average of the sub-model predictions
TASK1_RIDGE_BLENDER = BlenderConfig(
    method="ridge",
    blended_models=TASK1_BLENDED_MODELS,
    use_count_features=False,
    n_folds=5,
    ridge_alpha=1.0,
    lightgbm_n_estimators=0,                # not used by ridge
    lightgbm_learning_rate=0.0,
    lightgbm_num_leaves=0,
    lightgbm_min_child_samples=0,
    seed=0,
    fit_on_nonzero_ratings=True,            # the leaderboard leaves out ratings of 0 (lecture 6 Nov; public Task 1 score 0.5618 ~ our 0.566 without zeros)
)

# gradient-boosted trees on the sub-model predictions + #ratings of user and item (as the Netflix Prize blends)
TASK1_LIGHTGBM_BLENDER = BlenderConfig(
    method="lightgbm",
    blended_models=TASK1_BLENDED_MODELS,
    use_count_features=True,
    n_folds=5,
    ridge_alpha=0.0,                        # not used by LightGBM
    lightgbm_n_estimators=400,
    lightgbm_learning_rate=0.05,
    lightgbm_num_leaves=31,
    lightgbm_min_child_samples=200,
    seed=0,
    fit_on_nonzero_ratings=True,            # the leaderboard leaves out ratings of 0 (lecture 6 Nov; public Task 1 score 0.5618 ~ our 0.566 without zeros)
)

# the blend that writes output/task_1_predictions.csv
TASK1_SUBMITTED_BLENDER = TASK1_RIDGE_BLENDER


# #########################################################################################
# Task 2 - interaction prediction: which of two books did the user read?
# test negatives = the book of a random training interaction (in proportion to popularity) -> validation pairs built the
# same way (lib/data_loading.py); sub-models score (user, book), a LightGBM pair blender decides
# #########################################################################################

# ---- item cosine similarity to the user's history ----------------------------------------
# sweep 14 Nov (pair accuracy, mean of 3 validation splits): 1600 neighbours, no shrinkage, 1000 books per user 0.8690
# (100 / 20 / 300: 0.8328; 2400 neighbours equal, 3200 worse)
TASK2_ITEM_COSINE = ItemCosineConfig(
    n_neighbors=1600,
    similarity_shrinkage=0.0,
    max_interactions_per_user_for_similarity=1000,
    items_per_similarity_block=2000,
    seed=0,
)

# ---- EASE (linear autoencoder) -----------------------------------------------------------
TASK2_EASE = EaseConfig(
    regularization=1000.0,                  # sweep 14 Nov: 250 0.842, 500 0.849, 750 0.8508, 1000 0.8512, 2000 0.846
    max_items=60_000,                       # all ~51k books (the 25k most popular cover only 82 % of the interactions)
    use_gpu=False,                          # 51k x 51k float32 = 10.4 GB per matrix, ~21 GB peak -> in RAM, a few minutes per fit
    history_time_decay_days=0.0,            # plain EASE (the Task 2 test file has no timestamp)
    history_time_decay_floor=0.0,
    items_per_gram_block=2000,
    pairs_per_scoring_chunk=20_000,
)

# ---- Mult-VAE (non-linear autoencoder) ----------------------------------------------------
# starting point: Liang et al. 2018 (600 -> 200, dropout 0.5, beta up to 0.2): 0.8562
# sweep 14 Nov: dropout 0.1, beta 0.5, hidden 1000 0.8626 (latent 400 / lr 5e-4 or 2e-3 / annealing 10 or 40 not better)
TASK2_MULT_VAE = MultVaeConfig(
    hidden_dimension=1000,
    latent_dimension=200,
    input_dropout=0.1,
    kl_weight_max=0.5,
    kl_annealing_epochs=20,
    learning_rate=1e-3,
    users_per_batch=500,
    max_epochs=200,
    early_stopping_patience=10,
    max_items=60_000,                       # all books (input / output layer 51k x 600 -> ~60M weights, fine on the GPU)
    seed=0,
)

# ---- pair blender --------------------------------------------------------------------------
# validation negatives per held-out interaction (> 1: more pairs for the blender, scored on the original pairs; as Task 5)
# 15 Nov (jobs/gpu_10, accuracy on the original pairs): 1 -> 4 negatives 0.94025 -> 0.94091
TASK2_NEGATIVES_PER_VALIDATION_PAIR = 4

# which sub-models the blend combines: True = in the blend, False = left out (still trained and stored -> can be switched back on)
TASK2_PAIR_BLEND_MODEL_FLAGS = {
    "popularity": True,
    "item_cosine": True,
    "ease": True,
    "mult_vae": True,
}
# 15 Nov (jobs/gpu_10, stored sub-models, 1 negative): 0.93689 -> user span 0.93800, reading window 0.93957, both 0.94005,
# + bigger LightGBM (as Task 4) 0.94025
TASK2_PAIR_BLENDER = PairBlenderConfig(
    combined_models=tuple(model_name for model_name, is_blended in TASK2_PAIR_BLEND_MODEL_FLAGS.items() if is_blended),
    use_time_features=False,                # the Task 2 test file has no timestamp
    activity_window_days=30,                # not used without time features
    n_folds=5,
    lightgbm_n_estimators=800,
    lightgbm_learning_rate=0.03,
    lightgbm_num_leaves=63,
    lightgbm_min_child_samples=100,
    seed=0,
    use_user_span_features=True,            # the training file has timestamps: item's reads vs. the user's active span
    use_reading_window_features=True,       # how often others read the item within 5 books of the user's books
)


# #########################################################################################
# Task 3 - rating prediction, timestamp available
# the Task 1 models (timestamps ignored) + temporal models, blended together with time features
# Task 3 data: the test rows lie inside the training period (interpolation, not forecasting);
# bias-model residuals of a user's consecutive ratings are correlated 0.12 on the same day, 0.06 for 1 - 7 days apart,
# 0.04 for 1 - 4 weeks, ~0 after half a year; 1.7 % of the ratings are 0, 45 % if the user gave another 0 that day
# #########################################################################################

# ---- shared baseline mu + b_u + b_i of the interlinked models ----------------------------
# as in Task 1: the bias model, the item kNN and the time-decayed item kNN measure everything against the same static
# baseline; the blend combines them (ridge weights bias model -2.31 / item kNN +0.72 / time item kNN +1.55)
# -> defined once here, used by TASK3_BIAS_MODEL, TASK3_ITEM_KNN and TASK3_TIME_ITEM_KNN; after a change retrain the three + the blend
# not linked: SVD / SVD++ (own lambdas, trained further) and the time-aware bias model / timeSVD++ (own b_u, b_i fitted
# together with the temporal terms)
# closed-form grid on the Task 3 validation split: 3 / 3 -> 0.91382, 3 / 5 -> 0.91354 (empirical Bayes: users ~3.3, items ~5.6)
TASK3_SHARED_BASELINE_BIASES = BiasModelConfig(
    user_bias_regularization=3.0,
    item_bias_regularization=5.0,
    n_iterations=10,
)

# ---- bias model: mu + b_u + b_i in closed form ------------------------------------------
TASK3_BIAS_MODEL = TASK3_SHARED_BASELINE_BIASES

# ---- SVD -------------------------------------------------------------------------------
TASK3_SVD = LatentFactorConfig(
    use_implicit_feedback=False,
    use_test_pairs_as_implicit_feedback=False,
    n_factors=500,
    learning_rate=2.5e-4,
    user_bias_regularization=3.0,           # own lambdas (independent of TASK3_SHARED_BASELINE_BIASES)
    item_bias_regularization=5.0,
    factor_regularization=8.0,
    ratings_per_batch=512,
    max_epochs=1000,
    early_stopping_patience=6,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- SVD++ -----------------------------------------------------------------------------
# SVD++ rebuilds N(u) sums for every user in a batch -> bigger batches keep an epoch affordable
# sweep 14 Nov: factor penalty 4, learning rate 1e-3 0.88416 (8 / 2e-3: 0.88550)
TASK3_SVDPP = LatentFactorConfig(
    use_implicit_feedback=True,
    use_test_pairs_as_implicit_feedback=True,     # every known pair, 0-star ratings included (assignment1.rating_model_task)
                                                  # 15 Nov: 0.71092 -> 0.71022
    n_factors=500,
    learning_rate=1e-3,
    user_bias_regularization=3.0,           # own lambdas (independent of TASK3_SHARED_BASELINE_BIASES)
    item_bias_regularization=5.0,
    factor_regularization=4.0,
    ratings_per_batch=16384,
    max_epochs=1000,
    early_stopping_patience=6,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- item-item kNN on the bias model's residuals ----------------------------------------
# sweep 14 Nov: 200 neighbours 0.89924 (50: 0.89941, 100: 0.89931)
TASK3_ITEM_KNN = ItemKnnConfig(
    n_neighbors=200,
    similarity_shrinkage=100.0,
    prediction_shrinkage=1.0,
    baseline_biases=TASK3_SHARED_BASELINE_BIASES,   # = the bias model's baseline (blend: kNN - bias model)
    max_ratings_per_user_for_similarity=300,
    items_per_similarity_block=2000,
    seed=0,
)

# ---- zero model: P(rating = 0) -----------------------------------------------------------
TASK3_ZERO_MODEL = ZeroModelConfig(
    user_prior_strength=5.0,                # Task 1 values
    item_prior_strength=20.0,
    inverse_regularization=1.0,
    max_iterations=1000,
)

# ---- temporal bias terms (shared by the time-aware bias model and timeSVD++) -------------
# quick grid on the validation split (closed form): static bias model 0.9138 -> 0.8829 with these values
# (no user drift 0.8958 / drift regularization 10: 0.9028 -> the drift needs strong shrinkage; 10 / 60 bins: same as 30)
TASK3_TEMPORAL_BIASES = TemporalBiasConfig(
    n_time_bins=30,                         # Koren 2009 uses 30 bins on Netflix
    user_period_lengths_in_days=(1, 7, 30, 90, 365, 1000),  # test rows whose user has a training rating in the same day / week / month: 10 / 43 / 76 %
                                            # sweep 14 Nov: + 1000 days 0.88238 (without: 0.88291; 3-split mean 0.88651)
    drift_exponent=0.4,                     # Koren 2009 (PML Eq. 7.11); 0.2 same, 0.6 worse
    time_bias_regularization=50.0,          # mostly 1 - 2 ratings per (user, period) -> strong shrinkage (10: 0.9153, 25: 0.9016 w/o drift tuning)
    drift_regularization=3000.0,            # sum of dev_u(t)^2 over a user's ratings is in the thousands -> ridge needs a large value
)

# ---- time-aware bias model (closed form) -------------------------------------------------
TASK3_TIME_BIAS_MODEL = TimeBiasModelConfig(
    temporal_biases=TASK3_TEMPORAL_BIASES,
    bias_regularization=3.0,                # own lambda for b_u, b_i, b_bin (not linked to TASK3_SHARED_BASELINE_BIASES)
    n_iterations=10,
)

# ---- timeSVD++ ---------------------------------------------------------------------------
# = TASK3_SVDPP on top of the closed-form time-aware bias model (epoch 0 = TASK3_TIME_BIAS_MODEL)
TASK3_TIME_SVDPP = TimeSVDppConfig(
    use_implicit_feedback=True,
    use_test_pairs_as_implicit_feedback=True,     # every known pair, 0-star ratings included (assignment1.rating_model_task)
                                                  # 15 Nov: 0.69664 -> 0.69597
    n_factors=500,
    use_factor_drift=False,                 # PML Sec. 7.2.3: biases first; True adds drifting user factors p_u(t)
    time_biases=TASK3_TIME_BIAS_MODEL,
    train_temporal_biases=False,            # True: Adam trains them further (overfits them within ~1 epoch)
    learning_rate=1e-3,                     # sweep 14 Nov: penalty 4 / 1e-3 + the 1000-day period 0.85808 (8 / 2e-3: 0.86050;
    bias_regularization=3.0,                # 1 - 2 or 1000 factors not better; drifting user factors 0.877)
    factor_regularization=4.0,
    ratings_per_batch=16384,
    max_epochs=1000,
    early_stopping_patience=6,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- time-decayed item-item kNN ----------------------------------------------------------
TASK3_TIME_ITEM_KNN = TimeItemKnnConfig(
    n_neighbors=100,
    similarity_shrinkage=100.0,
    prediction_shrinkage=1.0,
    baseline_biases=TASK3_SHARED_BASELINE_BIASES,   # = the bias model's baseline (blend: kNN - bias model)
    max_ratings_per_user_for_similarity=300,
    items_per_similarity_block=2000,
    seed=0,
    # quick check on the validation split: no decay setting beats the plain kNN on its own (0.8997);
    # 365 days / floor 0.5: 0.9006 -> kept as a differently weighted neighbourhood for the blend
    # sweep 14 Nov: 3000 days / floor 0.7 / 100 neighbours 0.89938 (365 / 0.5 / 50: 0.90025; slower decays -> the plain kNN)
    time_decay_days=3000.0,
    time_decay_floor=0.7,
)

# ---- temporal zero model -----------------------------------------------------------------
TASK3_TIME_ZERO_MODEL = TimeZeroModelConfig(
    user_prior_strength=5.0,
    item_prior_strength=20.0,
    inverse_regularization=1.0,
    max_iterations=1000,
    window_lengths_in_days=(0, 7, 30, 365, 1000),   # same day + week / month / year / ~3 years around the rating
    window_prior_strength=5.0,              # sweep 14 Nov: log loss 0.05305 (windows 0 / 30, prior 2: 0.05382)
)

# ---- time-aware factorization machine ----------------------------------------------------
# = TASK1_FACTORIZATION_MACHINE settings (starting point, not tuned on Task 3) + one-hot calendar fields of the timestamp
# hour_of_day: all timestamps lie in 05:00 - 09:59 UTC (dates + an offset) -> little real signal
# sweep 14 Nov, calendar fields: all four 0.89674, weekday + month + year 0.89555, month + year 0.89413, year 0.89543
# sweep 14 Nov evening: batch 256, 60 factors, factor penalty 32, bias penalty 5 0.89303 (before: 0.89411)
TASK3_TIME_FACTORIZATION_MACHINE = TimeFactorizationMachineConfig(
    n_factors=60,
    n_activity_buckets=10,
    learning_rate=2.5e-4,
    bias_regularization=5.0,
    factor_regularization=32.0,
    ratings_per_batch=256,
    max_epochs=1000,
    early_stopping_patience=6,
    refit_on_train_and_valid=True,
    seed=0,
    calendar_fields=("month", "year"),
)

# ---- the Task 1 models added 14 Nov: starting points = the Task 1 sweep winners, then the Task 3 sweep (14 Nov evening) --
# ---- time-aware Bayesian factorization machine (Gibbs sampling with myFM; CPU, one core, ~5 s per draw) -----------
# = TASK1_BAYESIAN_FACTORIZATION_MACHINE (the best single Task 1 model) + the calendar fields of the time-aware FM above
# quick check 14 Nov (CPU, only 3 + 5 draws, no refit): without calendar fields 0.9109, month + year 0.8905
# sweep 14 Nov (rank 32, 20 + 180 draws): month + year 0.84921, weekday + month + year 0.84878, all four 0.84988;
# rank 16 0.85255, 64 0.84752, 128 0.84666; rank 64 with 30 + 380 draws 0.84445 (~35 min per fit on the PC)
# 15 Nov without the 0-star ratings (MSE on the validation ratings > 0): 0.67859; rank 128 0.67760; user periods 365 / 1000
# days + every known pair 0.67650 (jobs/cpu_06; quick check rank 8: user day / 30 days / book periods no gain);
# 16 Nov 2 chains (seeds 0 + 1 averaged, jobs/cpu_08) 0.67464 (blend 0.66773 -> 0.66712)
# (rank 128: 1 chain 0.67651, 2 chains 0.67461 but blend 0.66738 -> kept rank 64)
TASK3_TIME_BAYESIAN_FACTORIZATION_MACHINE = BayesianFactorizationMachineConfig(
    rank=64,
    n_burn_in_iterations=30,
    n_kept_samples=380,
    use_implicit_feedback=True,
    use_test_pairs_as_implicit_feedback=True,
    refit_on_train_and_valid=True,
    seed=0,
    calendar_fields=("weekday", "month", "year"),
    user_period_days=(365, 1000),
    n_chains=2,
)

# ---- rating transformer (attention over the user's rated books; timestamps not used) ------------------------------
# = TASK1_RATING_TRANSFORMER (Task 1 sweep 14 Nov: learning rate, dropout, width, blocks, history length all within ~0.001)
TASK3_RATING_TRANSFORMER = RatingTransformerConfig(
    embedding_dimension=64,
    n_blocks=2,
    n_heads=2,
    dropout=0.1,
    max_history_length=200,
    use_test_pairs_in_history=False,
    learning_rate=1e-3,
    bias_regularization=3.0,
    factor_regularization=1.0,
    ratings_per_batch=1024,
    max_epochs=1000,
    early_stopping_patience=3,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- I-AutoRec (item-based autoencoder on the bias-model residuals; timestamps not used) --------------------------
# Task 1 sweep 14 Nov: learning rate 3e-5 (0.7495 vs 1e-4: 0.7511), hidden 250 / masked 25 % 0.7490
# Task 3 sweep 14 Nov: hidden 500 0.89382 (250: 0.89461, 1000: 0.89399; learning rate 1e-4 worse)
TASK3_AUTOREC = AutoRecConfig(
    hidden_dimension=500,
    regularization=16.0,
    user_bias_regularization=1.0,
    item_bias_regularization=5.0,
    masked_fraction=0.25,
    learning_rate=3e-5,
    books_per_batch=256,
    max_epochs=1000,
    early_stopping_patience=5,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- integrated model (SVD++ + learned neighbour weights, Koren 2008; timestamps not used) ------------------------
# Task 1 sweep 14 Nov: 250 factors, factor penalty 24, neighbour penalty 0.1 -> 0.7327 (config 500 / 8 / 1.0: 0.7365)
# Task 3 sweep 14 Nov: 0.87506 (factor penalty 16 0.87539, 32 0.87531, 48 worse; 200 neighbours 0.87734)
TASK3_INTEGRATED_MODEL = IntegratedModelConfig(
    n_factors=250,
    use_test_pairs_as_implicit_feedback=True,     # every known pair, 0-star ratings included (assignment1.rating_model_task)
                                                  # 15 Nov: 0.70264 -> 0.70073; Task 3 blend 0.66804 -> 0.66773 with the three
    n_neighbors=100,
    similarity_shrinkage=100.0,
    max_ratings_per_user_for_similarity=300,
    items_per_similarity_block=2000,
    learning_rate=2.5e-4,
    user_bias_regularization=1.0,
    item_bias_regularization=5.0,
    factor_regularization=24.0,
    neighbour_regularization=0.1,
    ratings_per_batch=512,
    max_epochs=1000,
    early_stopping_patience=6,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- blenders ----------------------------------------------------------------------------
# which sub-models the blend combines: True = in the blend, False = left out (still trained and stored -> can be switched back on)
# (only together: bias model + item kNN + time-decayed kNN share one baseline; zero model + time zero model enter as a pair)
TASK3_BLEND_MODEL_FLAGS = {
    "bias_model": True,
    "svd": True,
    "svdpp": True,
    "item_knn": True,
    "zero_model": True,
    "time_bias_model": True,
    "time_svdpp": True,
    "time_item_knn": True,
    "time_zero_model": True,
    "time_factorization_machine": True,
    "time_bayesian_factorization_machine": True,
    "rating_transformer": True,
    "autorec": True,
    "integrated_model": True,
}
TASK3_BLENDED_MODELS = tuple(model_name for model_name, is_blended in TASK3_BLEND_MODEL_FLAGS.items() if is_blended)

# learned weighted average of the sub-model predictions
TASK3_RIDGE_BLENDER = BlenderConfig(
    method="ridge",
    blended_models=TASK3_BLENDED_MODELS,
    use_count_features=False,
    n_folds=5,
    ridge_alpha=1.0,
    lightgbm_n_estimators=0,                # not used by ridge
    lightgbm_learning_rate=0.0,
    lightgbm_num_leaves=0,
    lightgbm_min_child_samples=0,
    seed=0,
    use_time_features=False,                # a linear blend cannot use "when" features
    fit_on_nonzero_ratings=True,            # the leaderboard leaves out ratings of 0 (lecture 6 Nov; public Task 1 score 0.5618 ~ our 0.566 without zeros)
)

# gradient-boosted trees on the sub-model predictions + #ratings of user and item + when the rating happened
TASK3_LIGHTGBM_BLENDER = BlenderConfig(
    method="lightgbm",
    blended_models=TASK3_BLENDED_MODELS,
    use_count_features=True,
    n_folds=5,
    ridge_alpha=0.0,                        # not used by LightGBM
    lightgbm_n_estimators=400,
    lightgbm_learning_rate=0.05,
    lightgbm_num_leaves=31,
    lightgbm_min_child_samples=200,
    seed=0,
    use_time_features=True,
    activity_window_days=30,
    fit_on_nonzero_ratings=True,            # the leaderboard leaves out ratings of 0 (lecture 6 Nov; public Task 1 score 0.5618 ~ our 0.566 without zeros)
)

# the blend that writes output/task_3_predictions.csv
# smoke run (CPU, shortened training): ridge 0.8515 vs LightGBM 0.8554 out-of-fold -> compare again after the full GPU run
TASK3_SUBMITTED_BLENDER = TASK3_RIDGE_BLENDER


# #########################################################################################
# Task 4 - interaction prediction, timestamp available
# same sub-models as Task 2; the pair blender also sees the book's popularity around t and its age at t
# (test negatives are drawn regardless of time -> 25 % of the pairs contain a book first read only after t)
# #########################################################################################

# ---- item cosine similarity to the user's history ----------------------------------------
# sweep 14 Nov (mean of 3 validation splits): 20 neighbours, shrinkage 5, 2000 books per user 0.7233 (100 / 20 / 300: 0.6769)
TASK4_ITEM_COSINE = ItemCosineConfig(
    n_neighbors=20,
    similarity_shrinkage=5.0,
    max_interactions_per_user_for_similarity=2000,
    items_per_similarity_block=2000,
    seed=0,
)

# ---- EASE (linear autoencoder) -----------------------------------------------------------
TASK4_EASE = EaseConfig(
    regularization=500.0,
    max_items=60_000,                       # Task 4 has ~17k books -> all of them get weights (1.2 GB, on the GPU)
    use_gpu=True,
    history_time_decay_days=0.0,            # plain EASE
    history_time_decay_floor=0.0,
    items_per_gram_block=2000,
    pairs_per_scoring_chunk=20_000,
)

# ---- EASE with a time-weighted history ----------------------------------------------------
# same weights B; the user's books read close to the asked day count more
# quick check on the validation pairs: EASE alone 0.751 -> 0.787, blend 0.915 -> 0.919
# (decay 30 / 60 / 365 days or a floor of 0.1 - 0.3: alone 0.746 - 0.781, blend within 0.001)
TASK4_TIME_EASE = EaseConfig(
    regularization=500.0,
    max_items=60_000,
    use_gpu=True,
    history_time_decay_days=180.0,
    history_time_decay_floor=0.0,
    items_per_gram_block=2000,
    pairs_per_scoring_chunk=20_000,
)

# ---- Mult-VAE (non-linear autoencoder) ----------------------------------------------------
# sweep 14 Nov: hidden 1500, dropout 0.1, beta 0.1 0.7592 (600 / 0.5 / 0.2: 0.7476)
TASK4_MULT_VAE = MultVaeConfig(
    hidden_dimension=1500,
    latent_dimension=200,
    input_dropout=0.1,
    kl_weight_max=0.1,
    kl_annealing_epochs=20,
    learning_rate=1e-3,
    users_per_batch=500,
    max_epochs=200,
    early_stopping_patience=10,
    max_items=60_000,
    seed=0,
)

# ---- pair blender --------------------------------------------------------------------------
# which sub-models the blend combines: True = in the blend, False = left out (still trained and stored -> can be switched back on)
# validation negatives per held-out interaction (as Task 2); 15 Nov (jobs/gpu_11): 1 -> 4 negatives 0.93287 -> 0.93408
TASK4_NEGATIVES_PER_VALIDATION_PAIR = 4

TASK4_PAIR_BLEND_MODEL_FLAGS = {
    "popularity": True,
    "item_cosine": True,
    "ease": True,
    "time_ease": True,
    "mult_vae": True,
}
TASK4_PAIR_BLENDER = PairBlenderConfig(
    combined_models=tuple(model_name for model_name, is_blended in TASK4_PAIR_BLEND_MODEL_FLAGS.items() if is_blended),
    use_time_features=True,
    activity_window_days=30,                # "popularity around t" = interactions with the book within +- 30 days
    n_folds=5,
    lightgbm_n_estimators=800,             # 15 Nov grid (with reader timing): 63 leaves / min 100 / 800 trees / 0.03 0.93254 (31 / 200 / 400 / 0.05: 0.93164)
    lightgbm_learning_rate=0.03,
    lightgbm_num_leaves=63,
    lightgbm_min_child_samples=100,
    seed=0,
    use_reader_timing_features=True,       # 15 Nov: out-of-fold accuracy 0.93077 -> 0.93164 (readers after t, first reader after t, ...)
    use_user_span_features=True,           # 15 Nov (jobs/gpu_11, with the reading window): 0.93254 -> 0.93287 (window alone 0.93233)
    use_reading_window_features=True,
)


# #########################################################################################
# Task 5 - which book comes next? (one test pair per user: the book after the last training interaction)
# leave-last-out validation pairs (lib/data_loading.load_next_item_task); 87k users, 232k books, ~12 interactions per user
# Task 4 sub-models (EASE / Mult-VAE on the most popular books) + transition model + SASRec -> pair blender
# #########################################################################################

# ---- item cosine similarity to the user's history ----------------------------------------
# sweep 14 Nov (mean of 3 validation splits): 3200 neighbours, no shrinkage, 1000 books per user 0.7717 (100 / 20 / 300: 0.7228;
# 1600 0.7595, 2400 0.7683, 4800 0.7705, 6400 0.7667)
TASK5_ITEM_COSINE = ItemCosineConfig(
    n_neighbors=3200,
    similarity_shrinkage=0.0,
    max_interactions_per_user_for_similarity=1000,
    items_per_similarity_block=2000,
    seed=0,
)

# ---- transition model (which book follows which) ----------------------------------------
# sweep 14 Nov (mean of 3 validation splits): 0.7460 (3 steps / 0.5 / 5 / 5 books / 0.5: 0.7019; 200 steps: same)
TASK5_TRANSITION = TransitionConfig(
    steps_ahead=100,                        # j -> i also when i comes up to 100 books later (weights 1, 0.95, 0.95^2, ...)
    step_decay=0.95,
    shrinkage=1.0,
    context_length=20,                      # the user's last 20 books vote, weights 1, 0.8, 0.64, ...
    context_decay=0.8,
)

# ---- EASE + time-weighted EASE (one set of weights, in RAM) -------------------------------
# sweep 14 Nov: regularization 2000 0.7009 (plain; 500: 0.6885), time decay 730 days 0.7058 (1000 days: same)
TASK5_EASE = EaseConfig(
    regularization=2000.0,
    max_items=50_000,                       # of 232k books: covers 85 % of the interactions; 10 GB per matrix, ~20 GB peak
    use_gpu=False,
    history_time_decay_days=730.0,          # time-weighted variant: books read long before "now" (the last interaction) count less
    history_time_decay_floor=0.0,
    items_per_gram_block=2000,
    pairs_per_scoring_chunk=20_000,
)

# ---- Mult-VAE (non-linear autoencoder) ----------------------------------------------------
# sweep 14 Nov: hidden 1000, dropout 0.1, beta 0.1 0.7189 (600 / 0.5 / 0.2: 0.7104)
TASK5_MULT_VAE = MultVaeConfig(
    hidden_dimension=1000,
    latent_dimension=200,
    input_dropout=0.1,
    kl_weight_max=0.1,
    kl_annealing_epochs=20,
    learning_rate=1e-3,
    users_per_batch=500,
    max_epochs=200,
    early_stopping_patience=10,
    max_items=50_000,                       # the books EASE covers too
    seed=0,
)

# ---- SASRec (self-attention over the sequence) ---------------------------------------------
# Kang & McAuley 2018: 2 blocks, 1 head, d = 50, L = 50 and dropout 0.5 on their sparse datasets, Adam 1e-3
# sweep 14 Nov: d 256, learning rate 1.25e-4 0.8921 (64 / 1e-3: 0.8854; 128 / 5e-4 0.8885; 512 / 2.5e-4 0.8918)
TASK5_SASREC = SasRecConfig(
    hidden_dimension=256,
    n_blocks=2,
    n_heads=1,
    max_sequence_length=50,                 # 90 % of the users have fewer interactions
    dropout=0.5,
    learning_rate=1.25e-4,
    users_per_batch=256,
    max_epochs=200,
    early_stopping_patience=10,
    seed=0,
)

# ---- validation pairs: negatives per validation user (1 = as the test; > 1 = extra pairs for the blender to learn from,
# lib/data_loading.py load_next_item_task)
# 15 Nov (jobs/gpu_06, accuracy on the original 1-negative pairs): 1 -> 4 negatives 0.92502 -> 0.92688 (without the seed-1
# SASRec), + the 200-book SASRec 0.92749
TASK5_NEGATIVES_PER_VALIDATION_USER = 4

# ---- more SASRec runs for the blend (15 Nov): SASRec is the blend's strongest input -> a second seed and a longer history
TASK5_SASREC_SEED_1 = SasRecConfig(**{**TASK5_SASREC.__dict__, "seed": 1})
TASK5_SASREC_LONG_HISTORY = SasRecConfig(**{**TASK5_SASREC.__dict__, "max_sequence_length": 100, "seed": 2})
TASK5_SASREC_LONGER_HISTORY = SasRecConfig(**{**TASK5_SASREC.__dict__, "max_sequence_length": 200, "seed": 3})
# 15 Nov on their own: seed 1 0.89252 (seed 0: ~0.892), 100-book history 0.89935; blend of 9: 0.92462 (7 models: 0.92332;
# without the seed-1 run 0.92461, without the 100-book run 0.92405)

# ---- pair blender --------------------------------------------------------------------------
# which sub-models the blend combines: True = in the blend, False = left out (still trained and stored -> can be switched back on)
TASK5_PAIR_BLEND_MODEL_FLAGS = {
    "popularity": True,
    "item_cosine": True,
    "transition": True,
    "ease": True,
    "time_ease": True,
    "mult_vae": True,
    "sasrec": True,
    "sasrec_seed_1": False,                 # not retrained in the 4-negatives run (the 200-book run helps more)
    "sasrec_long_history": True,
    "sasrec_longer_history": True,          # 200-book history: 0.90261 on its own (100 books 0.89909, 50 books 0.89161)
}
TASK5_PAIR_BLENDER = PairBlenderConfig(
    combined_models=tuple(model_name for model_name, is_blended in TASK5_PAIR_BLEND_MODEL_FLAGS.items() if is_blended),
    use_time_features=True,                 # t = the day of the user's last interaction ("now")
    activity_window_days=30,
    n_folds=5,
    lightgbm_n_estimators=800,             # 15 Nov grid (with reader timing): 63 leaves / min 100 / 800 trees / 0.03 0.92332 (31 / 200 / 400 / 0.05: 0.92258)
    lightgbm_learning_rate=0.03,
    lightgbm_num_leaves=63,
    lightgbm_min_child_samples=100,
    seed=0,
    use_reader_timing_features=True,       # 15 Nov: out-of-fold accuracy 0.92149 -> 0.92258 (readers after t, first reader after t, ...)
    use_sequence_features=True,            # 15 Nov: what other readers read right after the user's last books: 0.92462 -> 0.92502
    use_user_span_features=True,           # 16 Nov (jobs/gpu_12, with the reading window, 4 negatives): 0.92749 -> 0.92804
    use_reading_window_features=True,      # (window alone 0.92745)
)


# #########################################################################################
# Task 6 - time until the user's next interaction (the next book is given)
# metric: most likely the mean absolute error in seconds (the per-user median baseline gives ~12.1M s MAE on a simulated
# last-gap holdout vs. 13.6M s on the leaderboard; its RMSE would be ~28M s) -> predict the conditional median
# #########################################################################################

# ---- linear median autoregression (in-book model) ------------------------------------------
TASK6_LINEAR_GAP_MODEL = GapRegressorConfig(
    method="linear_median",
    n_folds=5,
    lightgbm_n_estimators=0,                # not used by the linear model
    lightgbm_learning_rate=0.0,
    lightgbm_num_leaves=0,
    lightgbm_min_child_samples=0,
    seed=0,
)

# ---- LightGBM with an L1 loss ---------------------------------------------------------------
# 15 Nov (out-of-fold MAE, mean of 3 fold splits): old features 83.62 days -> + next-book reader timing and book-to-book
# timing (lib/time_gap.py) 81.65 -> 63 leaves / learning rate 0.02: 81.34 -> + the user's previous gap as an extra
# training row 81.12 (feature fraction 0.7: -0.07 only, not used; with the old features 2 or 4 earlier gaps were worse than 1)
TASK6_LIGHTGBM_GAP_MODEL = GapRegressorConfig(
    method="lightgbm",
    n_folds=5,
    lightgbm_n_estimators=1000,            # 16 Nov sweep (jobs/gpu_13): 600 trees 80.41 -> 1000 trees 80.38 days; 6 / 8 earlier
                                           # gaps 80.58 / 80.82, 511 leaves 80.46, min 50 / 200 per leaf 80.60 / 80.42
    lightgbm_learning_rate=0.02,
    lightgbm_num_leaves=255,
    lightgbm_min_child_samples=100,
    seed=0,
    n_earlier_gaps_as_training_rows=4,
    use_rows_from_end_feature=True,
)
# 15 Nov, later (mean of 2 fold splits): + "rows from end" input 80.96 -> 2 earlier gaps 80.82 -> + 127 leaves 80.52 ->
# 4 earlier gaps 80.48 -> 255 leaves 80.43; bagging (5 models, rows 0.8 / features 0.7): -0.06 with the old settings, +0.1
# (worse) with these -> not used (n_bagged_models stays 1)

# the model that writes output/task_6_predictions.csv -> compare the validation MAEs in task6_ensemble.ipynb
TASK6_SUBMITTED_GAP_MODEL = TASK6_LIGHTGBM_GAP_MODEL
