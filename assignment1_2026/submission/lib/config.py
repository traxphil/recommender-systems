from pathlib import Path

import torch

from lib.custom_types import (BiasModelConfig, BlenderConfig, EaseConfig, FactorizationMachineConfig, GapRegressorConfig,
                              ItemCosineConfig, ItemKnnConfig, LatentFactorConfig, MultVaeConfig, PairBlenderConfig,
                              SasRecConfig, TemporalBiasConfig, TimeBiasModelConfig, TimeItemKnnConfig, TimeSVDppConfig,
                              TimeZeroModelConfig, TransitionConfig, ZeroModelConfig)


# #########################################################################################
# Global
# #########################################################################################

# Rating Range
RATING_MIN, RATING_MAX = 0.0, 5.0

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
# starting point = the SVD settings above
TASK1_FACTORIZATION_MACHINE = FactorizationMachineConfig(
    n_factors=120,
    n_activity_buckets=10,
    learning_rate=2.5e-4,
    bias_regularization=3.0,
    factor_regularization=8.0,
    ratings_per_batch=512,
    max_epochs=1000,
    early_stopping_patience=6,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- blenders ----------------------------------------------------------------------------
TASK1_BLENDED_MODELS = ("bias_model", "svd", "svdpp", "item_knn", "zero_model", "factorization_machine")

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
)

# the blend that writes output/task_1_predictions.csv
TASK1_SUBMITTED_BLENDER = TASK1_RIDGE_BLENDER


# #########################################################################################
# Task 2 - interaction prediction: which of two books did the user read?
# test negatives = the book of a random training interaction (in proportion to popularity) -> validation pairs built the
# same way (lib/data_loading.py); sub-models score (user, book), a LightGBM pair blender decides
# #########################################################################################

# ---- item cosine similarity to the user's history ----------------------------------------
TASK2_ITEM_COSINE = ItemCosineConfig(
    n_neighbors=100,
    similarity_shrinkage=20.0,
    max_interactions_per_user_for_similarity=300,  # as the item kNN of Task 1 (memory)
    items_per_similarity_block=2000,
    seed=0,
)

# ---- EASE (linear autoencoder) -----------------------------------------------------------
TASK2_EASE = EaseConfig(
    regularization=500.0,                   # Steck 2019 uses 200 - 1000; Task 4 check: 500 best alone, 30 - 2000 within 0.002 in the blend
    max_items=60_000,                       # all ~51k books (the 25k most popular cover only 82 % of the interactions)
    use_gpu=False,                          # 51k x 51k float32 = 10.4 GB per matrix, ~21 GB peak -> in RAM, a few minutes per fit
    history_time_decay_days=0.0,            # plain EASE (the Task 2 test file has no timestamp)
    history_time_decay_floor=0.0,
    items_per_gram_block=2000,
    pairs_per_scoring_chunk=20_000,
)

# ---- Mult-VAE (non-linear autoencoder) ----------------------------------------------------
# architecture and beta as in Liang et al. 2018 (600 -> 200, dropout 0.5, beta up to 0.2)
TASK2_MULT_VAE = MultVaeConfig(
    hidden_dimension=600,
    latent_dimension=200,
    input_dropout=0.5,
    kl_weight_max=0.2,
    kl_annealing_epochs=20,
    learning_rate=1e-3,
    users_per_batch=500,
    max_epochs=200,
    early_stopping_patience=10,
    max_items=60_000,                       # all books (input / output layer 51k x 600 -> ~60M weights, fine on the GPU)
    seed=0,
)

# ---- pair blender --------------------------------------------------------------------------
TASK2_PAIR_BLENDER = PairBlenderConfig(
    combined_models=("popularity", "item_cosine", "ease", "mult_vae"),
    use_time_features=False,                # the Task 2 test file has no timestamp
    activity_window_days=30,                # not used without time features
    n_folds=5,
    lightgbm_n_estimators=400,
    lightgbm_learning_rate=0.05,
    lightgbm_num_leaves=31,
    lightgbm_min_child_samples=200,
    seed=0,
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
TASK3_SVDPP = LatentFactorConfig(
    use_implicit_feedback=True,
    n_factors=500,
    learning_rate=2e-3,
    user_bias_regularization=3.0,           # own lambdas (independent of TASK3_SHARED_BASELINE_BIASES)
    item_bias_regularization=5.0,
    factor_regularization=8.0,
    ratings_per_batch=16384,
    max_epochs=1000,
    early_stopping_patience=6,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- item-item kNN on the bias model's residuals ----------------------------------------
TASK3_ITEM_KNN = ItemKnnConfig(
    n_neighbors=50,                         # Task 1 values
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
    user_period_lengths_in_days=(1, 7, 30, 90, 365),  # test rows whose user has a training rating in the same day / week / month: 10 / 43 / 76 %
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
    n_factors=500,
    use_factor_drift=False,                 # PML Sec. 7.2.3: biases first; True adds drifting user factors p_u(t)
    time_biases=TASK3_TIME_BIAS_MODEL,
    train_temporal_biases=False,            # True: Adam trains them further (overfits them within ~1 epoch)
    learning_rate=2e-3,
    bias_regularization=3.0,
    factor_regularization=8.0,
    ratings_per_batch=16384,
    max_epochs=1000,
    early_stopping_patience=6,
    refit_on_train_and_valid=True,
    seed=0,
)

# ---- time-decayed item-item kNN ----------------------------------------------------------
TASK3_TIME_ITEM_KNN = TimeItemKnnConfig(
    n_neighbors=50,                         # neighbourhoods as in TASK3_ITEM_KNN
    similarity_shrinkage=100.0,
    prediction_shrinkage=1.0,
    baseline_biases=TASK3_SHARED_BASELINE_BIASES,   # = the bias model's baseline (blend: kNN - bias model)
    max_ratings_per_user_for_similarity=300,
    items_per_similarity_block=2000,
    seed=0,
    # quick check on the validation split: no decay setting beats the plain kNN on its own (0.8997);
    # 365 days / floor 0.5: 0.9006 -> kept as a differently weighted neighbourhood for the blend
    time_decay_days=365.0,
    time_decay_floor=0.5,
)

# ---- temporal zero model -----------------------------------------------------------------
TASK3_TIME_ZERO_MODEL = TimeZeroModelConfig(
    user_prior_strength=5.0,
    item_prior_strength=20.0,
    inverse_regularization=1.0,
    max_iterations=1000,
    window_lengths_in_days=(0, 30),         # same day + the month around the rating
    window_prior_strength=2.0,              # 1 zero among 1 other rating that day -> shrunk zero share 0.34
)

# ---- blenders ----------------------------------------------------------------------------
TASK3_BLENDED_MODELS = ("bias_model", "svd", "svdpp", "item_knn", "zero_model",
                        "time_bias_model", "time_svdpp", "time_item_knn", "time_zero_model")

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
TASK4_ITEM_COSINE = ItemCosineConfig(
    n_neighbors=100,
    similarity_shrinkage=20.0,
    max_interactions_per_user_for_similarity=300,
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
TASK4_MULT_VAE = MultVaeConfig(
    hidden_dimension=600,
    latent_dimension=200,
    input_dropout=0.5,
    kl_weight_max=0.2,
    kl_annealing_epochs=20,
    learning_rate=1e-3,
    users_per_batch=500,
    max_epochs=200,
    early_stopping_patience=10,
    max_items=60_000,
    seed=0,
)

# ---- pair blender --------------------------------------------------------------------------
TASK4_PAIR_BLENDER = PairBlenderConfig(
    combined_models=("popularity", "item_cosine", "ease", "time_ease", "mult_vae"),
    use_time_features=True,
    activity_window_days=30,                # "popularity around t" = interactions with the book within +- 30 days
    n_folds=5,
    lightgbm_n_estimators=400,
    lightgbm_learning_rate=0.05,
    lightgbm_num_leaves=31,
    lightgbm_min_child_samples=200,
    seed=0,
)


# #########################################################################################
# Task 5 - which book comes next? (one test pair per user: the book after the last training interaction)
# leave-last-out validation pairs (lib/data_loading.load_next_item_task); 87k users, 232k books, ~12 interactions per user
# Task 4 sub-models (EASE / Mult-VAE on the most popular books) + transition model + SASRec -> pair blender
# #########################################################################################

# ---- item cosine similarity to the user's history ----------------------------------------
TASK5_ITEM_COSINE = ItemCosineConfig(
    n_neighbors=100,
    similarity_shrinkage=20.0,
    max_interactions_per_user_for_similarity=300,
    items_per_similarity_block=2000,
    seed=0,
)

# ---- transition model (which book follows which) ----------------------------------------
TASK5_TRANSITION = TransitionConfig(
    steps_ahead=3,                          # j -> i also when i comes 2 or 3 books later (weights 1, 0.5, 0.25)
    step_decay=0.5,
    shrinkage=5.0,
    context_length=5,                       # the user's last 5 books vote, weights 1, 0.5, 0.25, ...
    context_decay=0.5,
)

# ---- EASE + time-weighted EASE (one set of weights, in RAM) -------------------------------
TASK5_EASE = EaseConfig(
    regularization=500.0,
    max_items=50_000,                       # of 232k books: covers 85 % of the interactions; 10 GB per matrix, ~20 GB peak
    use_gpu=False,
    history_time_decay_days=180.0,          # time-weighted variant: books read long before "now" (the last interaction) count less
    history_time_decay_floor=0.0,
    items_per_gram_block=2000,
    pairs_per_scoring_chunk=20_000,
)

# ---- Mult-VAE (non-linear autoencoder) ----------------------------------------------------
TASK5_MULT_VAE = MultVaeConfig(
    hidden_dimension=600,
    latent_dimension=200,
    input_dropout=0.5,
    kl_weight_max=0.2,
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
TASK5_SASREC = SasRecConfig(
    hidden_dimension=64,
    n_blocks=2,
    n_heads=1,
    max_sequence_length=50,                 # 90 % of the users have fewer interactions
    dropout=0.5,
    learning_rate=1e-3,
    users_per_batch=256,
    max_epochs=200,
    early_stopping_patience=10,
    seed=0,
)

# ---- pair blender --------------------------------------------------------------------------
TASK5_PAIR_BLENDER = PairBlenderConfig(
    combined_models=("popularity", "item_cosine", "transition", "ease", "time_ease", "mult_vae", "sasrec"),
    use_time_features=True,                 # t = the day of the user's last interaction ("now")
    activity_window_days=30,
    n_folds=5,
    lightgbm_n_estimators=400,
    lightgbm_learning_rate=0.05,
    lightgbm_num_leaves=31,
    lightgbm_min_child_samples=200,
    seed=0,
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
TASK6_LIGHTGBM_GAP_MODEL = GapRegressorConfig(
    method="lightgbm",
    n_folds=5,
    lightgbm_n_estimators=600,
    lightgbm_learning_rate=0.03,
    lightgbm_num_leaves=31,
    lightgbm_min_child_samples=100,
    seed=0,
)

# the model that writes output/task_6_predictions.csv -> compare the validation MAEs in task6_ensemble.ipynb
TASK6_SUBMITTED_GAP_MODEL = TASK6_LIGHTGBM_GAP_MODEL
