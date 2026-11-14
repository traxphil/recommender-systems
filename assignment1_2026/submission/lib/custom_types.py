
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

# Comments for Documentation: ------
# U = number of users in training file ~ 30k
# I = number of items in training file ~ 51k
# N = number of ratings in Rating Data ~ 1.67 Mil
# N_test = number of rows in the Test file
# K = index -> to N

# The users and items are indexed in the embedding vectors by their order of appearance


class RatingData(NamedTuple):
    # one rating per position k: user user_rows[k] -> gets the rating from ratings[k] for item item_rows[k]
    user_rows: np.ndarray                   # (N,)  int    values: 1-U          // which user gave rating k
    item_rows: np.ndarray                   # (N,)  int    values: 1-I          // which item was in rating k
    ratings: np.ndarray                     # (N,)  float  values: 0-5          // what was the score of rating k
    days: np.ndarray | None = None          # (N,)  int    values: 0-6575       // on which day rating k was given (day 0 = 2000-01-01)
    timestamps: np.ndarray | None = None    # (N,)  int    unix seconds (UTC)   // when rating k was given, clipped to 2000-01-01 .. 2018-01-01
    # days / timestamps are None when the task gives no time for its test rows (Task 1) -> no model can use time there

    # the ratings at the given positions (e.g. the training split); missing columns (None) stay None
    def at_positions(self, positions: np.ndarray) -> "RatingData":
        return RatingData(*(None if column is None else column[positions] for column in self))


class RatingTask(NamedTuple):
    training_data: RatingData               # random cut of the full data for training       
    validation_data: RatingData             # random cut of the full data for validation
    training_and_validation_data: RatingData  # entire data available for training & validation
    test_user_rows: np.ndarray              # (N_test,)  int    values: 0-U     // user for the test row, 0 for unknown!!
    test_item_rows: np.ndarray              # (N_test,)  int    values: 0-I     // item for the test row, 0 for unknown!!
    n_user_rows: int                        # U + 1 = rows of every user embedding table (row 0 = unknown)
    n_item_rows: int                        # I + 1 = rows of every item embedding table (row 0 = unknown)
    test_days: np.ndarray | None = None     # (N_test,)  int    values: 0-6575  // day of the test row / None if the test file has no timestamp (Task 1)
    test_timestamps: np.ndarray | None = None  # (N_test,) int unix seconds     // timestamp of the test row / None as test_days
    # N(u) for SVD++ is not stored here: the pipeline builds it from the ratings a model trains on (no test data)
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Interaction prediction (Tasks 2 / 4): which of two books did the user actually read?
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# P = number of pairs; one pair per position k: user user_rows[k] read first_item_rows[k] or second_item_rows[k]
class ChoicePairs(NamedTuple):
    user_rows: np.ndarray                   # (P,)  int    values: 1-U
    first_item_rows: np.ndarray             # (P,)  int    values: 0-I          // item1 of the test file
    second_item_rows: np.ndarray            # (P,)  int    values: 0-I          // item2 of the test file
    days: np.ndarray | None                 # (P,)  int    values: 0-6575       // None if the file has no timestamp (Task 2)
    first_is_positive: np.ndarray | None    # (P,)  bool                        // known for validation pairs, None for test pairs


class PairChoiceTask(NamedTuple):
    training_data: RatingData               # interactions the models learn from = training file minus the validation positives
    training_and_validation_data: RatingData  # every interaction of the training file (models refit on it for the test pairs)
    validation_pairs: ChoicePairs           # held-out interaction vs. a negative drawn like the test negatives
    test_pairs: ChoicePairs
    n_user_rows: int
    n_item_rows: int


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Time gap prediction (Task 6): how long until the user's next interaction?
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# interactions in chronological order per user; seconds = raw unix timestamps (gaps are measured in seconds)
class TimeGapTask(NamedTuple):
    training_data: RatingData               # every interaction except each user's last one (features of the validation rows)
    training_seconds: np.ndarray            # (N_train,) timestamps of training_data
    training_and_validation_data: RatingData  # every interaction of the training file (features of the test rows)
    training_and_validation_seconds: np.ndarray
    validation_user_rows: np.ndarray        # (U,) one row per user: their held-out last interaction
    validation_item_rows: np.ndarray        # (U,) the held-out (= next) book
    validation_gaps: np.ndarray             # (U,) seconds between the held-out and the previous interaction (the target)
    test_user_rows: np.ndarray              # (N_test,)
    test_item_rows: np.ndarray              # (N_test,) the next book, given in the test file
    n_user_rows: int
    n_item_rows: int


# The return type of the model
# (pair-choice models: predictions are (P, 2) scores of first / second item; the pair blender: (P,) P(first item is the real one))
class ModelResult(NamedTuple):
    model: Any                              # model of any kind
    history: list                           # one dict per epoch {"epoch", "train_mse", "valid_mse"}, epoch 0 = starting point / empty if not iterative
    valid_predictions: np.ndarray           # (N_valid,)    float   values: e.g. 0-5 for the rating task / standard interface for merging
    test_predictions: np.ndarray            # (N_test,)     float   values: e.g. 0-5


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Model configurations -> the values for every task are set in lib/config.py
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# r(u,i) = mu + b_u + b_i in closed form
@dataclass(frozen=True)
class BiasModelConfig:
    user_bias_regularization: float         # b_u = sum of residuals / (number of ratings + user_bias_regularization)
    item_bias_regularization: float         # b_i = sum of residuals / (number of ratings + item_bias_regularization)
    n_iterations: int                       # alternating user / item updates


# keyword arguments of assignment1.train_latent_factor_model (SVD / SVD++)
@dataclass(frozen=True)
class LatentFactorConfig:
    use_implicit_feedback: bool             # False = SVD, True = SVD++
    use_test_pairs_as_implicit_feedback: bool  # SVD++: N(u) = every known pair of the training + test file (lib/data_loading.py)
    n_factors: int
    learning_rate: float
    user_bias_regularization: float         # lambda of b_u (also for the closed-form starting biases)
    item_bias_regularization: float         # lambda of b_i
    factor_regularization: float
    ratings_per_batch: int
    max_epochs: int
    early_stopping_patience: int
    refit_on_train_and_valid: bool          # retrain on 100 % for the best number of epochs before predicting the test file
    seed: int


# item-item kNN on the residuals of the bias model
@dataclass(frozen=True)
class ItemKnnConfig:
    n_neighbors: int                        # most similar items kept per item
    similarity_shrinkage: float             # s_ij * n_ij / (n_ij + shrinkage) -> few common users = less trust
    prediction_shrinkage: float             # added to the sum of neighbour weights -> weak neighbourhoods stay near the baseline
    baseline_biases: BiasModelConfig        # baseline mu + b_u + b_i the residuals are measured against = the bias model's fit
                                            # (lib/config.py: TASKn_SHARED_BASELINE_BIASES, also used by TASKn_BIAS_MODEL)
    max_ratings_per_user_for_similarity: int  # very active users are subsampled for the similarities (memory)
    items_per_similarity_block: int         # similarities are computed for this many items at a time (memory)
    seed: int


# P(rating = 0) from how often the user / the item got a 0 elsewhere
@dataclass(frozen=True)
class ZeroModelConfig:
    user_prior_strength: float              # zero share = (zeros + strength * global share) / (ratings + strength)
    item_prior_strength: float
    inverse_regularization: float           # C of the logistic regression (bigger = weaker regularization)
    max_iterations: int


# factorization machine over user, item, user activity bucket, item popularity bucket (no time -> Task 1)
@dataclass(frozen=True)
class FactorizationMachineConfig:
    n_factors: int
    n_activity_buckets: int                 # users / items are also grouped by their number of ratings (quantiles)
    learning_rate: float
    bias_regularization: float              # one lambda for b_u and b_i (not split, unlike LatentFactorConfig)
    factor_regularization: float
    ratings_per_batch: int
    max_epochs: int
    early_stopping_patience: int
    refit_on_train_and_valid: bool
    seed: int


# Bayesian factorization machine (Gibbs sampling with myFM): no learning rate / lambda / early stopping to tune
@dataclass(frozen=True)
class BayesianFactorizationMachineConfig:
    rank: int                               # number of factors
    n_burn_in_iterations: int               # first Gibbs draws, thrown away (the sampler has not reached the posterior yet)
    n_kept_samples: int                     # the draws after the burn-in, averaged for the predictions
    use_implicit_feedback: bool             # also N(u) / N(i) as features (SVD++-style, normalised)
    use_test_pairs_as_implicit_feedback: bool  # N(u) / N(i) = every known pair of the training + test file
    refit_on_train_and_valid: bool
    seed: int


# set transformer over the user's rated books, asked about one more book
@dataclass(frozen=True)
class RatingTransformerConfig:
    embedding_dimension: int                # d of the book / rating-token embeddings and the encoder
    n_blocks: int                           # transformer encoder blocks
    n_heads: int
    dropout: float
    max_history_length: int                 # books per user history (a random subset of longer histories)
    use_test_pairs_in_history: bool         # also the known pairs without a visible rating (validation / test pairs)
    learning_rate: float
    bias_regularization: float              # one lambda for b_u and b_i
    factor_regularization: float            # on the asked book's embedding
    ratings_per_batch: int
    max_epochs: int
    early_stopping_patience: int
    refit_on_train_and_valid: bool
    seed: int


# I-AutoRec: item-based autoencoder over the users' ratings of every book
@dataclass(frozen=True)
class AutoRecConfig:
    hidden_dimension: int                   # Sedhain et al. 2015: 500
    regularization: float                   # lambda of lambda / 2 (||W||^2 + ||V||^2) on the scale of the sum loss
    user_bias_regularization: float         # closed-form baseline mu + b_u + b_i, the autoencoder models its residuals
    item_bias_regularization: float
    masked_fraction: float                  # share of a book's ratings hidden from its input in every batch (the ones predicted)
    learning_rate: float
    books_per_batch: int
    max_epochs: int
    early_stopping_patience: int
    refit_on_train_and_valid: bool
    seed: int


# integrated model: SVD++ + learned weights over every book's k neighbours (Koren 2008)
@dataclass(frozen=True)
class IntegratedModelConfig:
    n_factors: int
    use_test_pairs_as_implicit_feedback: bool  # N(u) = every known pair of the training + test file
    n_neighbors: int                        # k neighbours per book (shrunk residual cosine, as the item kNN)
    similarity_shrinkage: float
    max_ratings_per_user_for_similarity: int
    items_per_similarity_block: int
    learning_rate: float
    user_bias_regularization: float         # also the closed-form baseline of the residuals
    item_bias_regularization: float
    factor_regularization: float
    neighbour_regularization: float         # lambda on the squared w_ij, c_ij
    ratings_per_batch: int
    max_epochs: int
    early_stopping_patience: int
    refit_on_train_and_valid: bool
    seed: int


# blender = the trained model that combines the sub-models' predictions (stacking)
@dataclass(frozen=True)
class BlenderConfig:
    method: str                             # "ridge" (learned weighted average) or "lightgbm" (gradient-boosted trees)
    blended_models: tuple[str, ...]         # names of the sub-models whose predictions are combined
    use_count_features: bool                # also give log(1 + #ratings) of user and item ("support" meta-features)
    n_folds: int                            # cross-validation folds for the out-of-fold score
    ridge_alpha: float
    lightgbm_n_estimators: int
    lightgbm_learning_rate: float
    lightgbm_num_leaves: int
    lightgbm_min_child_samples: int
    seed: int
    # Task 3+: when the rating happened relative to the user's / item's other ratings (needs RatingData.days)
    # defaults -> the Task 1 configs and the stored Task 1 blenders stay valid without these fields
    use_time_features: bool = False
    activity_window_days: int = 30          # "user activity around t" = the user's ratings within +- this many days


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Temporal model configurations (Task 3)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# time-dependent bias terms on top of mu + b_u + b_i (PML Sec. 7.2.2, Eq. 7.8 - 7.12; Koren 2009):
#   b_bin(t) + b_{i,bin(t)} + alpha_u * dev_u(t) + sum over period lengths s of b_{u,period_s(t)}
@dataclass(frozen=True)
class TemporalBiasConfig:
    n_time_bins: int                        # global + per-item bias per time bin (bins = quantiles of the training days)
    user_period_lengths_in_days: tuple[int, ...]  # one user bias per (user, period) for every length, e.g. (7, 90) = week + quarter
    drift_exponent: float                   # dev_u(t) = sign(t - t_u) * |t - t_u|^exponent, t_u = the user's mean rating day
    time_bias_regularization: float         # item-bin and user-period biases: b = sum of residuals / (n + regularization)
    drift_regularization: float             # alpha_u


# r(u,i,t) = mu + b_u + b_i + temporal biases in closed form (alternating exact updates, like the static bias model)
@dataclass(frozen=True)
class TimeBiasModelConfig:
    temporal_biases: TemporalBiasConfig
    bias_regularization: float              # b_u, b_i, b_bin (same scheme as the static bias model)
    n_iterations: int                       # rounds of updating every bias table once


# timeSVD++ = SVD++ on top of the time-aware bias model (+ optionally drifting user factors p_u(t) = p_u + alpha_u^p * dev_u(t))
@dataclass(frozen=True)
class TimeSVDppConfig:
    use_implicit_feedback: bool             # False = timeSVD, True = timeSVD++
    use_test_pairs_as_implicit_feedback: bool  # N(u) = every known pair of the training + test file (lib/data_loading.py)
    n_factors: int
    use_factor_drift: bool                  # PML Eq. 7.13 without the per-day factor term
    time_biases: TimeBiasModelConfig        # closed-form time-aware bias model = starting point (epoch 0)
    train_temporal_biases: bool             # False = the temporal biases stay at their closed-form values
    learning_rate: float
    bias_regularization: float              # one lambda for b_u and b_i (not split, unlike LatentFactorConfig)
    factor_regularization: float            # also for the factor drift alpha_u^p
    ratings_per_batch: int
    max_epochs: int
    early_stopping_patience: int
    refit_on_train_and_valid: bool
    seed: int


# factorization machine + one-hot calendar fields of the rating's timestamp (Task 3)
@dataclass(frozen=True)
class TimeFactorizationMachineConfig(FactorizationMachineConfig):
    calendar_fields: tuple[str, ...]        # any of "hour_of_day", "weekday", "month", "year" (UTC; see lib/temporal_features.py)


# item-item kNN whose neighbours count less the further their rating day is from the asked day
@dataclass(frozen=True)
class TimeItemKnnConfig(ItemKnnConfig):
    time_decay_days: float                  # neighbour weight * (floor + (1 - floor) * exp(-|t - t_uj| / time_decay_days))
    time_decay_floor: float                 # share of the weight a neighbour keeps however far away in time (0 = pure decay)


# zero model + how many zeros the user gave around the asked day
@dataclass(frozen=True)
class TimeZeroModelConfig(ZeroModelConfig):
    window_lengths_in_days: tuple[int, ...]  # 0 = same day; +- window around t
    window_prior_strength: float            # zero share in the window shrunk towards the global share


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Interaction model configurations (Tasks 2 / 4)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# score(u, i) = sum of shrunk cosine similarities between i and the books u has read (top neighbours of i only)
@dataclass(frozen=True)
class ItemCosineConfig:
    n_neighbors: int                        # most similar items kept per item
    similarity_shrinkage: float             # cosine * n_ij / (n_ij + shrinkage), n_ij = users who read both
    max_interactions_per_user_for_similarity: int  # very active users are subsampled for the similarities (memory)
    items_per_similarity_block: int
    seed: int


# EASE: linear item-item autoencoder in closed form (Steck 2019)
@dataclass(frozen=True)
class EaseConfig:
    regularization: float                   # lambda on the diagonal of the item-item Gram matrix
    max_items: int                          # only the most popular items get weights (dense items x items matrix in memory)
    use_gpu: bool                           # False: inverse + weights in RAM (all 51k books of Task 2 = 10.4 GB per matrix)
    history_time_decay_days: float          # 0 = plain EASE; > 0: the user's books count (floor + (1 - floor) exp(-|t - t_uj| / decay))
    history_time_decay_floor: float
    items_per_gram_block: int               # Gram matrix X^T X is filled this many columns at a time
    pairs_per_scoring_chunk: int


# Mult-VAE: variational autoencoder over the user's 0/1 history with a multinomial likelihood (Liang et al. 2018)
@dataclass(frozen=True)
class MultVaeConfig:
    hidden_dimension: int                   # items -> hidden -> latent -> hidden -> items (Liang et al.: 600)
    latent_dimension: int                   # (Liang et al.: 200)
    input_dropout: float                    # share of the user's books hidden from the encoder in training (denoising)
    kl_weight_max: float                    # beta of the KL term, raised linearly from 0 (Liang et al.: 0.2)
    kl_annealing_epochs: int                # epochs until beta reaches kl_weight_max
    learning_rate: float
    users_per_batch: int
    max_epochs: int
    early_stopping_patience: int            # epochs without a better validation pair accuracy
    max_items: int                          # input / output width (the most popular items)
    seed: int


# Task 5: which book follows which (first-order Markov chain with a few skip steps, PML Sec. 7.4)
@dataclass(frozen=True)
class TransitionConfig:
    steps_ahead: int                        # j -> i counted when i comes 1 .. steps_ahead interactions after j
    step_decay: float                       # weight of a transition d steps ahead = step_decay ** (d - 1)
    shrinkage: float                        # P(i | j) = count(j -> i) / (count(j -> anything) + shrinkage)
    context_length: int                     # the user's last books that vote for the next one
    context_decay: float                    # weight of the m-th last book = context_decay ** (m - 1)


# Task 5: SASRec, self-attention over the user's sequence (Kang & McAuley 2018, PML Sec. 7.7.2)
@dataclass(frozen=True)
class SasRecConfig:
    hidden_dimension: int                   # item / position embedding size d
    n_blocks: int                           # self-attention blocks
    n_heads: int
    max_sequence_length: int                # only the user's last L books are read
    dropout: float
    learning_rate: float
    users_per_batch: int
    max_epochs: int
    early_stopping_patience: int            # epochs without a better validation pair accuracy
    seed: int


# Task 6: regression of log(1 + gap) with an absolute (L1) loss -> conditional median, the best prediction under MAE
@dataclass(frozen=True)
class GapRegressorConfig:
    method: str                             # "linear_median" (autoregression, PML Sec. 7.1) or "lightgbm"
    n_folds: int                            # cross-validation folds for the out-of-fold MAE
    lightgbm_n_estimators: int
    lightgbm_learning_rate: float
    lightgbm_num_leaves: int
    lightgbm_min_child_samples: int
    seed: int


# LightGBM classifier: P(first item is the real one) from the sub-models' scores of both items + meta-features
@dataclass(frozen=True)
class PairBlenderConfig:
    combined_models: tuple[str, ...]        # names of the sub-models whose scores are combined
    use_time_features: bool                 # Task 4: item popularity around t, item age at t (needs the pair days)
    activity_window_days: int               # "item popularity around t" = interactions within +- this many days
    n_folds: int                            # cross-validation folds for the out-of-fold accuracy
    lightgbm_n_estimators: int
    lightgbm_learning_rate: float
    lightgbm_num_leaves: int
    lightgbm_min_child_samples: int
    seed: int
