# Task 1 sweep: preliminary result (7 Nov 2026)

222 runs on the RTX 5080 (4.0 GPU hours), 11 rounds. Model and training code unchanged:
every run is one call of `assignment1.train_latent_factor_model` (refit skipped).
Score = validation MSE at the best early-stopping epoch (10 % split of `tasks_1_2_train.csv`).
Finalists were re-run on validation split seeds 1 and 2 and compared against the current settings on the same split.

## Recommended settings (bias_regularization stays 3.0)

| model | n_factors | factor_regularization | ratings_per_batch | learning_rate | mean valid MSE (splits 0/1/2) | current settings | fit time |
|---|---|---|---|---|---|---|---|
| SVD++ | 500 | 8 | 512 | 2.5e-4 | **0.74320** | 0.75539 | 344 s |
| SVD++ (faster) | 500 | 8 | 1024 | 3.5e-4 | 0.74436 | 0.75539 | 195 s |
| SVD | 120 | 8 | 512 | 2.5e-4 | 0.74982 | 0.76378 | 80 s |

The best SVD++ beats the current SVD++ on every split by -0.0120 / -0.0120 / -0.0126.
Seed noise of a single run is about +-0.0004.

## What mattered (largest effect first)

1. More factors with much less factor regularization: k 10 -> 120+ and reg 50 -> 8 (about -0.007). Gains flatten above k = 500 (k = 1000 is not better).
2. Smaller batches with a proportionally lower learning rate: 16384 / 2e-3 -> 512 / 2.5e-4 (about -0.004). Batch 256 is no better than 512.
3. bias_regularization 3 is best (1.5 and 6 are worse, 2 and 4 are tied).
4. The best epoch is always about 3: early stopping does a large part of the regularization.
5. SVD with the default lr = 0.01 never improves on the bias-only start (0.7647 on split 0) -> the current "svd" model is effectively the bias model.

## Files

- `sweep_results.csv`: one row per run (all parameters, best valid MSE, best epoch, run time)
- `sweep_epoch_history.csv`: train / valid MSE of every epoch of every run
- `sweep_log.txt`: everything the training printed
- `sweep_queue.json`: every config that was run, in run order
