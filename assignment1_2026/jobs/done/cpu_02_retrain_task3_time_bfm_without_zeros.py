# Task 3: time-aware Bayesian FM retrained on the ratings > 0 (CPU, ~75 min), stored (old one -> trained_models_backup_with_zeros/)
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import assignment1
import lib.config as model_configs
from lib.helper_functioins import mse
from lib.model_storage import save_model_result, trained_model_path
assert model_configs.TRAIN_RATING_MODELS_WITHOUT_ZERO_RATINGS
task = assignment1.load_task3()
result = assignment1.train_task3_time_bayesian_factorization_machine(task)
stored_path = trained_model_path("task3", "time_bayesian_factorization_machine")
backup_path = stored_path.parent.parent.parent / "trained_models_backup_with_zeros" / "task3" / stored_path.name
backup_path.parent.mkdir(parents=True, exist_ok=True)
if stored_path.exists() and not backup_path.exists():
    shutil.copy2(stored_path, backup_path)
save_model_result(result, "task3", "time_bayesian_factorization_machine")
nonzero = task.validation_data.ratings > 0
print(f"RESULT task3 time_bayesian_factorization_machine MSE on validation rows > 0 {mse(result.valid_predictions[nonzero], task.validation_data.ratings[nonzero]):.5f}", flush=True)
