from pathlib import Path

import torch
import torch.nn as nn

from lib.config import DEVICE, TRAINED_MODELS_DIR
from lib.custom_types import ModelResult
from lib.helper_functioins import log


# every trained model of a task lives in trained_models/<task>/<model>.pt
def trained_model_path(task_name: str, model_name: str) -> Path:
    return TRAINED_MODELS_DIR / task_name / f"{model_name}.pt"


# whole ModelResult in one file: model object, history, validation + test predictions (-> blending needs no retraining)
# stored as a plain dict -> still loads if fields are added to ModelResult later
def save_model_result(model_result: ModelResult, task_name: str, model_name: str) -> Path:
    file_path = trained_model_path(task_name, model_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_result._asdict(), file_path)
    log(f"stored {model_name} -> {file_path}")
    return file_path


# load on the CPU first; PyTorch models then move only their parameters to the device
# -> plain Python attributes (e.g. the SVD++ N(u) matrix, a scipy CSR matrix) stay on the CPU, exactly as during training
# -> models without a device (bias model, kNN, zero model, blenders) are returned as they are
# weights_only=False: the file contains pickled Python objects, not only tensors -> only load files you saved yourself
# Source: PyTorch docs, torch.load (weights_only defaults to True since PyTorch 2.6)
def load_model_result(task_name: str, model_name: str) -> ModelResult:
    stored_fields = torch.load(trained_model_path(task_name, model_name), map_location="cpu", weights_only=False)
    if isinstance(stored_fields["model"], nn.Module):
        stored_fields["model"].to(DEVICE)
    return ModelResult(**stored_fields)
