# first job: checks that the runner's Python finds the project and the GPU
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
import assignment1
import lib.config as model_configs
print("python", sys.executable)
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("working directory", Path.cwd())
print("trained models", model_configs.TRAINED_MODELS_DIR, "| task 1 blender:", model_configs.TASK1_SUBMITTED_BLENDER.method)
