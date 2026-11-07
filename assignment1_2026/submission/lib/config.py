import torch

# Rating Range
RATING_MIN, RATING_MAX = 0.0, 5.0

# GPU if GPU is available / otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")