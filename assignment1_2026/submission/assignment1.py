import gzip
import random
import scipy as sp
import numpy as np
import implicit
import surprise
import time
import math
import io
from pathlib import Path

from collections import defaultdict
# from implicit import bpr
from surprise import SVDpp, SVD, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split
from tqdm import tqdm as tqdm

# #########################################################################################
# Helper Functions for the Assignment
# #########################################################################################

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Infrastructure & Data Fetching
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# time-stamped log print
def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# the column mask is an array specifying which columns can be used - e.g. [1, 0, 1, 0]
def read_csv(path: Path, column_mask):

    reader = open(path, 'r')
    header = reader.readline().strip().split(',')
    mask = np.asarray(column_mask, dtype=bool)

    # store the data / remove letters so we can directly store the IDs as numeric values
    # runs on the assumption that all IDs have 10 digits
    # recover value with: f"u{NUMBER:010d}"
    body = reader.read().replace('u', '').replace('i', '')
    data = np.loadtxt(io.StringIO(body), delimiter=',', dtype=np.int64, usecols=np.flatnonzero(mask), ndmin=2)
    reader.close()

    # returns header names, data
    # only keeps the columns where the mask is set to True
    kept_header = np.array(header)[mask].tolist() 
    return kept_header, data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Errors, Constants, Math Operations
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# intersection over union
def jaccard_sim(set_a: set, set_b: set) -> float:
    intersection = len(set_a & set_b)
    union = len(set_a) + len(set_b) - intersection
    return intersection / union

def cosine_sim_set(set_a: set, set_b: set) -> float:
    pass

# np.ndarray
def cosine_sim(vec_a: dict, vec_b: dict) -> float:
    pass

def pearson_sim(set_a: set, set_b: set) -> float:
    pass

# mean squared error
def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2))

# root mean squared error
def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return math.sqrt(mse(predictions, targets))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Logic Helpers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Constants
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 




# #########################################################################################
# Task 1
# #########################################################################################

def svdppModel(trainFile):
    # Copy the training file without its header (Surprise can't skip it)
    a = open(trainFile, 'r')
    b = open("temp.csv", 'w')
    a.readline()
    for l in a.readlines():
        b.write(l)
    a.close()
    b.close()

    # Load the ratings; rating_scale=(0, 5) because the data contains 0-star ratings
    reader = Reader(line_format='user item timestamp rating', sep=',', rating_scale=(0, 5))
    data = Dataset.load_from_file("temp.csv", reader=reader)

    # SVD++ = SVD plus an implicit term built from all items each user has rated
    # model = SVDpp(n_factors=20, n_epochs=100, reg_bu=0.005, reg_bi=0.005, reg_pu=0.15, reg_qi=0.15, reg_yj=0.15, cache_ratings=True, random_state=0, verbose=True)
    # GridSearchCV
    model = SVD(n_factors=20, n_epochs=100, random_state=0, verbose=True, reg_bu=0.5, reg_bi=0.5, reg_pu=0.5, reg_qi=0.5)
    trainset, testset = train_test_split(data, test_size=.25, random_state=0)
    model.fit(trainset)

    # MSE on the held-out 25%
    predictions = model.test(testset)
    sse = 0
    for p in predictions:
        sse += (p.r_ui - p.est)**2
    print("Error on own test set = ", str(sse / len(predictions)))

    return trainset, model