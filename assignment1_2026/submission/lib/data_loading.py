import csv
import numpy as np

from lib.custom_types import RatingData, RatingTask
from lib.helper_functioins import ids_to_rows, number_by_first_appearance

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# read & write
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# read the CSV file into columns as trings
def read_csv_columns(csv_path) -> dict[str, tuple[str, ...]]:

    with open(csv_path, newline="") as csv_file:
        reader = csv.reader(csv_file)
        column_names = next(reader)
        rows = [row for row in reader if row]  # skip blank lines

    # zip(*rows) turns rows into columns
    return dict(zip(column_names, zip(*rows)))

# write prediction results into the file
def write_prediction_file(test_csv_path, output_csv_path, predictions_in_test_order):

    with open(test_csv_path) as test_file, open(output_csv_path, "w") as output_file:
        output_file.write(test_file.readline())  # header 
        test_lines = [line.strip() for line in test_file.read().splitlines() if line.strip()]

        assert len(test_lines) == len(predictions_in_test_order), \
            f"{len(test_lines)} test rows vs {len(predictions_in_test_order)} predictions"
        
        for test_line, prediction in zip(test_lines, predictions_in_test_order):
            output_file.write(f"{test_line},{prediction}\n")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Data loading Task 1
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# create the prediction task problem for task 1
def load_rating_prediction_task(training_csv_path, test_csv_path, validation_fraction: float = 0.1, split_seed: int = 0) -> RatingTask:
    
    # read both files as string columns (training: user, item, timestamp, rating | test: user, item)
    training_columns = read_csv_columns(training_csv_path)
    test_columns = read_csv_columns(test_csv_path)

    # numbering the users and items by their first appearance for the embedding index / id (unknowns to 0!)
    row_of_user = number_by_first_appearance(training_columns["user"])
    row_of_item = number_by_first_appearance(training_columns["item"])

    # populating the embeddings
    user_rows = ids_to_rows(training_columns["user"], row_of_user)
    item_rows = ids_to_rows(training_columns["item"], row_of_item)
    ratings = np.array(training_columns["rating"], dtype=np.float32)
    test_user_rows = ids_to_rows(test_columns["user"], row_of_user)
    test_item_rows = ids_to_rows(test_columns["item"], row_of_item)
    n_user_rows = len(row_of_user) + 1  # U users + row 0
    n_item_rows = len(row_of_item) + 1  # I items + row 0

    # training and validation split / randomized with a seed
    shuffled_indices = np.random.default_rng(split_seed).permutation(len(ratings))
    n_valid = int(len(ratings) * validation_fraction)
    valid_indices, train_indices = shuffled_indices[:n_valid], shuffled_indices[n_valid:]

    # create rating task instance
    return RatingTask(
        training_data=RatingData(user_rows[train_indices], item_rows[train_indices], ratings[train_indices]),
        validation_data=RatingData(user_rows[valid_indices], item_rows[valid_indices], ratings[valid_indices]),
        training_and_validation_data=RatingData(user_rows, item_rows, ratings),
        test_user_rows=test_user_rows,
        test_item_rows=test_item_rows,
        n_user_rows=n_user_rows,
        n_item_rows=n_item_rows,
    )

