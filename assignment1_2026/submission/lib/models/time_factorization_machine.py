import torch

from lib.config import DEVICE, FIRST_PLAUSIBLE_TIMESTAMP
from lib.custom_types import ModelResult, RatingData, RatingTask, TimeFactorizationMachineConfig
from lib.model_storage import load_model_result, save_model_result
from lib.models.factorization_machine import FactorizationMachine, base_field_arguments, training_options
from lib.temporal_features import (CALENDAR_FIELD_SIZES, SECONDS_PER_HOUR, calendar_field_rows_per_hour, first_and_last_year,
                                   require_test_timestamps)
from lib.torch_trainer import fit_refit_and_predict

# Time-aware factorization machine (Task 3): the factorization machine of Task 1 + one-hot calendar fields of the timestamp
# (Rendle 2010; PML Sec. 6.1: Eq. 6.1 encodes the weekday one-hot, Exercise 6.1 suggests "the season or day of week"):
#   fields: user | item | user activity bucket | item popularity bucket | hour of day | weekday | month | year
#   r = mu + sum_f w_f + sum_{f < g} <v_f, v_g>
#   -> <v_item, v_month> = a book's seasonal popularity, <v_user, v_weekday> = a user's weekday habits,
#      <v_item, v_year> = how a book's ratings drift over the years, ...
# the calendar fields are in UTC (the users' time zones are unknown); config.calendar_fields selects which ones are used
# note on this data: every timestamp lies between 05:00 and 09:59 UTC and that window moves by one hour with US daylight
# saving time -> the timestamps look like dates (US Central midnight) + an offset, so "hour_of_day" carries little real signal


class TimeFactorizationMachine(FactorizationMachine):

    # lib/torch_trainer.py passes the unix timestamp of every rating (the calendar fields need the hour, not only the day)
    uses_timestamps = True

    # Parameters: as FactorizationMachine, the calendar fields appended after the four base fields
    # calendar_field_rows_of_hour   (hours of the plausible range, calendar fields)   buffer: lookup table, not trained

    def __init__(self, calendar_fields: tuple[str, ...], year_range: tuple[int, int], **base_field_arguments):

        # size of every requested calendar field; the year field covers the years of the training ratings
        first_year, last_year = year_range
        field_sizes = {**CALENDAR_FIELD_SIZES, "year": last_year - first_year + 1}
        super().__init__(**base_field_arguments, extra_field_sizes=tuple(field_sizes[name] for name in calendar_fields))

        # calendar field rows of every hour of the plausible timestamp range (moves to the GPU together with the model)
        # -> the calendar fields of a rating are one lookup by its hour since FIRST_PLAUSIBLE_TIMESTAMP
        self.register_buffer("calendar_field_rows_of_hour",
                             torch.as_tensor(calendar_field_rows_per_hour(calendar_fields, first_year, last_year)))

    # (B, 4 + calendar fields): the four base fields, then the calendar fields of the rating's hour
    def field_rows(self, user_rows, item_rows, timestamps):
        hour_rows = torch.div(timestamps - FIRST_PLAUSIBLE_TIMESTAMP, SECONDS_PER_HOUR, rounding_mode="floor")
        return torch.cat([super().field_rows(user_rows, item_rows), self.calendar_field_rows_of_hour[hour_rows]], dim=1)

    def forward(self, user_rows, item_rows, timestamps):
        return self.prediction(self.field_rows(user_rows, item_rows, timestamps) + self.field_offsets)

    # calendar features are shared by many ratings each -> left unregularised like the bucket features
    def l2_penalty(self, user_rows, item_rows, timestamps, user_penalty_weights, item_penalty_weights,
                   user_bias_regularization, item_bias_regularization, factor_regularization):
        return super().l2_penalty(user_rows, item_rows, user_penalty_weights, item_penalty_weights,
                                  user_bias_regularization, item_bias_regularization, factor_regularization)


def train_time_factorization_machine(task: RatingTask, config: TimeFactorizationMachineConfig) -> ModelResult:
    require_test_timestamps(task, "time-aware factorization machine")

    def build_model(training_ratings: RatingData) -> TimeFactorizationMachine:
        torch.manual_seed(config.seed)  # identical initial factors for the fit and the refit
        return TimeFactorizationMachine(calendar_fields=config.calendar_fields,
                                        year_range=first_and_last_year(training_ratings.timestamps),
                                        **base_field_arguments(training_ratings, task.n_user_rows, task.n_item_rows, config)).to(DEVICE)

    # fit with early stopping -> validation predictions -> optional refit on 100 % -> test predictions
    return fit_refit_and_predict(task, build_model, **training_options(config))


def store_time_factorization_machine(model_result: ModelResult, task_name: str):
    return save_model_result(model_result, task_name, "time_factorization_machine")


def load_time_factorization_machine(task_name: str) -> ModelResult:
    return load_model_result(task_name, "time_factorization_machine")
