from components import (
    load_data,
    clean_data,
    featurize_data,
    inference,
)
from datetime import timedelta, datetime

from mltrace import Task


import numpy as np
import pandas as pd
import random
import sklearn
import string
import time

task = Task("taxi_data")


def generate_labels(n):
    return [
        "".join(random.choice(string.ascii_uppercase) for _ in range(10))
        for _ in range(n)
    ]


def f1_score(y_true, y_pred):
    # Round y_pred to the nearest integer
    y_pred = np.round(y_pred).astype(int)
    return sklearn.metrics.f1_score(y_true, y_pred)


def log_predictions(predictions, labels):
    task.logOutputs(predictions, labels)


def log_feedbacks(feedback, labels):
    # TODO(shreyashankar): add some lag here and run this in the background
    task.logFeedbacks(feedback, labels)


def run_predictions():
    start_date = datetime(2020, 2, 1)
    end_date = datetime(2020, 5, 31)
    logging_times = []
    metric_computation_times = []
    start_dates = []
    end_dates = []
    num_points = []

    prev_dt = start_date
    for n in range(2, int((end_date - start_date).days) + 1, 2):
        curr_dt = start_date + timedelta(n)

        df = load_data(start_date=prev_dt, end_date=curr_dt)

        clean_df = clean_data(
            df, prev_dt.strftime("%Y-%m-%d"), curr_dt.strftime("%Y-%m-%d")
        )
        if len(clean_df) == 0:
            continue
        print(
            f"Running predictions for {len(clean_df)} points in the range {prev_dt} to {curr_dt}"
        )

        start_dates.append(prev_dt.strftime("%Y-%m-%d"))
        end_dates.append(curr_dt.strftime("%Y-%m-%d"))
        num_points.append(len(clean_df))
        features_df = featurize_data(clean_df)

        feature_columns = [
            "pickup_weekday",
            "pickup_hour",
            "pickup_minute",
            "work_hours",
            "passenger_count",
            "trip_distance",
            "RatecodeID",
            "congestion_surcharge",
            "loc_code_diffs",
        ]
        label_column = "high_tip_indicator"
        predictions, _ = inference(features_df, feature_columns, label_column)

        # Log scores
        labels = generate_labels(len(predictions))
        outputs = predictions["prediction"].to_list()
        feedbacks = predictions["high_tip_indicator"].astype("int").to_list()
        start = time.time()
        log_predictions(
            outputs,
            labels,
        )
        logging_time = (
            time.time() - start
        ) * 2  # multiply by 2 to account for log_feedbacks, which will run
        # separately with a delay
        logging_times.append(logging_time)

        log_feedbacks(
            feedbacks,
            labels,
        )

        # Print rolling score
        start = time.time()
        rolling_f1_score = task.computeMetric(f1_score)
        metric_computation_time = time.time() - start
        metric_computation_times.append(metric_computation_time)
        print(f"Rolling F1 score: {rolling_f1_score}")

        prev_dt = curr_dt

    print("Exited loop of inference.")

    timing_df = pd.DataFrame(
        {
            "start_date": start_dates,
            "end_date": end_dates,
            "logging_times": logging_times,
            "metric_computation_times": metric_computation_times,
            "num_points": num_points,
        }
    )

    print(timing_df.head(60))


if __name__ == "__main__":
    run_predictions()
