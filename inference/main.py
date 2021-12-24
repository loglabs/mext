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


def run_predictions():
    start_date = datetime(2020, 2, 1)
    end_date = datetime(2020, 5, 31)

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
        task.logOutputs(
            predictions["prediction"].to_list(),
            labels,
        )
        task.logFeedbacks(
            predictions["high_tip_indicator"].astype("int").to_list(),
            labels,
        )

        # Print rolling score
        rolling_f1_score = task.computeMetric(f1_score)
        print(f"F1 score: {rolling_f1_score}")

        prev_dt = curr_dt

    print("Exited loop of inference.")


# @app.route("/", methods=["GET"])
# def index():
#     return "Hello, world!"


if __name__ == "__main__":
    # app.run(debug=True, host="0.0.0.0", port=1000)
    run_predictions()
