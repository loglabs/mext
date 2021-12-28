from components import (
    load_data,
    clean_data,
    featurize_data,
    inference,
)
from datetime import timedelta, datetime
from multiprocessing import Process
from prometheus_client import start_http_server

from hawk import BinaryClassificationMetric
from mltrace import Task, Metric, clean_db


import numpy as np
import pandas as pd
import random
import requests
import sklearn
import string
import time


def generate_labels(n):
    return [
        "".join(random.choice(string.ascii_uppercase) for _ in range(10))
        for _ in range(n)
    ]


def accuracy_score(y_true, y_pred):
    # Round y_pred to the nearest integer
    y_pred = np.round(y_pred).astype(int)
    return sklearn.metrics.accuracy_score(y_true, y_pred)


# Clean prom DB
# result = "Service Unavailable"
# while result == "Service Unavailable":
#     time.sleep(5)
#     response = requests.post(
#         "http://example-prometheus:9090/api/v1/admin/tsdb/clean_tombstones",
#         data={},
#     )
#     result = response.text
#     print(f"Tried to delete prometheus data. Got response:{result}")

# Clean mltrace db
clean_db()
print("MLTrace data deleted.")


# Setup prometheus and mltrace metrics
task = Task("taxi_data")
task.registerMetric(Metric("accuracy_score", fn=accuracy_score))
prom_metric = BinaryClassificationMetric(
    "taxi_data",
    "Binary classification metric for tip prediction",
    ["output_id"],
)
accuracy_query = prom_metric.get_query_strings()["accuracy"]


def log_predictions_mltrace(predictions, identifiers):
    task.logOutputs(predictions, identifiers)


def log_predictions_prometheus(predictions, identifiers):
    prom_metric.log_pred_batch(predictions, identifiers)


def log_feedbacks(feedback, identifiers, start_date, end_date):
    # TODO(shreyashankar): add some lag here and run this in the background
    # sleep_time = np.random.normal(loc=3, scale=1, size=1)[0]
    # time.sleep(sleep_time)
    task.logFeedbacks(feedback, identifiers)
    prom_metric.log_true_batch(feedback, identifiers)
    print(
        f"Logged feedback for {len(feedback)} points in the range {start_date} to {end_date}"
    )


def run_predictions():
    start_date = datetime(2020, 3, 1)
    end_date = datetime(2020, 5, 31)
    mltrace_logging_times = []
    mltrace_metric_computation_times = []
    prometheus_logging_times = []
    prometheus_metric_computation_times = []
    start_dates = []
    end_dates = []
    num_points = []
    processes = []

    prev_dt = start_date
    for n in range(2, int((end_date - start_date).days) + 1, 2):
        curr_dt = start_date + timedelta(n)

        df = load_data(start_date=prev_dt, end_date=curr_dt)
        df = df.head(10000)

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
        print("Finished predictions.")

        # Log predictions and feedbacks
        identifiers = generate_labels(len(predictions))
        outputs = predictions["prediction"].to_list()
        feedbacks = predictions["high_tip_indicator"].astype("int").to_list()
        start = time.time()
        log_predictions_mltrace(
            outputs,
            identifiers,
        )
        # multiply by 2 to account for log_feedbacks, which will run
        # separately with a delay
        mltrace_logging_times.append((time.time() - start) * 2)
        start = time.time()
        log_predictions_prometheus(outputs, identifiers)
        prometheus_logging_times.append((time.time() - start) * 2)
        print(
            f"Logged predictions for {len(outputs)} points in the range {prev_dt} to {curr_dt}"
        )

        # Log feedback
        # p = Process(
        #     target=log_feedbacks,
        #     args=(feedbacks, identifiers, prev_dt, curr_dt),
        # )
        # p.start()
        # processes.append(p)
        log_feedbacks(feedbacks, identifiers, prev_dt, curr_dt)

        # Print rolling score computed by mltrace
        start = time.time()
        rolling_accuracy = task.computeMetrics()
        mltrace_metric_computation_times.append(time.time() - start)
        print(f"Rolling accuracy from mltrace: {rolling_accuracy}")

        # TODO (shreyashankar): add prometheus metric computation time
        start = time.time()
        response = requests.get(
            "http://example-prometheus:9090/api/v1/query",
            params={"query": accuracy_query},
        )
        prometheus_metric_computation_times.append(time.time() - start)
        print(f"Prometheus accuracy: {response.text}")

        prev_dt = curr_dt

    print("Exited loop of inference.")

    # Wait for all threads to finish
    # for p in processes:
    #     p.join()

    timing_df = pd.DataFrame(
        {
            "start_date": start_dates,
            "end_date": end_dates,
            "mltrace_logging_times": mltrace_logging_times,
            "mltrace_metric_computation_times": mltrace_metric_computation_times,
            "prometheus_logging_times": prometheus_logging_times,
            "prometheus_metric_computation_times": prometheus_metric_computation_times,
            "num_points": num_points,
        }
    )

    print(timing_df.head(60))
    timing_df.to_csv("timing_df.csv", index=False)


if __name__ == "__main__":
    # Start http server for Prometheus
    start_http_server(1000)

    print("Starting inference.")

    run_predictions()
