import os
import logging
import pandas as pd
from ttb import Dataset
import typing

from components.defs import *
from datetime import datetime
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

ds = Dataset("taxi_data", cutoff_date=datetime(2021, 1, 1), backend="pandas")


def load_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Format: %Y-%m-%d
    This function loads the trip data corresponding to the specified
    dates. The data must be stored in the "data" folder and can
    be populated using the download.sh script.
    """

    return ds.load(start_date, end_date)


# @Cleaning().run(auto_log=True)
def clean_data(
    df: pd.DataFrame, start_date: str = None, end_date: str = None
) -> pd.DataFrame:
    """
    This function removes rows with negligible fare amounts and out of bounds of the start and end dates.
    Args:
        df: pd dataframe representing data
        start_date (optional): minimum date in the resulting dataframe
        end_date (optional): maximum date in the resulting dataframe (not inclusive)
    Returns:
        pd: DataFrame representing the cleaned dataframe
    """
    df = df[df.fare_amount > 5]  # throw out neglibible fare amounts
    if start_date:
        df = df[df.tpep_dropoff_datetime.dt.strftime("%Y-%m-%d") >= start_date]
    if end_date:
        df = df[df.tpep_dropoff_datetime.dt.strftime("%Y-%m-%d") < end_date]

    clean_df = df.reset_index(drop=True)
    return clean_df


# @Featuregen().run(input_vars={"df": "customer_label"}, auto_log=True)
def featurize_data(
    df: pd.DataFrame, tip_fraction: float = 0.1, imputation_value: float = -1.0
) -> pd.DataFrame:
    """
    This function constructs features from the dataframe.
    """
    # Compute labels for mltrace
    # customer_label = list(("user_" + df["pulocationid"].astype(str)).unique())

    # Compute pickup features
    pickup_weekday = df.tpep_pickup_datetime.dt.weekday
    pickup_hour = df.tpep_pickup_datetime.dt.hour
    pickup_minute = df.tpep_pickup_datetime.dt.minute
    work_hours = (
        (pickup_weekday >= 0)
        & (pickup_weekday <= 4)
        & (pickup_hour >= 8)
        & (pickup_hour <= 18)
    )

    # Compute time and speed features
    trip_time = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.seconds
    trip_speed = df.trip_distance / (trip_time + 1e7)

    # Compute label
    tip_fraction_col = df.tip_amount / df.fare_amount

    # Join all features, identifier, and label
    features_df = pd.DataFrame(
        {
            "tpep_pickup_datetime": df.tpep_pickup_datetime,
            "pickup_weekday": pickup_weekday,
            "pickup_hour": pickup_hour,
            "pickup_minute": pickup_minute,
            "work_hours": work_hours,
            "trip_time": trip_time,
            "trip_speed": trip_speed,
            "trip_distance": df.trip_distance,
            "passenger_count": df.passenger_count,
            "congestion_surcharge": df.congestion_surcharge,
            "loc_code_diffs": (df.dolocationid - df.pulocationid).abs(),
            "PULocationID": df.pulocationid,
            "DOLocationID": df.dolocationid,
            "RatecodeID": df.ratecodeid,
            "VendorID": df.vendorid,
            "tip_amount": df.tip_amount,
            "fare_amount": df.fare_amount,
            "tip_fraction": tip_fraction_col,
            "high_tip_indicator": tip_fraction_col > tip_fraction,
        }
    ).fillna(imputation_value)

    return features_df


# @TrainTestSplit().run(auto_log=True)
def train_test_split(
    df: pd.DataFrame,
) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function splits the dataframe into train and test.
    """
    # Split into train and test
    date_column = "tpep_pickup_datetime"
    label_column = "high_tip_indicator"
    df = df.sort_values(by=date_column, ascending=True)
    train_df, test_df = (
        df.iloc[: int(len(df) * 0.8)],
        df.iloc[int(len(df) * 0.8) :],
    )

    return train_df, test_df


# Score model
def score(df, model, feature_columns, label_column) -> pd.DataFrame:
    rounded_preds = model.predict_proba(df[feature_columns].values)[
        :, 1
    ].round()
    return {
        "accuracy_score": accuracy_score(
            df[label_column].values, rounded_preds
        ),
        "f1_score": f1_score(df[label_column].values, rounded_preds),
        "precision_score": precision_score(
            df[label_column].values, rounded_preds
        ),
        "recall_score": recall_score(df[label_column].values, rounded_preds),
    }


# @Training().run(auto_log=True)
def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: typing.List[str],
    label_column: str,
) -> None:
    """
    This function runs training on the dataframe with the given
    feature and label columns. The model is saved locally
    to "model.joblib".
    """

    params = {"max_depth": 4, "n_estimators": 10, "random_state": 42}

    # Create and train model
    model = RandomForestClassifier(**params)
    model.fit(train_df[feature_columns].values, train_df[label_column].values)

    # Print scores
    train_scores = score(train_df, model, feature_columns, label_column)
    test_scores = score(test_df, model, feature_columns, label_column)
    logging.info("Train scores:")
    logging.info(train_scores)
    logging.info("Test scores:")
    logging.info(test_scores)

    # Print feature importances
    feature_importances = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )
    logging.info(feature_importances)

    # Save model
    dump(model, "model.joblib")


# @Inference().run(auto_log=True, staleness_threshold=30)
def inference(
    features_df: pd.DataFrame,
    feature_columns: typing.List[str],
    label_column: str,
    model=load("model.joblib") if os.path.exists("model.joblib") else None,
):
    """
    This function runs inference on the dataframe.
    """
    if not model:
        raise ValueError("Please run this pipeline in training mode first!")

    # Predict
    predictions = model.predict_proba(features_df[feature_columns].values)[
        :, 1
    ]
    scores = score(features_df, model, feature_columns, label_column)
    predictions_df = features_df
    predictions_df["prediction"] = predictions

    return predictions_df, scores
