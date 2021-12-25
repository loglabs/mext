# hawk

This project aims to benchmark open-source ML monitoring tools. Tools in the benchmark include:

* Prometheus
* `mltrace` Monitoring

## ML Task and Pipeline Architecture

### Data source

We use the [NYC taxicab data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page), which has been migrated from a bucket of flat files to an AWS RDS instance via the [TTB project](https://github.com/loglabs/ttb). 

### Feedback lag

To simulate lag that a real-world system might experience, we inject a delay sampled from a Gaussian distribution. (TODO: shreyashankar)

## Pipeline

The ML task is to predict whether a passenger will give a taxi driver a sizeable tip (10% or more). Pipeline components are defined in the `components` folder and are called in `train.py` to train a model on Jan 2020 data.  The inference code is in `inference/main.py`, which runs the model on data in 2-day increments from Feb 1 2020 to May 31 2020.

## Prometheus Extension

## `mltrace` Monitoring Extension

Functions to log outputs and feedback to `mltrace` are used in `inference/main.py`.

## Experiments

| Method      | Number of points | Logging Time | Query Time |
| ----------- | ----------- | ----------- | ----------- |
| `mltrace` Monitoring      | Title       | |
| Prometheus   |         | |
