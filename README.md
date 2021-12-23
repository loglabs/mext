# hawk

This project aims to benchmark open-source ML monitoring tools. Tools in the benchmark include:

* Prometheus
* `mltrace` Monitoring

## ML Task and Pipeline Architecture

### Data source

We use the [NYC taxicab data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page), which has been migrated from a bucket of flat files to an AWS RDS instance. 

### Feedback lag

To simulate lag that a real-world system might experience, we inject a delay sampled from a Gaussian distribution.

## Prometheus Extension

## `mltrace` Monitoring Extension

## Experiments
