from prometheus_client import start_http_server, Gauge
from hawk import BinaryClassificationMetric
import random
import string
import time

# Create a metric to track predictions.
# pred = Gauge("prediction_float", "y_pred", labelnames=["output_id"])
# true = Gauge("true_float", "y_true", labelnames=["output_id"])
metric = BinaryClassificationMetric(
    "random", "Testing binary classification metric", ["output_id"]
)
print(metric.get_query_strings())
N = 10

# Decorate function with metric.
def process_request(t):
    """Create and log prediction and label."""
    time.sleep(t)

    output_id = "".join(
        random.choice(string.ascii_uppercase) for _ in range(N)
    )
    metric.log_pred(t, output_id)
    metric.log_true(random.randint(0, 1), output_id)


if __name__ == "__main__":
    # Start up the server to expose the metrics.
    start_http_server(1000)
    # Generate some requests.
    while True:
        process_request(random.random())
